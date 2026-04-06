/*
 * Copyright (c) 2015 - 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "SimpleInferencing.h"
#include "DirectoryHelper.h"
#include <donut/core/log.h>
#include <nvrhi/utils.h>
#include <format>
#include <iostream>
#include <random>
#include <numeric>

using namespace donut;
using namespace donut::math;

extern const char* g_windowTitle;

SimpleInferencing::SimpleInferencing(app::DeviceManager* deviceManager, UIData* uiParams, const std::string& modelPath) 
    : IRenderPass(deviceManager), m_userInterfaceParameters(uiParams), m_modelPath(modelPath)
{
}

bool SimpleInferencing::Init()
{
    m_networkUtils = std::make_shared<rtxns::NetworkUtilities>(GetDevice());
    rtxns::HostNetwork net(m_networkUtils);
    if (!net.InitialiseFromFile(GetLocalPath("assets/data").string() + std::string("/disney.ns.bin")))
    {
        log::debug("Loaded Neural Shading Network from file failed.");
        return false;
    }

    assert(net.GetNetworkLayout().networkLayers.size() == 4);

    rtxns::NetworkLayout deviceNetworkLayout = m_networkUtils->GetNewMatrixLayout(net.GetNetworkLayout(), rtxns::MatrixLayout::InferencingOptimal);

    m_weightOffsets = dm::uint4(deviceNetworkLayout.networkLayers[0].weightOffset, deviceNetworkLayout.networkLayers[1].weightOffset,
                                deviceNetworkLayout.networkLayers[2].weightOffset, deviceNetworkLayout.networkLayers[3].weightOffset);

    m_biasOffsets = dm::uint4(deviceNetworkLayout.networkLayers[0].biasOffset, deviceNetworkLayout.networkLayers[1].biasOffset, deviceNetworkLayout.networkLayers[2].biasOffset,
                              deviceNetworkLayout.networkLayers[3].biasOffset);

    std::filesystem::path frameworkShaderPath = app::GetDirectoryWithExecutable() / "shaders/framework" / app::GetShaderTypeName(GetDevice()->getGraphicsAPI());
    std::filesystem::path appShaderPath = app::GetDirectoryWithExecutable() / "shaders/SimpleInferencing" / app::GetShaderTypeName(GetDevice()->getGraphicsAPI());

    std::shared_ptr<vfs::RootFileSystem> rootFS = std::make_shared<vfs::RootFileSystem>();
    rootFS->mount("/shaders/donut", frameworkShaderPath);
    rootFS->mount("/shaders/app", appShaderPath);

    m_shaderFactory = std::make_shared<engine::ShaderFactory>(GetDevice(), rootFS, "/shaders");
    m_vertexShader = m_shaderFactory->CreateShader("app/SimpleInferencing", "main_vs", nullptr, nvrhi::ShaderType::Vertex);
    m_pixelShader = m_shaderFactory->CreateShader("app/SimpleInferencing", "main_ps", nullptr, nvrhi::ShaderType::Pixel);

    if (!m_vertexShader || !m_pixelShader)
    {
        return false;
    }

    m_rootFS = rootFS;
    auto nativeFS = std::make_shared<vfs::NativeFileSystem>();
    m_rootFS->mount("/", nativeFS);

    m_textureCache = std::make_shared<engine::TextureCache>(GetDevice(), nativeFS, nullptr);
    m_commonPasses = std::make_shared<engine::CommonRenderPasses>(GetDevice(), m_shaderFactory);
    m_lightProbePass = std::make_shared<donut::render::LightProbeProcessingPass>(GetDevice(), m_shaderFactory, m_commonPasses);

    m_skyboxRenderer = std::make_unique<SkyboxRenderer>(GetDevice(), m_shaderFactory, m_commonPasses, m_lightProbePass, m_textureCache);
    if (!m_skyboxRenderer->Init())
        return false;

    LoadSkyboxTexture(GetLocalPath("assets/environment_maps/14-Hamarikyu_Bridge_B_3k.hdr").string());

    if (!m_modelPath.empty())
    {
        LoadModel(m_modelPath);
    }
    else
    {
        LoadModel(GetLocalPath("assets/3dmodel/sphere.obj").string());
    }

    const auto& params = net.GetNetworkParams();

    nvrhi::BufferDesc bufferDesc;
    bufferDesc.byteSize = params.size();
    bufferDesc.debugName = "MLPParamsUploadBuffer";
    bufferDesc.initialState = nvrhi::ResourceStates::CopyDest;
    bufferDesc.keepInitialState = true;
    m_mlpHostBuffer = GetDevice()->createBuffer(bufferDesc);

    bufferDesc.byteSize = deviceNetworkLayout.networkByteSize;
    bufferDesc.canHaveRawViews = true;
    bufferDesc.canHaveUAVs = true;
    bufferDesc.debugName = "MLPParamsByteAddressBuffer";
    bufferDesc.initialState = nvrhi::ResourceStates::UnorderedAccess;
    m_mlpDeviceBuffer = GetDevice()->createBuffer(bufferDesc);

    m_commandList = GetDevice()->createCommandList();
    m_commandList->open();

    m_commandList->writeBuffer(m_mlpHostBuffer, params.data(), params.size());

    m_networkUtils->ConvertWeights(net.GetNetworkLayout(), deviceNetworkLayout, m_mlpHostBuffer, 0, m_mlpDeviceBuffer, 0, GetDevice(), m_commandList);

    m_commandList->setBufferState(m_mlpDeviceBuffer, nvrhi::ResourceStates::ShaderResource);
    m_commandList->commitBarriers();

    m_commandList->close();
    GetDevice()->executeCommandList(m_commandList);

    if (!m_constantBuffer) {
        m_constantBuffer = GetDevice()->createBuffer(
            nvrhi::utils::CreateStaticConstantBufferDesc(sizeof(NeuralConstants), "ConstantBuffer").setInitialState(nvrhi::ResourceStates::ConstantBuffer).setKeepInitialState(true));
    }

    nvrhi::BindingSetDesc bindingSetDesc;
    bindingSetDesc.bindings = {
        nvrhi::BindingSetItem::ConstantBuffer(0, m_constantBuffer),
        nvrhi::BindingSetItem::RawBuffer_SRV(0, m_mlpDeviceBuffer)
    };

    if (!nvrhi::utils::CreateBindingSetAndLayout(GetDevice(), nvrhi::ShaderType::All, 0, bindingSetDesc, m_bindingLayout, m_bindingSet, true))
    {
        log::error("Couldn't create the binding set or layout");
        return false;
    }

    m_neuralTimer = GetDevice()->createTimerQuery();

    // ===== Initialize unified MLP (SSS) via NeuralMLP =====
    {
        NeuralMLPConfig cfg;
        cfg.inputNeurons = UNIFIED_INPUT_NEURONS;
        cfg.hiddenNeurons = UNIFIED_HIDDEN_NEURONS;
        cfg.outputNeurons = UNIFIED_OUTPUT_NEURONS;
        cfg.numHiddenLayers = UNIFIED_NUM_HIDDEN_LAYERS;
        cfg.numTransitions = UNIFIED_NUM_TRANSITIONS;
        cfg.numTransitionsAlign4 = UNIFIED_NUM_TRANSITIONS_ALIGN4;
        cfg.batchSize = UNIFIED_BATCH_SIZE;
        cfg.batchCount = UNIFIED_BATCH_COUNT;
        cfg.threadsPerGroup = UNIFIED_THREADS_PER_GROUP;
        cfg.threadsPerGroupOpt = UNIFIED_THREADS_PER_GROUP_OPT;
        cfg.learningRate = UNIFIED_LEARNING_RATE;
        cfg.lossScale = UNIFIED_LOSS_SCALE;
        cfg.trainingShaderName = "app/UnifiedTraining";
        cfg.optimizerShaderName = "app/UnifiedOptimizer";
        cfg.inferBindingSet = 3;
        cfg.debugPrefix = "Unified";
        cfg.trainingCBSize = sizeof(UnifiedTrainingConstants);

        // Try loading pre-trained weights from JSON
        auto preloaded = std::make_unique<rtxns::HostNetwork>(m_networkUtils);
        auto nativeFS = std::make_shared<donut::vfs::NativeFileSystem>();
        if (preloaded->InitialiseFromJson(*nativeFS, "c:/tmp/jade_sss_data/jade_sss_weights.json"))
        {
            m_unifiedMLP.InitFromNetwork(GetDevice(), m_networkUtils, m_shaderFactory, cfg, std::move(preloaded));
        }
        else
        {
            log::warning("Failed to load unified MLP from json, falling back to random init.");
            m_unifiedMLP.Init(GetDevice(), m_networkUtils, m_shaderFactory, cfg);
        }
        m_unifiedMLP.onPipelineInvalidated = [this]() { m_pipeline = nullptr; };
    }

    // ===== Initialize IBL Sampler MLP =====
    {
        NeuralMLPConfig cfg;
        cfg.inputNeurons = IBL_INPUT_NEURONS;
        cfg.hiddenNeurons = IBL_HIDDEN_NEURONS;
        cfg.outputNeurons = IBL_OUTPUT_NEURONS;
        cfg.numHiddenLayers = IBL_NUM_HIDDEN_LAYERS;
        cfg.numTransitions = IBL_NUM_TRANSITIONS;
        cfg.numTransitionsAlign4 = IBL_NUM_TRANSITIONS_ALIGN4;
        cfg.batchSize = IBL_BATCH_SIZE;
        cfg.batchCount = IBL_BATCH_COUNT;
        cfg.threadsPerGroup = IBL_THREADS_PER_GROUP;
        cfg.threadsPerGroupOpt = IBL_THREADS_PER_GROUP_OPT;
        cfg.learningRate = IBL_LEARNING_RATE;
        cfg.lossScale = IBL_LOSS_SCALE;
        cfg.trainingShaderName = "app/IBLTraining";
        cfg.optimizerShaderName = "app/IBLOptimizer";
        cfg.inferBindingSet = 4;
        cfg.debugPrefix = "IBL";
        cfg.trainingCBSize = sizeof(IBLTrainingConstants);

        m_iblMLP.Init(GetDevice(), m_networkUtils, m_shaderFactory, cfg);
        m_iblMLP.onPipelineInvalidated = [this]() { m_pipeline = nullptr; };
    }

    return true;
}

std::shared_ptr<engine::ShaderFactory> SimpleInferencing::GetShaderFactory() const
{
    return m_shaderFactory;
}

bool SimpleInferencing::LoadModel(const std::string& path)
{
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    std::vector<MaterialParams> mats;
    std::vector<uint32_t> matIndices;
    std::vector<GltfTextureData> textures;

    bool loaded = false;
    if (path.length() >= 4 && path.substr(path.length() - 4) == ".obj")
    {
        loaded = LoadOBJ(path, vertices, indices, mats, matIndices);
        m_hasPerVertexMaterials = false;
    }
    else
    {
        loaded = LoadGLTF(path, vertices, indices, mats, textures);
        m_hasPerVertexMaterials = !textures.empty() || mats.size() > 1;
    }

    if (!loaded)
    {
        log::error("Failed to load model: %s", path.c_str());
        return false;
    }

    m_modelPath = path;
    UpdateGeometryBuffers(vertices, indices);
    CreateMaterialResources(mats, textures);

    // Log SSS texture status
    for (size_t i = 0; i < mats.size(); ++i)
    {
        const auto& m = mats[i];
        log::info("Material %zu SSS textures: thickness=%d, ao=%d, curvature=%d",
            i, m.thicknessTexIdx, m.occlusionTexIdx, m.curvatureTexIdx);
    }

    // SSS-GGX-MLP: Initialize UI sliders from the loaded model's material properties.
    // This ensures the MLP receives the correct roughness/metallic/specular on first render,
    // and the user can adjust them in real-time via the sliders.
    if (!mats.empty() && m_userInterfaceParameters)
    {
        const auto& mat0 = mats[0];
        m_userInterfaceParameters->roughness = mat0.roughness;
        m_userInterfaceParameters->metallic = mat0.metallic;
        m_userInterfaceParameters->specular = mat0.specular;
        log::info("UI material params initialized from GLB: roughness=%.2f, metallic=%.2f, specular=%.2f",
            mat0.roughness, mat0.metallic, mat0.specular);
    }

    return true;
}

void SimpleInferencing::LoadSkyboxTexture(const std::string& path)
{
    if (m_skyboxRenderer)
    {
        m_skyboxRenderer->LoadSkyboxTexture(path);
    }
}

void SimpleInferencing::CreateMaterialResources(const std::vector<MaterialParams>& materials, const std::vector<GltfTextureData>& textures)
{
    m_materials = materials;
    m_materialCount = (uint32_t)materials.size();
    m_textureCount = (uint32_t)textures.size();

    // Reset previous resources
    m_materialBindingSet = nullptr;
    m_materialBindingLayout = nullptr;
    m_materialBuffer = nullptr;
    m_materialTextures.clear();
    m_pipeline = nullptr; // Force pipeline rebuild

    auto commandList = GetDevice()->createCommandList();
    commandList->open();

    // Create material structured buffer
    {
        nvrhi::BufferDesc desc;
        desc.byteSize = std::max((size_t)1, materials.size()) * sizeof(MaterialParams);
        desc.structStride = sizeof(MaterialParams);
        desc.debugName = "MaterialParamsBuffer";
        desc.initialState = nvrhi::ResourceStates::CopyDest;
        desc.keepInitialState = false;
        desc.canHaveRawViews = false;
        m_materialBuffer = GetDevice()->createBuffer(desc);

        commandList->beginTrackingBufferState(m_materialBuffer, nvrhi::ResourceStates::CopyDest);
        commandList->writeBuffer(m_materialBuffer, materials.data(), materials.size() * sizeof(MaterialParams));
        commandList->setPermanentBufferState(m_materialBuffer, nvrhi::ResourceStates::ShaderResource);
    }

    // Create material sampler
    if (!m_materialSampler)
    {
        nvrhi::SamplerDesc sd;
        sd.setAllAddressModes(nvrhi::SamplerAddressMode::Repeat);
        sd.setAllFilters(true);
        m_materialSampler = GetDevice()->createSampler(sd);
    }

    // Upload individual textures - store debugName strings to keep them alive
    std::vector<std::string> texDebugNames(textures.size());
    m_materialTextures.resize(textures.size());
    for (size_t i = 0; i < textures.size(); ++i)
    {
        const auto& tex = textures[i];

        texDebugNames[i] = tex.name.empty() ? ("MaterialTex_" + std::to_string(i)) : tex.name;

        nvrhi::TextureDesc texDesc;
        texDesc.width = tex.width;
        texDesc.height = tex.height;
        texDesc.format = nvrhi::Format::RGBA8_UNORM;
        texDesc.dimension = nvrhi::TextureDimension::Texture2D;
        texDesc.isRenderTarget = false;
        texDesc.mipLevels = 1;
        texDesc.debugName = texDebugNames[i].c_str();
        texDesc.initialState = nvrhi::ResourceStates::CopyDest;
        texDesc.keepInitialState = false;

        m_materialTextures[i] = GetDevice()->createTexture(texDesc);
        commandList->beginTrackingTextureState(m_materialTextures[i], nvrhi::TextureSubresourceSet(0, 1, 0, 1), nvrhi::ResourceStates::CopyDest);
        commandList->writeTexture(m_materialTextures[i], 0, 0, tex.pixels.data(), tex.width * 4);
        commandList->setPermanentTextureState(m_materialTextures[i], nvrhi::ResourceStates::ShaderResource);
    }

    commandList->commitBarriers();
    commandList->close();
    GetDevice()->executeCommandList(commandList);

    // Create binding layout for material set
    // registerSpace=2 maps to Vulkan descriptor set 2 via registerSpaceIsDescriptorSet
    nvrhi::BindingLayoutDesc matLayoutDesc;
    matLayoutDesc.visibility = nvrhi::ShaderType::Pixel;
    matLayoutDesc.registerSpace = 2;
    matLayoutDesc.registerSpaceIsDescriptorSet = true;
    matLayoutDesc.bindings.push_back(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(0)); // t0: material buffer

    // t1: texture array (descriptor array of size N)
    uint32_t texSlotCount = std::max(1u, (uint32_t)textures.size());
    auto texArrayItem = nvrhi::BindingLayoutItem::Texture_SRV(1);
    texArrayItem.size = (uint16_t)texSlotCount;
    matLayoutDesc.bindings.push_back(texArrayItem);

    matLayoutDesc.bindings.push_back(nvrhi::BindingLayoutItem::Sampler(0)); // s0: sampler
    m_materialBindingLayout = GetDevice()->createBindingLayout(matLayoutDesc);

    // Create binding set
    nvrhi::BindingSetDesc matSetDesc;
    matSetDesc.bindings.push_back(nvrhi::BindingSetItem::StructuredBuffer_SRV(0, m_materialBuffer));

    if (textures.empty())
    {
        // Bind a dummy 1x1 white texture to avoid null binding
        nvrhi::TextureDesc dummyDesc;
        dummyDesc.width = 1;
        dummyDesc.height = 1;
        dummyDesc.format = nvrhi::Format::RGBA8_UNORM;
        dummyDesc.debugName = "DummyWhiteTex";
        dummyDesc.initialState = nvrhi::ResourceStates::CopyDest;
        dummyDesc.keepInitialState = false;

        auto dummyTex = GetDevice()->createTexture(dummyDesc);
        uint8_t white[4] = { 255, 255, 255, 255 };
        auto cmdList2 = GetDevice()->createCommandList();
        cmdList2->open();
        cmdList2->beginTrackingTextureState(dummyTex, nvrhi::TextureSubresourceSet(0, 1, 0, 1), nvrhi::ResourceStates::CopyDest);
        cmdList2->writeTexture(dummyTex, 0, 0, white, 4);
        cmdList2->setPermanentTextureState(dummyTex, nvrhi::ResourceStates::ShaderResource);
        cmdList2->commitBarriers();
        cmdList2->close();
        GetDevice()->executeCommandList(cmdList2);

        matSetDesc.bindings.push_back(nvrhi::BindingSetItem::Texture_SRV(1, dummyTex));
        m_materialTextures.push_back(dummyTex); // Keep alive
    }
    else
    {
        // Each texture is an element in the descriptor array at slot 1
        for (uint32_t t = 0; t < (uint32_t)textures.size(); ++t)
        {
            auto item = nvrhi::BindingSetItem::Texture_SRV(1, m_materialTextures[t]);
            item.setArrayElement(t);
            matSetDesc.bindings.push_back(item);
        }
    }

    matSetDesc.bindings.push_back(nvrhi::BindingSetItem::Sampler(0, m_materialSampler));
    m_materialBindingSet = GetDevice()->createBindingSet(matSetDesc, m_materialBindingLayout);
}

void SimpleInferencing::UpdateGeometryBuffers(const std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices)
{
    m_indicesNum = (int)indices.size();

    if (!m_inputLayout) {
        nvrhi::VertexAttributeDesc attributes[] = {
            nvrhi::VertexAttributeDesc().setName("POSITION").setFormat(nvrhi::Format::RGB32_FLOAT).setOffset(offsetof(Vertex, position)).setBufferIndex(0).setElementStride(sizeof(Vertex)),
            nvrhi::VertexAttributeDesc().setName("NORMAL").setFormat(nvrhi::Format::RGB32_FLOAT).setOffset(offsetof(Vertex, normal)).setBufferIndex(0).setElementStride(sizeof(Vertex)),
            nvrhi::VertexAttributeDesc().setName("TEXCOORD").setFormat(nvrhi::Format::RG32_FLOAT).setOffset(offsetof(Vertex, uv)).setBufferIndex(0).setElementStride(sizeof(Vertex)),
            nvrhi::VertexAttributeDesc().setName("TANGENT").setFormat(nvrhi::Format::RGBA32_FLOAT).setOffset(offsetof(Vertex, tangent)).setBufferIndex(0).setElementStride(sizeof(Vertex)),
            nvrhi::VertexAttributeDesc().setName("MATERIAL_IDX").setFormat(nvrhi::Format::R32_UINT).setOffset(offsetof(Vertex, materialIndex)).setBufferIndex(0).setElementStride(sizeof(Vertex)),
        };
        m_inputLayout = GetDevice()->createInputLayout(attributes, uint32_t(std::size(attributes)), m_vertexShader);
    }

    m_pipeline = nullptr; // Force pipeline rebuild with new layout

    auto commandList = GetDevice()->createCommandList();
    commandList->open();

    nvrhi::BufferDesc vertexBufferDesc;
    vertexBufferDesc.byteSize = vertices.size() * sizeof(vertices[0]);
    vertexBufferDesc.isVertexBuffer = true;
    vertexBufferDesc.debugName = "VertexBuffer";
    vertexBufferDesc.initialState = nvrhi::ResourceStates::CopyDest;
    m_vertexBuffer = GetDevice()->createBuffer(vertexBufferDesc);

    commandList->beginTrackingBufferState(m_vertexBuffer, nvrhi::ResourceStates::CopyDest);
    commandList->writeBuffer(m_vertexBuffer, vertices.data(), vertices.size() * sizeof(vertices[0]));
    commandList->setPermanentBufferState(m_vertexBuffer, nvrhi::ResourceStates::VertexBuffer);

    nvrhi::BufferDesc indexBufferDesc;
    indexBufferDesc.byteSize = indices.size() * sizeof(indices[0]);
    indexBufferDesc.isIndexBuffer = true;
    indexBufferDesc.debugName = "IndexBuffer";
    indexBufferDesc.initialState = nvrhi::ResourceStates::CopyDest;
    m_indexBuffer = GetDevice()->createBuffer(indexBufferDesc);

    commandList->beginTrackingBufferState(m_indexBuffer, nvrhi::ResourceStates::CopyDest);
    commandList->writeBuffer(m_indexBuffer, indices.data(), indices.size() * sizeof(indices[0]));
    commandList->setPermanentBufferState(m_indexBuffer, nvrhi::ResourceStates::IndexBuffer);

    commandList->close();
    GetDevice()->executeCommandList(commandList);
}

bool SimpleInferencing::KeyboardUpdate(int key, int scancode, int action, int mods)
{
    return m_camera.KeyboardUpdate(key, scancode, action, mods);
}

bool SimpleInferencing::MousePosUpdate(double xpos, double ypos)
{
    return m_camera.MousePosUpdate(xpos, ypos);
}

bool SimpleInferencing::MouseButtonUpdate(int button, int action, int mods)
{
    return m_camera.MouseButtonUpdate(button, action, mods);
}

bool SimpleInferencing::MouseScrollUpdate(double xoffset, double yoffset)
{
    return m_camera.MouseScrollUpdate(xoffset, yoffset);
}

void SimpleInferencing::Animate(float seconds)
{
    m_camera.Animate(seconds);

    auto t = int(GetDevice()->getTimerQueryTime(m_neuralTimer) * 1000000);
    if (t != 0)
    {
        if (m_unifiedMLP.IsTrainingActive())
            m_extraStatus = std::format(" - Neural - {:3d}us - Training epoch {} loss {:.4f}", t, m_unifiedMLP.GetEpoch(), 0.0f);
        else if (m_iblMLP.IsTrainingActive())
            m_extraStatus = std::format(" - IBL MLP - {:3d}us - Training epoch {}", t, m_iblMLP.GetEpoch());
        else if (m_userInterfaceParameters->enableNeuralSSS && m_unifiedMLP.IsReady())
            m_extraStatus = std::format(" - Unified MLP - {:3d}us", t);
        else
            m_extraStatus = std::format(" - Neural - {:3d}us", t);
    }

    GetDeviceManager()->SetInformativeWindowTitle(g_windowTitle, true, m_extraStatus.c_str());
}

void SimpleInferencing::BackBufferResizing()
{
    m_pipeline = nullptr;
    if (m_skyboxRenderer)
        m_skyboxRenderer->BackBufferResizing();
}

void SimpleInferencing::Render(nvrhi::IFramebuffer* framebuffer)
{
    const nvrhi::FramebufferInfoEx& fbinfo = framebuffer->getFramebufferInfo();
    const float width = float(fbinfo.width);
    const float height = float(fbinfo.height);
    const float left = 0;
    const float top = 0;

    bool updateStat = GetDeviceManager()->GetCurrentBackBufferIndex() % 100 == 0;

    if (!m_pipeline)
    {
        nvrhi::GraphicsPipelineDesc psoDesc;
        psoDesc.VS = m_vertexShader;
        psoDesc.PS = m_pixelShader;
        psoDesc.inputLayout = m_inputLayout;
        psoDesc.bindingLayouts = { m_bindingLayout, m_skyboxRenderer->GetIBLBindingLayout() };
        if (m_materialBindingLayout)
            psoDesc.bindingLayouts.push_back(m_materialBindingLayout);
        if (m_unifiedMLP.GetInferLayout())
            psoDesc.bindingLayouts.push_back(m_unifiedMLP.GetInferLayout());
        if (m_iblMLP.GetInferLayout())
            psoDesc.bindingLayouts.push_back(m_iblMLP.GetInferLayout());
        psoDesc.primType = nvrhi::PrimitiveType::TriangleList;
        psoDesc.renderState.depthStencilState.depthTestEnable = true;
        psoDesc.renderState.depthStencilState.depthWriteEnable = true;

        m_pipeline = GetDevice()->createGraphicsPipeline(psoDesc, framebuffer->getFramebufferInfo());
    }

    m_commandList->open();

    auto fbDesc = framebuffer->getDesc();
    if (fbDesc.depthAttachment.valid())
    {
        nvrhi::utils::ClearDepthStencilAttachment(m_commandList, framebuffer, 1.0f, 0);
    }

    float3 camPos = m_camera.GetPosition();
    float3 camDir = m_camera.GetDir();
    float3 camUp = m_camera.GetUp();

    NeuralConstants modelConstant{};
    memset(&modelConstant, 0, sizeof(modelConstant));
    modelConstant.cameraPos = { camPos.x, camPos.y, camPos.z, 0 };
    modelConstant.lightDir = float4(normalize(m_userInterfaceParameters->lightDir), 1.f);
    modelConstant.lightIntensity = float4(m_userInterfaceParameters->lightIntensity);
    modelConstant.baseColor = m_userInterfaceParameters->baseColor;
    modelConstant.specular = m_userInterfaceParameters->specular;
    modelConstant.roughness = m_userInterfaceParameters->roughness;
    modelConstant.metallic = m_userInterfaceParameters->metallic;
    modelConstant.enableNeuralShading = m_userInterfaceParameters->enableNeuralShading ? 1u : 0u;
    modelConstant.weightOffsets = m_weightOffsets;
    modelConstant.biasOffsets = m_biasOffsets;
    modelConstant.usePerVertexMaterial = m_hasPerVertexMaterials ? 1u : 0u;
    modelConstant.materialCount = m_materialCount;
    modelConstant.textureCount = m_textureCount;
    modelConstant.enableNeuralSSS = (m_userInterfaceParameters->enableNeuralSSS && m_unifiedMLP.IsReady()) ? 1u : 0u;
    m_unifiedMLP.FillInferOffsets(modelConstant.uniWeightOffsets, modelConstant.uniBiasOffsets);

    // IBL Sampler MLP constants
    modelConstant.enableNeuralIBL = (m_userInterfaceParameters->enableNeuralIBL && m_iblMLP.IsReady()) ? 1u : 0u;
    m_iblMLP.FillInferOffsets(modelConstant.iblWeightOffsets, modelConstant.iblBiasOffsets);
    modelConstant.view = affineToHomogeneous(translation(-camPos) * lookatZ(-camDir, camUp));
    modelConstant.viewProject = modelConstant.view * perspProjD3DStyle(radians(67.4f), float(width) / float(height), 0.1f, 10.f);
    modelConstant.inverseViewProject = inverse(modelConstant.viewProject);

    // Rotation-only inverse VP for skybox (follows Donut SkyPass pattern):
    // Decompose into invProj * invViewRot (with translation zeroed).
    float4x4 proj = perspProjD3DStyle(radians(67.4f), float(width) / float(height), 0.1f, 10.f);
    auto viewAffine = lookatZ(-camDir, camUp);  // rotation-only affine (no translation)
    viewAffine.m_translation = float3(0.f);     // ensure zero translation
    modelConstant.inverseViewProjectNoTranslation = inverse(proj) * affineToHomogeneous(inverse(viewAffine));

    m_commandList->writeBuffer(m_constantBuffer, &modelConstant, sizeof(modelConstant));

    if (m_skyboxRenderer)
    {
        m_skyboxRenderer->Render(m_commandList, framebuffer, m_constantBuffer);
    }

    nvrhi::GraphicsState state;
    nvrhi::BindingSetHandle iblSet = m_skyboxRenderer ? m_skyboxRenderer->GetIBLBindingSet() : nullptr;
    if (iblSet)
        state.bindings = { m_bindingSet, iblSet };
    else
        state.bindings = { m_bindingSet };
    
    if (m_materialBindingSet)
        state.bindings.push_back(m_materialBindingSet);

    if (m_unifiedMLP.GetInferSet())
        state.bindings.push_back(m_unifiedMLP.GetInferSet());

    if (m_iblMLP.GetInferSet())
        state.bindings.push_back(m_iblMLP.GetInferSet());

    state.indexBuffer = { m_indexBuffer, nvrhi::Format::R32_UINT, 0 };

    state.vertexBuffers = {
        { m_vertexBuffer, 0, 0 },
    };
    state.pipeline = m_pipeline;
    state.framebuffer = framebuffer;

    const nvrhi::Viewport viewport = nvrhi::Viewport(left, left + width, top, top + height, 0.f, 1.f);
    state.viewport.addViewportAndScissorRect(viewport);

    if (updateStat)
    {
        GetDevice()->resetTimerQuery(m_neuralTimer);
        m_commandList->beginTimerQuery(m_neuralTimer);
    }

    m_commandList->setGraphicsState(state);

    nvrhi::DrawArguments args;
    args.vertexCount = m_indicesNum;
    m_commandList->drawIndexed(args);

    if (updateStat)
    {
        m_commandList->endTimerQuery(m_neuralTimer);
    }

    m_commandList->close();
    GetDevice()->executeCommandList(m_commandList);

    // Run unified SSS training if active
    if (m_userInterfaceParameters->trainUnified && m_skyboxRenderer && m_skyboxRenderer->GetIBLBindingSet())
    {
        if (!m_unifiedMLP.IsTrainingActive())
        {
            m_unifiedMLP.CreateTrainingResources(
                { m_skyboxRenderer->GetIBLBindingLayout() },
                { m_skyboxRenderer->GetIBLBindingSet() });
            m_unifiedMLP.SetTrainingActive(true);
        }

        auto trainCmd = GetDevice()->createCommandList();
        trainCmd->open();
        m_unifiedMLP.TrainStep(trainCmd, { m_skyboxRenderer->GetIBLBindingSet() });
        trainCmd->close();
        GetDevice()->executeCommandList(trainCmd);

        m_unifiedMLP.SetEpoch(m_unifiedMLP.GetEpoch() + 1);

        if (m_unifiedMLP.GetEpoch() % 10 == 0)
        {
            m_unifiedMLP.ConvertToInferencing();
            m_unifiedMLP.SetReady(true);
        }
    }
    else if (m_unifiedMLP.IsTrainingActive() && !m_userInterfaceParameters->trainUnified)
    {
        m_unifiedMLP.ConvertToInferencing();
        m_unifiedMLP.SetReady(true);
        m_unifiedMLP.SetTrainingActive(false);
    }

    // Run IBL Sampler training if active
    if (m_userInterfaceParameters->trainIBL)
    {
        if (!m_iblMLP.IsTrainingActive())
        {
            m_iblMLP.CreateTrainingResources(); // No extra bindings (pure math)
            m_iblMLP.SetTrainingActive(true);
        }

        auto iblCmd = GetDevice()->createCommandList();
        iblCmd->open();
        m_iblMLP.TrainStep(iblCmd); // No extra sets
        iblCmd->close();
        GetDevice()->executeCommandList(iblCmd);

        m_iblMLP.SetEpoch(m_iblMLP.GetEpoch() + 1);

        if (m_iblMLP.GetEpoch() % 10 == 0)
        {
            m_iblMLP.ConvertToInferencing();
            m_iblMLP.SetReady(true);
        }
    }
    else if (m_iblMLP.IsTrainingActive() && !m_userInterfaceParameters->trainIBL)
    {
        m_iblMLP.ConvertToInferencing();
        m_iblMLP.SetReady(true);
        m_iblMLP.SetTrainingActive(false);
    }
}

// =============================================================================
// Save / Load unified weights — thin wrappers around NeuralMLP
// =============================================================================
void SimpleInferencing::SaveUnifiedWeights(const std::string& path)
{
    m_unifiedMLP.SaveWeights(path, m_commandList);
}

void SimpleInferencing::LoadUnifiedWeights(const std::string& path)
{
    if (m_unifiedMLP.LoadWeights(path))
        m_pipeline = nullptr; // force rebuild in case buffer changed
}

