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

    // ===== Initialize unified MLP (Disney + IBL baked) =====
    InitUnifiedNetwork();

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
        if (m_unifiedTrainingActive)
            m_extraStatus = std::format(" - Neural - {:3d}us - Training epoch {} loss {:.4f}", t, m_unifiedEpoch, 0.0f);
        else if (m_userInterfaceParameters->enableNeuralIBL && m_unifiedReady)
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
        if (m_unifiedInferLayout)
            psoDesc.bindingLayouts.push_back(m_unifiedInferLayout);
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
    modelConstant.baseColor = float4(.82f, .67f, .16f, 1.f);
    modelConstant.specular = m_userInterfaceParameters->specular;
    modelConstant.roughness = m_userInterfaceParameters->roughness;
    modelConstant.metallic = m_userInterfaceParameters->metallic;
    modelConstant.enableNeuralShading = m_userInterfaceParameters->enableNeuralShading ? 1u : 0u;
    modelConstant.weightOffsets = m_weightOffsets;
    modelConstant.biasOffsets = m_biasOffsets;
    modelConstant.usePerVertexMaterial = m_hasPerVertexMaterials ? 1u : 0u;
    modelConstant.materialCount = m_materialCount;
    modelConstant.textureCount = m_textureCount;
    modelConstant.enableNeuralIBL = (m_userInterfaceParameters->enableNeuralIBL && m_unifiedReady) ? 1u : 0u;
    for (int i = 0; i < UNIFIED_NUM_TRANSITIONS_ALIGN4; ++i)
    {
        modelConstant.uniWeightOffsets[i] = m_unifiedWeightOffsets[i];
        modelConstant.uniBiasOffsets[i] = m_unifiedBiasOffsets[i];
    }
    modelConstant.view = affineToHomogeneous(translation(-camPos) * lookatZ(-camDir, camUp));
    modelConstant.viewProject = modelConstant.view * perspProjD3DStyle(radians(67.4f), float(width) / float(height), 0.1f, 10.f);
    modelConstant.inverseViewProject = inverse(modelConstant.viewProject);

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

    if (m_unifiedInferSet)
        state.bindings.push_back(m_unifiedInferSet);

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

    // Run unified training if active
    if (m_userInterfaceParameters->trainUnified && m_skyboxRenderer && m_skyboxRenderer->GetIBLBindingSet())
    {
        if (!m_unifiedTrainingActive)
        {
            CreateUnifiedTrainingResources();
            m_unifiedTrainingActive = true;
            m_unifiedTrainingStep = 0;
            m_unifiedEpoch = 0;
        }

        auto trainCmd = GetDevice()->createCommandList();
        trainCmd->open();
        TrainUnifiedStep(trainCmd);
        trainCmd->close();
        GetDevice()->executeCommandList(trainCmd);

        ++m_unifiedEpoch;

        // Convert to inferencing layout every 10 epochs for live preview
        if (m_unifiedEpoch % 10 == 0)
        {
            ConvertUnifiedToInferencing();
            m_unifiedReady = true;
        }
    }
    else if (m_unifiedTrainingActive && !m_userInterfaceParameters->trainUnified)
    {
        // Training stopped — do final conversion
        ConvertUnifiedToInferencing();
        m_unifiedReady = true;
        m_unifiedTrainingActive = false;
    }
}

// =============================================================================
// Unified MLP: Initialize the network
// =============================================================================
void SimpleInferencing::InitUnifiedNetwork()
{
    rtxns::NetworkArchitecture arch;
    arch.numHiddenLayers = UNIFIED_NUM_HIDDEN_LAYERS;
    arch.inputNeurons = UNIFIED_INPUT_NEURONS;
    arch.hiddenNeurons = UNIFIED_HIDDEN_NEURONS;
    arch.outputNeurons = UNIFIED_OUTPUT_NEURONS;
    arch.weightPrecision = rtxns::Precision::F16;
    arch.biasPrecision = rtxns::Precision::F16;

    m_unifiedNetwork = std::make_unique<rtxns::HostNetwork>(m_networkUtils);
    if (!m_unifiedNetwork->Initialise(arch))
    {
        log::error("Failed to create unified MLP network.");
        return;
    }

    // Create inferencing-optimal layout for rendering
    auto inferLayout = m_networkUtils->GetNewMatrixLayout(
        m_unifiedNetwork->GetNetworkLayout(), rtxns::MatrixLayout::InferencingOptimal);

    // Store inferencing offsets (5 layers packed into 2 x uint4)
    memset(m_unifiedWeightOffsets, 0, sizeof(m_unifiedWeightOffsets));
    memset(m_unifiedBiasOffsets, 0, sizeof(m_unifiedBiasOffsets));
    for (int i = 0; i < UNIFIED_NUM_TRANSITIONS; ++i)
    {
        reinterpret_cast<uint32_t*>(m_unifiedWeightOffsets)[i] = inferLayout.networkLayers[i].weightOffset;
        reinterpret_cast<uint32_t*>(m_unifiedBiasOffsets)[i] = inferLayout.networkLayers[i].biasOffset;
    }

    // Create the inferencing buffer
    {
        nvrhi::BufferDesc bufDesc;
        bufDesc.byteSize = inferLayout.networkByteSize;
        bufDesc.canHaveRawViews = true;
        bufDesc.canHaveUAVs = true;
        bufDesc.debugName = "UnifiedMLPInferBuffer";
        bufDesc.initialState = nvrhi::ResourceStates::ShaderResource;
        bufDesc.keepInitialState = true;
        m_unifiedMLPInferBuffer = GetDevice()->createBuffer(bufDesc);
    }

    // Create binding layout + set for Set 3 (inferencing)
    {
        nvrhi::BindingLayoutDesc layoutDesc;
        layoutDesc.visibility = nvrhi::ShaderType::Pixel;
        layoutDesc.registerSpace = 3;
        layoutDesc.registerSpaceIsDescriptorSet = true;
        layoutDesc.bindings = {
            nvrhi::BindingLayoutItem::RawBuffer_SRV(0)
        };
        m_unifiedInferLayout = GetDevice()->createBindingLayout(layoutDesc);

        nvrhi::BindingSetDesc setDesc;
        setDesc.bindings = {
            nvrhi::BindingSetItem::RawBuffer_SRV(0, m_unifiedMLPInferBuffer)
        };
        m_unifiedInferSet = GetDevice()->createBindingSet(setDesc, m_unifiedInferLayout);
    }

    // Force pipeline rebuild to include the new binding layout
    m_pipeline = nullptr;

    // Compile training shaders
    m_unifiedTrainingCS = m_shaderFactory->CreateShader("app/UnifiedTraining", "main_cs", nullptr, nvrhi::ShaderType::Compute);
    m_unifiedOptimizerCS = m_shaderFactory->CreateShader("app/UnifiedOptimizer", "adam_cs", nullptr, nvrhi::ShaderType::Compute);

    if (!m_unifiedTrainingCS || !m_unifiedOptimizerCS)
    {
        log::warning("Failed to compile unified training/optimizer shaders.");
    }

    // Create training constant buffer
    m_unifiedTrainingCB = GetDevice()->createBuffer(
        nvrhi::utils::CreateStaticConstantBufferDesc(sizeof(UnifiedTrainingConstants), "UnifiedTrainingCB")
            .setInitialState(nvrhi::ResourceStates::ConstantBuffer)
            .setKeepInitialState(true));

    log::info("Unified MLP initialized: %d input features, %d hidden neurons, %d layers, %d output neurons",
              UNIFIED_INPUT_FEATURES, UNIFIED_HIDDEN_NEURONS, UNIFIED_NUM_HIDDEN_LAYERS, UNIFIED_OUTPUT_NEURONS);
}

// =============================================================================
// Unified MLP: Create/reset GPU training resources
// =============================================================================
void SimpleInferencing::CreateUnifiedTrainingResources()
{
    // Reset network to fresh random weights
    rtxns::NetworkArchitecture arch;
    arch.numHiddenLayers = UNIFIED_NUM_HIDDEN_LAYERS;
    arch.inputNeurons = UNIFIED_INPUT_NEURONS;
    arch.hiddenNeurons = UNIFIED_HIDDEN_NEURONS;
    arch.outputNeurons = UNIFIED_OUTPUT_NEURONS;
    arch.weightPrecision = rtxns::Precision::F16;
    arch.biasPrecision = rtxns::Precision::F16;

    m_unifiedNetwork = std::make_unique<rtxns::HostNetwork>(m_networkUtils);
    if (!m_unifiedNetwork->Initialise(arch))
    {
        log::error("Failed to create unified MLP.");
        return;
    }

    // Get training-optimal layout
    m_unifiedDeviceLayout = m_networkUtils->GetNewMatrixLayout(
        m_unifiedNetwork->GetNetworkLayout(), rtxns::MatrixLayout::TrainingOptimal);

    auto hostBufferSize = m_unifiedNetwork->GetNetworkLayout().networkByteSize;
    auto deviceBufferSize = m_unifiedDeviceLayout.networkByteSize;

    assert((deviceBufferSize % sizeof(uint16_t)) == 0);
    m_unifiedTotalParams = uint32_t(deviceBufferSize / sizeof(uint16_t));

    // Create GPU buffers
    {
        nvrhi::BufferDesc desc;

        desc.debugName = "UnifiedMLPHostBuffer";
        desc.initialState = nvrhi::ResourceStates::CopyDest;
        desc.byteSize = hostBufferSize;
        desc.canHaveUAVs = true;
        desc.keepInitialState = true;
        m_unifiedMLPHostBuffer = GetDevice()->createBuffer(desc);

        desc.debugName = "UnifiedMLPDeviceBuffer";
        desc.initialState = nvrhi::ResourceStates::UnorderedAccess;
        desc.byteSize = deviceBufferSize;
        desc.canHaveRawViews = true;
        desc.canHaveTypedViews = true;
        desc.format = nvrhi::Format::R16_FLOAT;
        m_unifiedMLPDeviceBuffer = GetDevice()->createBuffer(desc);

        desc.debugName = "UnifiedMLPFP32Buffer";
        desc.canHaveRawViews = false;
        desc.byteSize = m_unifiedTotalParams * sizeof(float);
        desc.format = nvrhi::Format::R32_FLOAT;
        m_unifiedMLPFP32Buffer = GetDevice()->createBuffer(desc);

        desc.debugName = "UnifiedGradientsBuffer";
        desc.byteSize = (m_unifiedTotalParams * sizeof(uint16_t) + 3) & ~3;
        desc.canHaveRawViews = true;
        desc.structStride = sizeof(uint16_t);
        desc.format = nvrhi::Format::R16_FLOAT;
        m_unifiedGradientsBuffer = GetDevice()->createBuffer(desc);

        desc.debugName = "UnifiedMoments1";
        desc.byteSize = m_unifiedTotalParams * sizeof(float);
        desc.format = nvrhi::Format::R32_FLOAT;
        desc.canHaveRawViews = false;
        desc.structStride = 0;
        m_unifiedMoments1 = GetDevice()->createBuffer(desc);

        desc.debugName = "UnifiedMoments2";
        m_unifiedMoments2 = GetDevice()->createBuffer(desc);

        desc.debugName = "UnifiedLossBuffer";
        desc.byteSize = UNIFIED_BATCH_SIZE * sizeof(float);
        desc.format = nvrhi::Format::R32_FLOAT;
        desc.canHaveTypedViews = true;
        m_unifiedLossBuffer = GetDevice()->createBuffer(desc);
    }

    // Upload initial weights and convert layout
    {
        auto cmdList = GetDevice()->createCommandList();
        cmdList->open();

        cmdList->writeBuffer(m_unifiedMLPHostBuffer, 
            m_unifiedNetwork->GetNetworkParams().data(), 
            m_unifiedNetwork->GetNetworkParams().size());

        m_networkUtils->ConvertWeights(
            m_unifiedNetwork->GetNetworkLayout(), m_unifiedDeviceLayout,
            m_unifiedMLPHostBuffer, 0, m_unifiedMLPDeviceBuffer, 0,
            GetDevice(), cmdList);

        // Clear gradient + moment buffers
        cmdList->beginTrackingBufferState(m_unifiedGradientsBuffer, nvrhi::ResourceStates::UnorderedAccess);
        cmdList->clearBufferUInt(m_unifiedGradientsBuffer, 0);
        cmdList->beginTrackingBufferState(m_unifiedMoments1, nvrhi::ResourceStates::UnorderedAccess);
        cmdList->clearBufferUInt(m_unifiedMoments1, 0);
        cmdList->beginTrackingBufferState(m_unifiedMoments2, nvrhi::ResourceStates::UnorderedAccess);
        cmdList->clearBufferUInt(m_unifiedMoments2, 0);

        cmdList->close();
        GetDevice()->executeCommandList(cmdList);
    }

    // Build training weight/bias offset arrays (training layout)
    uint4 trainWeightOffsets[UNIFIED_NUM_TRANSITIONS_ALIGN4] = {};
    uint4 trainBiasOffsets[UNIFIED_NUM_TRANSITIONS_ALIGN4] = {};
    for (int i = 0; i < UNIFIED_NUM_TRANSITIONS; ++i)
    {
        reinterpret_cast<uint32_t*>(trainWeightOffsets)[i] = m_unifiedDeviceLayout.networkLayers[i].weightOffset;
        reinterpret_cast<uint32_t*>(trainBiasOffsets)[i] = m_unifiedDeviceLayout.networkLayers[i].biasOffset;
    }

    // Create training binding layout + set
    {
        // Training layout: CB + SRV(weights) + UAV(gradients) + UAV(loss) — Set 0
        // Also needs IBL cubemaps — Set 1
        nvrhi::BindingLayoutDesc trainLayoutDesc;
        trainLayoutDesc.visibility = nvrhi::ShaderType::Compute;
        trainLayoutDesc.registerSpaceIsDescriptorSet = true;
        trainLayoutDesc.bindings = {
            nvrhi::BindingLayoutItem::ConstantBuffer(0),
            nvrhi::BindingLayoutItem::RawBuffer_SRV(0),
            nvrhi::BindingLayoutItem::RawBuffer_UAV(0),
            nvrhi::BindingLayoutItem::TypedBuffer_UAV(1)
        };
        m_unifiedTrainingLayout = GetDevice()->createBindingLayout(trainLayoutDesc);

        nvrhi::BindingSetDesc trainSetDesc;
        trainSetDesc.bindings = {
            nvrhi::BindingSetItem::ConstantBuffer(0, m_unifiedTrainingCB),
            nvrhi::BindingSetItem::RawBuffer_SRV(0, m_unifiedMLPDeviceBuffer),
            nvrhi::BindingSetItem::RawBuffer_UAV(0, m_unifiedGradientsBuffer),
            nvrhi::BindingSetItem::TypedBuffer_UAV(1, m_unifiedLossBuffer)
        };
        m_unifiedTrainingSet = GetDevice()->createBindingSet(trainSetDesc, m_unifiedTrainingLayout);

        nvrhi::ComputePipelineDesc cpDesc;
        cpDesc.CS = m_unifiedTrainingCS;
        cpDesc.bindingLayouts = { m_unifiedTrainingLayout, m_skyboxRenderer->GetIBLBindingLayout() };
        m_unifiedTrainingPipeline = GetDevice()->createComputePipeline(cpDesc);
    }

    // Create optimizer binding layout + set
    {
        nvrhi::BindingLayoutDesc optLayoutDesc;
        optLayoutDesc.visibility = nvrhi::ShaderType::Compute;
        optLayoutDesc.registerSpaceIsDescriptorSet = true;
        optLayoutDesc.bindings = {
            nvrhi::BindingLayoutItem::ConstantBuffer(0),
            nvrhi::BindingLayoutItem::TypedBuffer_UAV(0),  // fp16 weights
            nvrhi::BindingLayoutItem::TypedBuffer_UAV(1),  // fp32 weights
            nvrhi::BindingLayoutItem::TypedBuffer_UAV(2),  // gradients
            nvrhi::BindingLayoutItem::TypedBuffer_UAV(3),  // moments1
            nvrhi::BindingLayoutItem::TypedBuffer_UAV(4)   // moments2
        };
        m_unifiedOptimizerLayout = GetDevice()->createBindingLayout(optLayoutDesc);

        nvrhi::BindingSetDesc optSetDesc;
        optSetDesc.bindings = {
            nvrhi::BindingSetItem::ConstantBuffer(0, m_unifiedTrainingCB),
            nvrhi::BindingSetItem::TypedBuffer_UAV(0, m_unifiedMLPDeviceBuffer),
            nvrhi::BindingSetItem::TypedBuffer_UAV(1, m_unifiedMLPFP32Buffer),
            nvrhi::BindingSetItem::TypedBuffer_UAV(2, m_unifiedGradientsBuffer),
            nvrhi::BindingSetItem::TypedBuffer_UAV(3, m_unifiedMoments1),
            nvrhi::BindingSetItem::TypedBuffer_UAV(4, m_unifiedMoments2)
        };
        m_unifiedOptimizerSet = GetDevice()->createBindingSet(optSetDesc, m_unifiedOptimizerLayout);

        nvrhi::ComputePipelineDesc cpDesc;
        cpDesc.CS = m_unifiedOptimizerCS;
        cpDesc.bindingLayouts = { m_unifiedOptimizerLayout };
        m_unifiedOptimizerPipeline = GetDevice()->createComputePipeline(cpDesc);
    }

    m_unifiedTrainingStep = 0;
    m_unifiedEpoch = 0;
    m_unifiedReady = false;

    log::info("Unified MLP training resources created: %d total params (%d bytes fp16)",
              m_unifiedTotalParams, m_unifiedTotalParams * 2);
}

// =============================================================================
// Unified MLP: Run one epoch of training
// =============================================================================
void SimpleInferencing::TrainUnifiedStep(nvrhi::ICommandList* cmdList)
{
    std::uniform_int_distribution<uint64_t> ldist;
    uint64_t seed = ldist(m_rd);

    // Build training-layout offset arrays
    uint4 trainWeightOffsets[UNIFIED_NUM_TRANSITIONS_ALIGN4] = {};
    uint4 trainBiasOffsets[UNIFIED_NUM_TRANSITIONS_ALIGN4] = {};
    for (int i = 0; i < UNIFIED_NUM_TRANSITIONS; ++i)
    {
        reinterpret_cast<uint32_t*>(trainWeightOffsets)[i] = m_unifiedDeviceLayout.networkLayers[i].weightOffset;
        reinterpret_cast<uint32_t*>(trainBiasOffsets)[i] = m_unifiedDeviceLayout.networkLayers[i].biasOffset;
    }

    for (int batch = 0; batch < UNIFIED_BATCH_COUNT; ++batch)
    {
        ++m_unifiedTrainingStep;

        UnifiedTrainingConstants trainConst = {};
        for (int i = 0; i < UNIFIED_NUM_TRANSITIONS_ALIGN4; ++i)
        {
            trainConst.weightOffsets[i] = trainWeightOffsets[i];
            trainConst.biasOffsets[i] = trainBiasOffsets[i];
        }
        trainConst.maxParamSize = m_unifiedTotalParams;
        trainConst.learningRate = UNIFIED_LEARNING_RATE;
        trainConst.currentStep = float(m_unifiedTrainingStep);
        trainConst.batchSize = UNIFIED_BATCH_SIZE;
        trainConst.seed = seed + batch;

        cmdList->writeBuffer(m_unifiedTrainingCB, &trainConst, sizeof(trainConst));

        // Training dispatch
        nvrhi::ComputeState trainState;
        trainState.pipeline = m_unifiedTrainingPipeline;
        trainState.bindings = { m_unifiedTrainingSet, m_skyboxRenderer->GetIBLBindingSet() };
        cmdList->setComputeState(trainState);
        cmdList->dispatch(UNIFIED_BATCH_SIZE / UNIFIED_THREADS_PER_GROUP, 1, 1);

        // Optimizer dispatch
        nvrhi::ComputeState optState;
        optState.pipeline = m_unifiedOptimizerPipeline;
        optState.bindings = { m_unifiedOptimizerSet };
        cmdList->setComputeState(optState);
        cmdList->dispatch((m_unifiedTotalParams + UNIFIED_THREADS_PER_GROUP_OPT - 1) / UNIFIED_THREADS_PER_GROUP_OPT, 1, 1);
    }
}

// =============================================================================
// Unified MLP: Convert training weights to inferencing-optimal layout
// =============================================================================
void SimpleInferencing::ConvertUnifiedToInferencing()
{
    auto inferLayout = m_networkUtils->GetNewMatrixLayout(
        m_unifiedNetwork->GetNetworkLayout(), rtxns::MatrixLayout::InferencingOptimal);

    // Update inferencing offsets
    memset(m_unifiedWeightOffsets, 0, sizeof(m_unifiedWeightOffsets));
    memset(m_unifiedBiasOffsets, 0, sizeof(m_unifiedBiasOffsets));
    for (int i = 0; i < UNIFIED_NUM_TRANSITIONS; ++i)
    {
        reinterpret_cast<uint32_t*>(m_unifiedWeightOffsets)[i] = inferLayout.networkLayers[i].weightOffset;
        reinterpret_cast<uint32_t*>(m_unifiedBiasOffsets)[i] = inferLayout.networkLayers[i].biasOffset;
    }

    // Ensure inferencing buffer is large enough
    if (!m_unifiedMLPInferBuffer || m_unifiedMLPInferBuffer->getDesc().byteSize < inferLayout.networkByteSize)
    {
        nvrhi::BufferDesc bufDesc;
        bufDesc.byteSize = inferLayout.networkByteSize;
        bufDesc.canHaveRawViews = true;
        bufDesc.canHaveUAVs = true;
        bufDesc.debugName = "UnifiedMLPInferBuffer";
        bufDesc.initialState = nvrhi::ResourceStates::UnorderedAccess;
        bufDesc.keepInitialState = false;
        m_unifiedMLPInferBuffer = GetDevice()->createBuffer(bufDesc);

        // Rebuild binding set with new buffer
        nvrhi::BindingSetDesc setDesc;
        setDesc.bindings = {
            nvrhi::BindingSetItem::RawBuffer_SRV(0, m_unifiedMLPInferBuffer)
        };
        m_unifiedInferSet = GetDevice()->createBindingSet(setDesc, m_unifiedInferLayout);
        m_pipeline = nullptr; // Force pipeline rebuild
    }

    // Convert weights from training layout to inferencing layout
    auto cmdList = GetDevice()->createCommandList();
    cmdList->open();

    cmdList->setBufferState(m_unifiedMLPInferBuffer, nvrhi::ResourceStates::UnorderedAccess);
    cmdList->commitBarriers();

    m_networkUtils->ConvertWeights(
        m_unifiedDeviceLayout, inferLayout,
        m_unifiedMLPDeviceBuffer, 0, m_unifiedMLPInferBuffer, 0,
        GetDevice(), cmdList);

    cmdList->setBufferState(m_unifiedMLPInferBuffer, nvrhi::ResourceStates::ShaderResource);
    cmdList->commitBarriers();

    cmdList->close();
    GetDevice()->executeCommandList(cmdList);
}

// =============================================================================
// Unified MLP: Save and Load weights
// =============================================================================
void SimpleInferencing::SaveUnifiedWeights(const std::string& path)
{
    if (!m_unifiedNetwork || !m_unifiedMLPDeviceBuffer)
    {
        log::warning("No unified weights to save.");
        return;
    }

    m_unifiedNetwork->UpdateFromBufferToFile(
        m_unifiedMLPHostBuffer, m_unifiedMLPDeviceBuffer,
        m_unifiedNetwork->GetNetworkLayout(), m_unifiedDeviceLayout,
        path, GetDevice(), m_commandList);

    log::info("Saved unified MLP weights to: %s", path.c_str());
}

void SimpleInferencing::LoadUnifiedWeights(const std::string& path)
{
    auto loaded = std::make_unique<rtxns::HostNetwork>(m_networkUtils);
    bool success = false;

    // Detect if file is JSON or native binary
    if (path.length() >= 5 && path.substr(path.length() - 5) == ".json")
    {
        auto nativeFS = std::make_shared<donut::vfs::NativeFileSystem>();
        success = loaded->InitialiseFromJson(*nativeFS, path);
    }
    else
    {
        success = loaded->InitialiseFromFile(path);
    }

    if (!success)
    {
        log::error("Failed to load unified weights from: %s", path.c_str());
        return;
    }

    m_unifiedNetwork = std::move(loaded);

    // Convert to inferencing layout and upload
    auto inferLayout = m_networkUtils->GetNewMatrixLayout(
        m_unifiedNetwork->GetNetworkLayout(), rtxns::MatrixLayout::InferencingOptimal);

    // Update offsets
    memset(m_unifiedWeightOffsets, 0, sizeof(m_unifiedWeightOffsets));
    memset(m_unifiedBiasOffsets, 0, sizeof(m_unifiedBiasOffsets));
    for (int i = 0; i < UNIFIED_NUM_TRANSITIONS; ++i)
    {
        reinterpret_cast<uint32_t*>(m_unifiedWeightOffsets)[i] = inferLayout.networkLayers[i].weightOffset;
        reinterpret_cast<uint32_t*>(m_unifiedBiasOffsets)[i] = inferLayout.networkLayers[i].biasOffset;
    }

    // Recreate inferencing buffer if needed
    if (!m_unifiedMLPInferBuffer || m_unifiedMLPInferBuffer->getDesc().byteSize < inferLayout.networkByteSize)
    {
        nvrhi::BufferDesc bufDesc;
        bufDesc.byteSize = inferLayout.networkByteSize;
        bufDesc.canHaveRawViews = true;
        bufDesc.canHaveUAVs = true;
        bufDesc.debugName = "UnifiedMLPInferBuffer";
        bufDesc.initialState = nvrhi::ResourceStates::UnorderedAccess;
        bufDesc.keepInitialState = false;
        m_unifiedMLPInferBuffer = GetDevice()->createBuffer(bufDesc);

        nvrhi::BindingSetDesc setDesc;
        setDesc.bindings = {
            nvrhi::BindingSetItem::RawBuffer_SRV(0, m_unifiedMLPInferBuffer)
        };
        m_unifiedInferSet = GetDevice()->createBindingSet(setDesc, m_unifiedInferLayout);
        m_pipeline = nullptr;
    }

    // Upload host and convert to device
    nvrhi::BufferDesc hostDesc;
    hostDesc.byteSize = m_unifiedNetwork->GetNetworkParams().size();
    hostDesc.debugName = "UnifiedMLPLoadHost";
    hostDesc.initialState = nvrhi::ResourceStates::CopyDest;
    hostDesc.keepInitialState = true;
    auto hostBuf = GetDevice()->createBuffer(hostDesc);

    auto cmdList = GetDevice()->createCommandList();
    cmdList->open();

    cmdList->writeBuffer(hostBuf, m_unifiedNetwork->GetNetworkParams().data(),
                         m_unifiedNetwork->GetNetworkParams().size());

    cmdList->setBufferState(m_unifiedMLPInferBuffer, nvrhi::ResourceStates::UnorderedAccess);
    cmdList->commitBarriers();

    m_networkUtils->ConvertWeights(
        m_unifiedNetwork->GetNetworkLayout(), inferLayout,
        hostBuf, 0, m_unifiedMLPInferBuffer, 0,
        GetDevice(), cmdList);

    cmdList->setBufferState(m_unifiedMLPInferBuffer, nvrhi::ResourceStates::ShaderResource);
    cmdList->commitBarriers();

    cmdList->close();
    GetDevice()->executeCommandList(cmdList);

    m_unifiedReady = true;
    log::info("Loaded unified MLP weights from: %s", path.c_str());
}
