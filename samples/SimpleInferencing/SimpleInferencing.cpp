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

    LoadSkyboxTexture(GetLocalPath("assets/skybox/kloofendal_43d_clear_1k.exr").string());

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

    if (!nvrhi::utils::CreateBindingSetAndLayout(GetDevice(), nvrhi::ShaderType::All, 0, bindingSetDesc, m_bindingLayout, m_bindingSet))
    {
        log::error("Couldn't create the binding set or layout");
        return false;
    }

    m_neuralTimer = GetDevice()->createTimerQuery();

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

    bool loaded = false;
    if (path.length() >= 4 && path.substr(path.length() - 4) == ".obj")
    {
        loaded = LoadOBJ(path, vertices, indices, mats, matIndices);
    }
    else
    {
        loaded = LoadGLTF(path, vertices, indices, mats, matIndices);
    }

    if (!loaded)
    {
        log::error("Failed to load model: %s", path.c_str());
        return false;
    }

    m_modelPath = path;
    UpdateGeometryBuffers(vertices, indices);
    return true;
}

void SimpleInferencing::LoadSkyboxTexture(const std::string& path)
{
    if (m_skyboxRenderer)
    {
        m_skyboxRenderer->LoadSkyboxTexture(path);
    }
}

void SimpleInferencing::UpdateGeometryBuffers(const std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices)
{
    m_indicesNum = (int)indices.size();

    if (!m_inputLayout) {
        nvrhi::VertexAttributeDesc attributes[] = {
            nvrhi::VertexAttributeDesc().setName("POSITION").setFormat(nvrhi::Format::RGB32_FLOAT).setOffset(0).setBufferIndex(0).setElementStride(sizeof(Vertex)),
            nvrhi::VertexAttributeDesc().setName("NORMAL").setFormat(nvrhi::Format::RGB32_FLOAT).setOffset(0).setBufferIndex(1).setElementStride(sizeof(Vertex)),
        };
        m_inputLayout = GetDevice()->createInputLayout(attributes, uint32_t(std::size(attributes)), m_vertexShader);
    }

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

    NeuralConstants modelConstant{ {},
                                   {},
                                   {},
                                   { camPos.x, camPos.y, camPos.z, 0 },
                                   float4(normalize(m_userInterfaceParameters->lightDir), 1.f),
                                   float4(m_userInterfaceParameters->lightIntensity),
                                   float4(.82f, .67f, .16f, 1.f),
                                   m_userInterfaceParameters->specular,
                                   m_userInterfaceParameters->roughness,
                                   m_userInterfaceParameters->metallic,
                                   m_userInterfaceParameters->enableNeuralShading ? 1u : 0u,
                                   m_weightOffsets,
                                   m_biasOffsets };
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
    state.indexBuffer = { m_indexBuffer, nvrhi::Format::R32_UINT, 0 };

    state.vertexBuffers = {
        { m_vertexBuffer, 0, offsetof(Vertex, position) },
        { m_vertexBuffer, 1, offsetof(Vertex, normal) },
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
}
