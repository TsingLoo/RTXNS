#include "SkyboxRenderer.h"
#include <donut/core/log.h>

using namespace donut;

SkyboxRenderer::SkyboxRenderer(
    nvrhi::DeviceHandle device,
    std::shared_ptr<donut::engine::ShaderFactory> shaderFactory,
    std::shared_ptr<donut::engine::CommonRenderPasses> commonPasses,
    std::shared_ptr<donut::render::LightProbeProcessingPass> lightProbePass,
    std::shared_ptr<donut::engine::TextureCache> textureCache)
    : m_device(device)
    , m_shaderFactory(shaderFactory)
    , m_commonPasses(commonPasses)
    , m_lightProbePass(lightProbePass)
    , m_textureCache(textureCache)
{
}

bool SkyboxRenderer::Init()
{
    m_skyboxVertexShader = m_shaderFactory->CreateShader("app/Skybox", "main_vs", nullptr, nvrhi::ShaderType::Vertex);
    m_skyboxPixelShader = m_shaderFactory->CreateShader("app/Skybox", "main_ps", nullptr, nvrhi::ShaderType::Pixel);
    m_equirectToCubeCS = m_shaderFactory->CreateShader("app/EquirectToCube", "main_cs", nullptr, nvrhi::ShaderType::Compute);

    if (!m_skyboxVertexShader || !m_skyboxPixelShader || !m_equirectToCubeCS)
    {
        return false;
    }

    nvrhi::BindingLayoutDesc blDescCS;
    blDescCS.visibility = nvrhi::ShaderType::Compute;
    blDescCS.bindings = {
        nvrhi::BindingLayoutItem::Texture_SRV(0),
        nvrhi::BindingLayoutItem::Sampler(0),
        nvrhi::BindingLayoutItem::Texture_UAV(0)
    };
    m_equirectToCubeBindingLayout = m_device->createBindingLayout(blDescCS);

    nvrhi::ComputePipelineDesc cpDesc;
    cpDesc.CS = m_equirectToCubeCS;
    cpDesc.bindingLayouts = { m_equirectToCubeBindingLayout };
    m_equirectToCubePipeline = m_device->createComputePipeline(cpDesc);
    
    nvrhi::BindingLayoutDesc iblDesc;
    iblDesc.visibility = nvrhi::ShaderType::Pixel;
    iblDesc.bindings = {
        nvrhi::BindingLayoutItem::Texture_SRV(1),  // t_Skybox (Specular)
        nvrhi::BindingLayoutItem::Texture_SRV(2),  // t_Irradiance (Diffuse)
        nvrhi::BindingLayoutItem::Sampler(0)       // shared sampler
    };
    m_iblBindingLayout = m_device->createBindingLayout(iblDesc);

    return true;
}

void SkyboxRenderer::LoadSkyboxTexture(const std::string& path)
{
    m_skyboxPath = path;
    
    auto commandList = m_device->createCommandList();
    commandList->open();
    
    m_skyboxTexture = m_textureCache->LoadTextureFromFile(path, false, m_commonPasses.get(), commandList);

    if (!m_skyboxTexture || !m_skyboxTexture->texture) {
        log::warning("Failed to load skybox texture: %s", path.c_str());
        commandList->close();
        m_device->executeCommandList(commandList);
        return;
    }

    if (!m_skyboxSampler) {
        nvrhi::SamplerDesc samplerDesc;
        samplerDesc.setAllAddressModes(nvrhi::SamplerAddressMode::Repeat);
        samplerDesc.setAllFilters(true);
        m_skyboxSampler = m_device->createSampler(samplerDesc);
    }

    uint32_t cubeSize = 1024;
    nvrhi::TextureDesc cubeDesc;
    cubeDesc.width = cubeSize;
    cubeDesc.height = cubeSize;
    cubeDesc.arraySize = 6;
    cubeDesc.isRenderTarget = true;
    cubeDesc.isUAV = true;
    uint32_t mips = 0;
    for (uint32_t s = cubeSize; s > 0; s >>= 1) mips++;
    cubeDesc.mipLevels = mips;
    cubeDesc.dimension = nvrhi::TextureDimension::TextureCube;
    cubeDesc.format = m_skyboxTexture->texture->getDesc().format;
    cubeDesc.debugName = "SkyboxCubemap";
    cubeDesc.initialState = nvrhi::ResourceStates::UnorderedAccess;
    cubeDesc.keepInitialState = true;

    m_skyboxCubemap = m_device->createTexture(cubeDesc);

    nvrhi::BindingSetDesc bsDesc;
    bsDesc.bindings = {
        nvrhi::BindingSetItem::Texture_SRV(0, m_skyboxTexture->texture),
        nvrhi::BindingSetItem::Sampler(0, m_skyboxSampler),
        nvrhi::BindingSetItem::Texture_UAV(0, m_skyboxCubemap, nvrhi::Format::UNKNOWN, nvrhi::TextureSubresourceSet(0, 1, 0, 6))
    };
    nvrhi::BindingSetHandle bindingSet = m_device->createBindingSet(bsDesc, m_equirectToCubeBindingLayout);

    nvrhi::ComputeState state;
    state.pipeline = m_equirectToCubePipeline;
    state.bindings = { bindingSet };

    commandList->setComputeState(state);
    commandList->dispatch((cubeSize + 7) / 8, (cubeSize + 7) / 8, 6);

    commandList->setTextureState(m_skyboxCubemap, nvrhi::TextureSubresourceSet(0, 1, 0, 6), nvrhi::ResourceStates::CopySource);
    commandList->commitBarriers();

    m_lightProbePass->GenerateCubemapMips(commandList, m_skyboxCubemap, 0, 0, cubeDesc.mipLevels - 1);

    commandList->setTextureState(m_skyboxCubemap, nvrhi::TextureSubresourceSet(0, cubeDesc.mipLevels, 0, 6), nvrhi::ResourceStates::ShaderResource);

    nvrhi::TextureDesc irrDesc;
    irrDesc.width = 32;
    irrDesc.height = 32;
    irrDesc.arraySize = 6;
    irrDesc.dimension = nvrhi::TextureDimension::TextureCube;
    irrDesc.format = nvrhi::Format::RGBA16_FLOAT;
    irrDesc.isRenderTarget = true;
    irrDesc.mipLevels = 1;
    irrDesc.debugName = "IrradianceCubemap";
    irrDesc.initialState = nvrhi::ResourceStates::RenderTarget;
    irrDesc.keepInitialState = true;
    m_irradianceCubemap = m_device->createTexture(irrDesc);

    m_lightProbePass->RenderDiffuseMap(commandList, m_skyboxCubemap, nvrhi::TextureSubresourceSet(0, 1, 0, 6), m_irradianceCubemap, 0, 0);

    commandList->setTextureState(m_irradianceCubemap, nvrhi::TextureSubresourceSet(0, 1, 0, 6), nvrhi::ResourceStates::ShaderResource);

    commandList->close();
    m_device->executeCommandList(commandList);

    m_skyboxBindingSet = nullptr; // Reset binding set to be rebuilt during Render if needed
    m_iblBindingSet = nullptr;
}

void SkyboxRenderer::BackBufferResizing()
{
    m_skyboxPipeline = nullptr;
}

void SkyboxRenderer::Render(nvrhi::ICommandList* commandList, nvrhi::IFramebuffer* framebuffer, nvrhi::BufferHandle constantBuffer)
{
    if (!m_skyboxPipeline)
    {
        nvrhi::GraphicsPipelineDesc psoDesc;
        psoDesc.VS = m_skyboxVertexShader;
        psoDesc.PS = m_skyboxPixelShader;
        nvrhi::BindingLayoutDesc blDesc;
        blDesc.visibility = nvrhi::ShaderType::All;
        blDesc.bindings = {
            nvrhi::BindingLayoutItem::Texture_SRV(0),
            nvrhi::BindingLayoutItem::Sampler(0),
            nvrhi::BindingLayoutItem::ConstantBuffer(0)
        };
        m_skyboxBindingLayout = m_device->createBindingLayout(blDesc);
        psoDesc.bindingLayouts = { m_skyboxBindingLayout };
        psoDesc.primType = nvrhi::PrimitiveType::TriangleList;
        
        // Draw behind everything, depth test enabled but write disabled. 
        psoDesc.renderState.depthStencilState.depthTestEnable = false;
        psoDesc.renderState.depthStencilState.depthWriteEnable = false;
        psoDesc.renderState.rasterState.cullMode = nvrhi::RasterCullMode::None;

        m_skyboxPipeline = m_device->createGraphicsPipeline(psoDesc, framebuffer->getFramebufferInfo());
    }

    if (!m_skyboxBindingSet && m_skyboxCubemap)
    {
        nvrhi::BindingSetDesc setDesc;
        setDesc.bindings = {
            nvrhi::BindingSetItem::Texture_SRV(0, m_skyboxCubemap),
            nvrhi::BindingSetItem::Sampler(0, m_skyboxSampler),
            nvrhi::BindingSetItem::ConstantBuffer(0, constantBuffer)
        };
        m_skyboxBindingSet = m_device->createBindingSet(setDesc, m_skyboxBindingLayout);

        nvrhi::BindingSetDesc iblSetDesc;
        iblSetDesc.bindings = {
            nvrhi::BindingSetItem::Texture_SRV(1, m_skyboxCubemap),
            nvrhi::BindingSetItem::Texture_SRV(2, m_irradianceCubemap),
            nvrhi::BindingSetItem::Sampler(0, m_skyboxSampler)
        };
        m_iblBindingSet = m_device->createBindingSet(iblSetDesc, m_iblBindingLayout);
    }

    if (m_skyboxPipeline && m_skyboxBindingSet)
    {
        nvrhi::GraphicsState skyboxState;
        skyboxState.bindings = { m_skyboxBindingSet };
        skyboxState.pipeline = m_skyboxPipeline;
        skyboxState.framebuffer = framebuffer;
        const nvrhi::FramebufferInfoEx& fbinfo = framebuffer->getFramebufferInfo();
        const nvrhi::Viewport viewport = nvrhi::Viewport(0, float(fbinfo.width), 0, float(fbinfo.height), 0.f, 1.f);
        skyboxState.viewport.addViewportAndScissorRect(viewport);

        commandList->setGraphicsState(skyboxState);
        
        nvrhi::DrawArguments args;
        args.vertexCount = 3;
        commandList->draw(args);
    }
}
