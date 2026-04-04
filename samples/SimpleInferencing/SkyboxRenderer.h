#pragma once

#include <nvrhi/nvrhi.h>
#include <donut/engine/ShaderFactory.h>
#include <donut/engine/CommonRenderPasses.h>
#include <donut/engine/TextureCache.h>
#include <donut/render/LightProbeProcessingPass.h>
#include <string>
#include <memory>

struct IBLConstants
{
    float roughness;
    float cubeSize;
    float pad0;
    float pad1;
};

class SkyboxRenderer
{
public:
    SkyboxRenderer(
        nvrhi::DeviceHandle device,
        std::shared_ptr<donut::engine::ShaderFactory> shaderFactory,
        std::shared_ptr<donut::engine::CommonRenderPasses> commonPasses,
        std::shared_ptr<donut::render::LightProbeProcessingPass> lightProbePass,
        std::shared_ptr<donut::engine::TextureCache> textureCache);

    bool Init();
    void LoadSkyboxTexture(const std::string& path);
    void Render(nvrhi::ICommandList* commandList, nvrhi::IFramebuffer* framebuffer, nvrhi::BufferHandle constantBuffer);
    
    void BackBufferResizing();

    nvrhi::BindingLayoutHandle GetIBLBindingLayout() const { return m_iblBindingLayout; }
    nvrhi::BindingSetHandle GetIBLBindingSet() const { return m_iblBindingSet; }

private:
    nvrhi::DeviceHandle m_device;
    std::shared_ptr<donut::engine::ShaderFactory> m_shaderFactory;
    std::shared_ptr<donut::engine::CommonRenderPasses> m_commonPasses;
    std::shared_ptr<donut::render::LightProbeProcessingPass> m_lightProbePass;
    std::shared_ptr<donut::engine::TextureCache> m_textureCache;

    nvrhi::ShaderHandle m_skyboxVertexShader;
    nvrhi::ShaderHandle m_skyboxPixelShader;
    nvrhi::BindingLayoutHandle m_skyboxBindingLayout;
    nvrhi::BindingSetHandle m_skyboxBindingSet;
    nvrhi::GraphicsPipelineHandle m_skyboxPipeline;
    nvrhi::BindingLayoutHandle m_iblBindingLayout;
    nvrhi::BindingSetHandle m_iblBindingSet;

    std::shared_ptr<donut::engine::LoadedTexture> m_skyboxTexture;
    std::string m_skyboxPath;
    nvrhi::SamplerHandle m_skyboxSampler;
    nvrhi::TextureHandle m_skyboxCubemap;
    nvrhi::TextureHandle m_irradianceCubemap;
    nvrhi::TextureHandle m_specularCubemap;
    nvrhi::TextureHandle m_brdfLutTexture;
    nvrhi::ShaderHandle m_equirectToCubeCS;
    nvrhi::BindingLayoutHandle m_equirectToCubeBindingLayout;
    nvrhi::ComputePipelineHandle m_equirectToCubePipeline;

    nvrhi::ShaderHandle m_irradianceCS;
    nvrhi::BindingLayoutHandle m_irradianceBindingLayout;
    nvrhi::ComputePipelineHandle m_irradiancePipeline;

    nvrhi::ShaderHandle m_glossyCS;
    nvrhi::BindingLayoutHandle m_glossyBindingLayout;
    nvrhi::ComputePipelineHandle m_glossyPipeline;
    nvrhi::BufferHandle m_convolutionCB;
};
