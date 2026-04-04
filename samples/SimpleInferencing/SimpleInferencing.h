#pragma once

#include <donut/app/ApplicationBase.h>
#include <donut/engine/ShaderFactory.h>
#include <donut/engine/CommonRenderPasses.h>
#include <donut/engine/TextureCache.h>
#include <donut/render/LightProbeProcessingPass.h>
#include <donut/core/vfs/VFS.h>
#include <donut/core/math/math.h>
#include <nvrhi/nvrhi.h>

using namespace donut::math;
#include "NetworkConfig.h"
#include "GraphicsResources.h"
#include "GeometryUtils.h"
#include "NeuralNetwork.h"

#include "UIData.h"
#include "SimpleCamera.h"
#include "SkyboxRenderer.h"

#include <memory>
#include <string>
#include <vector>

class SimpleInferencing : public donut::app::IRenderPass
{
public:
    SimpleInferencing(donut::app::DeviceManager* deviceManager, UIData* uiParams, const std::string& modelPath);

    bool Init();
    std::shared_ptr<donut::engine::ShaderFactory> GetShaderFactory() const;
    bool LoadModel(const std::string& path);
    void LoadSkyboxTexture(const std::string& path);
    
    // IRenderPass
    bool KeyboardUpdate(int key, int scancode, int action, int mods) override;
    bool MousePosUpdate(double xpos, double ypos) override;
    bool MouseButtonUpdate(int button, int action, int mods) override;
    bool MouseScrollUpdate(double xoffset, double yoffset) override;
    void Animate(float seconds) override;
    void BackBufferResizing() override;
    void Render(nvrhi::IFramebuffer* framebuffer) override;

private:
    void UpdateGeometryBuffers(const std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices);

    std::string m_extraStatus;
    nvrhi::TimerQueryHandle m_neuralTimer;
    nvrhi::ShaderHandle m_vertexShader;
    nvrhi::ShaderHandle m_pixelShader;
    nvrhi::BufferHandle m_constantBuffer;
    nvrhi::BufferHandle m_mlpHostBuffer;
    nvrhi::BufferHandle m_mlpDeviceBuffer;
    nvrhi::BufferHandle m_vertexBuffer;
    nvrhi::BufferHandle m_indexBuffer;

    nvrhi::InputLayoutHandle m_inputLayout;
    nvrhi::BindingLayoutHandle m_bindingLayout;
    nvrhi::BindingSetHandle m_bindingSet;
    nvrhi::GraphicsPipelineHandle m_pipeline;
    nvrhi::CommandListHandle m_commandList;

    std::shared_ptr<donut::engine::ShaderFactory> m_shaderFactory;
    std::shared_ptr<donut::engine::CommonRenderPasses> m_commonPasses;
    std::shared_ptr<donut::render::LightProbeProcessingPass> m_lightProbePass;
    std::shared_ptr<donut::vfs::RootFileSystem> m_rootFS;
    std::shared_ptr<rtxns::NetworkUtilities> m_networkUtils;
    std::shared_ptr<donut::engine::TextureCache> m_textureCache;

    int m_indicesNum = 0;
    std::string m_modelPath;

    donut::math::uint4 m_weightOffsets;
    donut::math::uint4 m_biasOffsets;

    UIData* m_userInterfaceParameters;
    
    SimpleCamera m_camera;
    std::unique_ptr<SkyboxRenderer> m_skyboxRenderer;
};
