#pragma once

#include <random>

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
    void SaveUnifiedWeights(const std::string& path);
    void LoadUnifiedWeights(const std::string& path);
    
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
    void CreateMaterialResources(const std::vector<MaterialParams>& materials, const std::vector<GltfTextureData>& textures);

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

    // Material system
    nvrhi::BufferHandle m_materialBuffer;                       // Structured buffer of MaterialParams
    std::vector<nvrhi::TextureHandle> m_materialTextures;       // Individual Texture2D handles
    nvrhi::SamplerHandle m_materialSampler;
    nvrhi::BindingLayoutHandle m_materialBindingLayout;
    nvrhi::BindingSetHandle m_materialBindingSet;
    std::vector<MaterialParams> m_materials;
    uint32_t m_materialCount = 0;
    uint32_t m_textureCount = 0;
    bool m_hasPerVertexMaterials = false;

    // ===== Unified MLP (Disney + IBL baked into one network) =====
    // Network
    std::unique_ptr<rtxns::HostNetwork> m_unifiedNetwork;
    rtxns::NetworkLayout m_unifiedDeviceLayout;
    donut::math::uint4 m_unifiedWeightOffsets[2]; // UNIFIED_NUM_TRANSITIONS_ALIGN4
    donut::math::uint4 m_unifiedBiasOffsets[2];

    // GPU buffers
    nvrhi::BufferHandle m_unifiedMLPHostBuffer;
    nvrhi::BufferHandle m_unifiedMLPDeviceBuffer;   // fp16 weights (training layout)
    nvrhi::BufferHandle m_unifiedMLPInferBuffer;     // fp16 weights (inferencing layout)
    nvrhi::BufferHandle m_unifiedMLPFP32Buffer;      // fp32 shadow copy
    nvrhi::BufferHandle m_unifiedGradientsBuffer;
    nvrhi::BufferHandle m_unifiedMoments1;
    nvrhi::BufferHandle m_unifiedMoments2;
    nvrhi::BufferHandle m_unifiedLossBuffer;
    nvrhi::BufferHandle m_unifiedTrainingCB;

    // Training shaders and pipelines
    nvrhi::ShaderHandle m_unifiedTrainingCS;
    nvrhi::ShaderHandle m_unifiedOptimizerCS;
    nvrhi::ComputePipelineHandle m_unifiedTrainingPipeline;
    nvrhi::ComputePipelineHandle m_unifiedOptimizerPipeline;
    nvrhi::BindingLayoutHandle m_unifiedTrainingLayout;
    nvrhi::BindingSetHandle m_unifiedTrainingSet;
    nvrhi::BindingLayoutHandle m_unifiedOptimizerLayout;
    nvrhi::BindingSetHandle m_unifiedOptimizerSet;

    // Inference binding for unified weights (Set 3)
    nvrhi::BindingLayoutHandle m_unifiedInferLayout;
    nvrhi::BindingSetHandle m_unifiedInferSet;

    // Training state
    bool m_unifiedTrainingActive = false;
    uint32_t m_unifiedTrainingStep = 0;
    uint32_t m_unifiedEpoch = 0;
    uint32_t m_unifiedTotalParams = 0;
    bool m_unifiedReady = false;  // weights loaded or trained
    std::random_device m_rd;

    // Methods
    void InitUnifiedNetwork();
    void CreateUnifiedTrainingResources();
    void TrainUnifiedStep(nvrhi::ICommandList* cmdList);
    void ConvertUnifiedToInferencing();

    // ===== IBL Sampler MLP (learns GGX importance sampling directions + LOD) =====
    // Network
    std::unique_ptr<rtxns::HostNetwork> m_iblNetwork;
    rtxns::NetworkLayout m_iblDeviceLayout;
    donut::math::uint4 m_iblWeightOffsets[IBL_NUM_TRANSITIONS_ALIGN4];
    donut::math::uint4 m_iblBiasOffsets[IBL_NUM_TRANSITIONS_ALIGN4];

    // GPU buffers
    nvrhi::BufferHandle m_iblMLPHostBuffer;
    nvrhi::BufferHandle m_iblMLPDeviceBuffer;
    nvrhi::BufferHandle m_iblMLPInferBuffer;
    nvrhi::BufferHandle m_iblMLPFP32Buffer;
    nvrhi::BufferHandle m_iblGradientsBuffer;
    nvrhi::BufferHandle m_iblMoments1;
    nvrhi::BufferHandle m_iblMoments2;
    nvrhi::BufferHandle m_iblLossBuffer;
    nvrhi::BufferHandle m_iblTrainingCB;

    // Training shaders and pipelines
    nvrhi::ShaderHandle m_iblTrainingCS;
    nvrhi::ShaderHandle m_iblOptimizerCS;
    nvrhi::ComputePipelineHandle m_iblTrainingPipeline;
    nvrhi::ComputePipelineHandle m_iblOptimizerPipeline;
    nvrhi::BindingLayoutHandle m_iblTrainingLayout;
    nvrhi::BindingSetHandle m_iblTrainingSet;
    nvrhi::BindingLayoutHandle m_iblOptimizerLayout;
    nvrhi::BindingSetHandle m_iblOptimizerSet;

    // Inference binding (Set 4)
    nvrhi::BindingLayoutHandle m_iblInferLayout;
    nvrhi::BindingSetHandle m_iblInferSet;

    // Training state
    bool m_iblTrainingActive = false;
    uint32_t m_iblTrainingStep = 0;
    uint32_t m_iblEpoch = 0;
    uint32_t m_iblTotalParams = 0;
    bool m_iblReady = false;

    // Methods
    void InitIBLNetwork();
    void CreateIBLTrainingResources();
    void TrainIBLStep(nvrhi::ICommandList* cmdList);
    void ConvertIBLToInferencing();
};
