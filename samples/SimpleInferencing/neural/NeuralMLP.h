/*
 * NeuralMLP.h
 *
 * Reusable GPU MLP manager for neural rendering effects.
 * Handles: network initialization, GPU buffer management, training dispatch,
 * optimizer dispatch, and training→inference weight layout conversion.
 *
 * Used by both the Unified SSS MLP and the IBL Sampler MLP.
 */

#pragma once

#include <random>
#include <string>
#include <memory>
#include <vector>

#include <donut/core/math/math.h>
#include <donut/engine/ShaderFactory.h>
#include <nvrhi/nvrhi.h>

#include "NeuralNetwork.h"
#include "NetworkConfig.h"

using namespace donut::math;

// Configuration for creating a NeuralMLP instance
struct NeuralMLPConfig
{
    // Architecture
    int inputNeurons = 0;
    int hiddenNeurons = 0;
    int outputNeurons = 0;
    int numHiddenLayers = 0;
    int numTransitions = 0;
    int numTransitionsAlign4 = 0;

    // Training hyperparameters
    int batchSize = 0;
    int batchCount = 0;
    int threadsPerGroup = 64;
    int threadsPerGroupOpt = 32;
    float learningRate = 0.001f;
    float lossScale = 128.0f;

    // Shader names (e.g. "app/UnifiedTraining", "app/IBLTraining")
    std::string trainingShaderName;
    std::string optimizerShaderName;

    // Binding register space for inference (3 for Unified, 4 for IBL)
    int inferBindingSet = 0;

    // Debug name prefix (e.g. "Unified", "IBL")
    std::string debugPrefix = "MLP";

    // Size of the training constant buffer struct
    size_t trainingCBSize = 0;
};

// GPU MLP lifecycle manager — one instance per neural effect
class NeuralMLP
{
public:
    NeuralMLP() = default;

    // Initialize: create host network, inference buffer, binding layout/set, compile shaders
    bool Init(nvrhi::IDevice* device,
              std::shared_ptr<rtxns::NetworkUtilities> networkUtils,
              std::shared_ptr<donut::engine::ShaderFactory> shaderFactory,
              const NeuralMLPConfig& config);

    // Initialize from pre-loaded weights (e.g. from JSON)
    bool InitFromNetwork(nvrhi::IDevice* device,
                         std::shared_ptr<rtxns::NetworkUtilities> networkUtils,
                         std::shared_ptr<donut::engine::ShaderFactory> shaderFactory,
                         const NeuralMLPConfig& config,
                         std::unique_ptr<rtxns::HostNetwork> preloadedNetwork);

    // Create/reset all GPU training resources (buffers, pipelines)
    // extraTrainingLayouts/extraTrainingSets: additional binding layouts/sets for training pipeline
    void CreateTrainingResources(
        const std::vector<nvrhi::BindingLayoutHandle>& extraTrainingLayouts = {},
        const std::vector<nvrhi::BindingSetHandle>& extraTrainingSets = {});

    // Run one epoch (batchCount batches) of training
    void TrainStep(nvrhi::ICommandList* cmdList,
                   const std::vector<nvrhi::BindingSetHandle>& extraTrainingSets = {});

    // Convert training-optimal weights to inferencing-optimal layout
    void ConvertToInferencing();

    // Load weights from file (JSON or binary)
    bool LoadWeights(const std::string& path);

    // Save current training weights to file
    void SaveWeights(const std::string& path, nvrhi::CommandListHandle cmdList);

    // Upload host network weights to inference buffer (for initial load or external update)
    void UploadToInference();

    // ---- Accessors ----
    nvrhi::BindingLayoutHandle GetInferLayout() const { return m_inferLayout; }
    nvrhi::BindingSetHandle GetInferSet() const { return m_inferSet; }
    bool IsReady() const { return m_ready; }
    void SetReady(bool r) { m_ready = r; }
    bool IsTrainingActive() const { return m_trainingActive; }
    void SetTrainingActive(bool a) { m_trainingActive = a; }
    uint32_t GetEpoch() const { return m_epoch; }
    void SetEpoch(uint32_t e) { m_epoch = e; }
    uint32_t GetTrainingStep() const { return m_trainingStep; }

    // Fill weight/bias offset arrays for the constant buffer (inferencing layout)
    void FillInferOffsets(uint4* weightOffsets, uint4* biasOffsets) const;

    // Get config
    const NeuralMLPConfig& GetConfig() const { return m_config; }

    // Access to host network (for Save/Load)
    rtxns::HostNetwork* GetHostNetwork() { return m_network.get(); }

    // Pipeline invalidation callback (set by owner to trigger pipeline rebuild)
    std::function<void()> onPipelineInvalidated;

private:
    void CreateInferBuffer(const rtxns::NetworkLayout& inferLayout);
    void UpdateInferBindingSet();
    void FillTrainOffsets(uint4* weightOffsets, uint4* biasOffsets) const;

    NeuralMLPConfig m_config;
    nvrhi::IDevice* m_device = nullptr;
    std::shared_ptr<rtxns::NetworkUtilities> m_networkUtils;
    std::shared_ptr<donut::engine::ShaderFactory> m_shaderFactory;

    // Host network
    std::unique_ptr<rtxns::HostNetwork> m_network;
    rtxns::NetworkLayout m_deviceLayout; // training-optimal layout

    // Inference
    nvrhi::BufferHandle m_inferBuffer;
    nvrhi::BindingLayoutHandle m_inferLayout;
    nvrhi::BindingSetHandle m_inferSet;
    uint4 m_inferWeightOffsets[2] = {}; // max align4 = 2 for up to 8 transitions
    uint4 m_inferBiasOffsets[2] = {};

    // Training GPU buffers
    nvrhi::BufferHandle m_hostBuffer;
    nvrhi::BufferHandle m_deviceBuffer;
    nvrhi::BufferHandle m_fp32Buffer;
    nvrhi::BufferHandle m_gradientsBuffer;
    nvrhi::BufferHandle m_moments1;
    nvrhi::BufferHandle m_moments2;
    nvrhi::BufferHandle m_lossBuffer;
    nvrhi::BufferHandle m_trainingCB;

    // Training shaders & pipelines
    nvrhi::ShaderHandle m_trainingCS;
    nvrhi::ShaderHandle m_optimizerCS;
    nvrhi::ComputePipelineHandle m_trainingPipeline;
    nvrhi::ComputePipelineHandle m_optimizerPipeline;
    nvrhi::BindingLayoutHandle m_trainingLayout;
    nvrhi::BindingSetHandle m_trainingSet;
    nvrhi::BindingLayoutHandle m_optimizerLayout;
    nvrhi::BindingSetHandle m_optimizerSet;

    // Extra training binding sets stored from CreateTrainingResources
    std::vector<nvrhi::BindingLayoutHandle> m_extraTrainingLayouts;

    // State
    uint32_t m_totalParams = 0;
    uint32_t m_trainingStep = 0;
    uint32_t m_epoch = 0;
    bool m_ready = false;
    bool m_trainingActive = false;
    std::random_device m_rd;
};
