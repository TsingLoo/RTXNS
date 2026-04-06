/*
 * NeuralMLP.cpp
 *
 * Implementation of the reusable GPU MLP manager.
 * Handles all common GPU operations: buffer creation, weight conversion,
 * training dispatch, and optimizer dispatch.
 */

#include "NeuralMLP.h"
#include <donut/core/log.h>
#include <nvrhi/utils.h>
#include <cassert>
#include <random>
#include <functional>
#include <donut/core/vfs/VFS.h>

using namespace donut;

// =============================================================================
// Init: Create host network, inference buffer, bindings, compile shaders
// =============================================================================
bool NeuralMLP::Init(nvrhi::IDevice* device,
                     std::shared_ptr<rtxns::NetworkUtilities> networkUtils,
                     std::shared_ptr<donut::engine::ShaderFactory> shaderFactory,
                     const NeuralMLPConfig& config)
{
    m_device = device;
    m_networkUtils = networkUtils;
    m_shaderFactory = shaderFactory;
    m_config = config;

    rtxns::NetworkArchitecture arch;
    arch.numHiddenLayers = config.numHiddenLayers;
    arch.inputNeurons = config.inputNeurons;
    arch.hiddenNeurons = config.hiddenNeurons;
    arch.outputNeurons = config.outputNeurons;
    arch.weightPrecision = rtxns::Precision::F16;
    arch.biasPrecision = rtxns::Precision::F16;

    m_network = std::make_unique<rtxns::HostNetwork>(m_networkUtils);
    if (!m_network->Initialise(arch))
    {
        log::error("[%s] Failed to create MLP network.", config.debugPrefix.c_str());
        return false;
    }

    bool ok = InitFromNetwork(device, networkUtils, shaderFactory, config, std::move(m_network));
    // Random init → not ready for inference (must be trained first)
    m_ready = false;
    return ok;
}

// =============================================================================
// InitFromNetwork: Use a pre-loaded HostNetwork (e.g. from JSON/binary)
// =============================================================================
bool NeuralMLP::InitFromNetwork(nvrhi::IDevice* device,
                                std::shared_ptr<rtxns::NetworkUtilities> networkUtils,
                                std::shared_ptr<donut::engine::ShaderFactory> shaderFactory,
                                const NeuralMLPConfig& config,
                                std::unique_ptr<rtxns::HostNetwork> preloadedNetwork)
{
    m_device = device;
    m_networkUtils = networkUtils;
    m_shaderFactory = shaderFactory;
    m_config = config;
    m_network = std::move(preloadedNetwork);

    // Create inferencing-optimal layout
    auto inferLayout = m_networkUtils->GetNewMatrixLayout(
        m_network->GetNetworkLayout(), rtxns::MatrixLayout::InferencingOptimal);

    // Fill inference offsets
    memset(m_inferWeightOffsets, 0, sizeof(m_inferWeightOffsets));
    memset(m_inferBiasOffsets, 0, sizeof(m_inferBiasOffsets));
    for (int i = 0; i < config.numTransitions; ++i)
    {
        reinterpret_cast<uint32_t*>(m_inferWeightOffsets)[i] = inferLayout.networkLayers[i].weightOffset;
        reinterpret_cast<uint32_t*>(m_inferBiasOffsets)[i] = inferLayout.networkLayers[i].biasOffset;
    }

    // Create inference buffer + upload weights
    CreateInferBuffer(inferLayout);

    {
        auto params = m_network->GetNetworkParams();
        nvrhi::BufferDesc hostBufDesc;
        hostBufDesc.byteSize = params.size();
        hostBufDesc.debugName = (config.debugPrefix + "UploadBuffer").c_str();
        hostBufDesc.initialState = nvrhi::ResourceStates::CopyDest;
        hostBufDesc.keepInitialState = true;
        auto hostBuffer = m_device->createBuffer(hostBufDesc);

        auto cmdList = m_device->createCommandList();
        cmdList->open();
        cmdList->writeBuffer(hostBuffer, params.data(), params.size());

        m_networkUtils->ConvertWeights(m_network->GetNetworkLayout(), inferLayout,
                                       hostBuffer, 0, m_inferBuffer, 0,
                                       m_device, cmdList);

        cmdList->setBufferState(m_inferBuffer, nvrhi::ResourceStates::ShaderResource);
        cmdList->commitBarriers();
        cmdList->close();
        m_device->executeCommandList(cmdList);
    }

    m_ready = true;

    // Create binding layout + set for inference
    {
        nvrhi::BindingLayoutDesc layoutDesc;
        layoutDesc.visibility = nvrhi::ShaderType::Pixel;
        layoutDesc.registerSpace = config.inferBindingSet;
        layoutDesc.registerSpaceIsDescriptorSet = true;
        layoutDesc.bindings = {
            nvrhi::BindingLayoutItem::RawBuffer_SRV(0)
        };
        m_inferLayout = m_device->createBindingLayout(layoutDesc);
        UpdateInferBindingSet();
    }

    // Compile training shaders
    if (!config.trainingShaderName.empty())
    {
        m_trainingCS = m_shaderFactory->CreateShader(
            config.trainingShaderName.c_str(), "main_cs", nullptr, nvrhi::ShaderType::Compute);
    }
    if (!config.optimizerShaderName.empty())
    {
        m_optimizerCS = m_shaderFactory->CreateShader(
            config.optimizerShaderName.c_str(), "adam_cs", nullptr, nvrhi::ShaderType::Compute);
    }

    if ((!config.trainingShaderName.empty() && !m_trainingCS) ||
        (!config.optimizerShaderName.empty() && !m_optimizerCS))
    {
        log::warning("[%s] Failed to compile training/optimizer shaders.", config.debugPrefix.c_str());
    }

    // Create training constant buffer
    if (config.trainingCBSize > 0)
    {
        // Always allocate CB large enough for MAX_TRANSITIONS_ALIGN4 arrays
        size_t cbSize = std::max(config.trainingCBSize, sizeof(uint4) * MAX_TRANSITIONS_ALIGN4 * 2 + 32);
        m_trainingCB = m_device->createBuffer(
            nvrhi::utils::CreateStaticConstantBufferDesc(
                cbSize, (config.debugPrefix + "TrainingCB").c_str())
                .setInitialState(nvrhi::ResourceStates::ConstantBuffer)
                .setKeepInitialState(true));
    }

    log::info("[%s] MLP initialized: %d→%dx%d→%d (set %d)",
              config.debugPrefix.c_str(),
              config.inputNeurons, config.hiddenNeurons, config.numHiddenLayers,
              config.outputNeurons, config.inferBindingSet);

    return true;
}

// =============================================================================
// CreateTrainingResources: Create/reset all GPU training buffers + pipelines
// =============================================================================
void NeuralMLP::CreateTrainingResources(
    const std::vector<nvrhi::BindingLayoutHandle>& extraTrainingLayouts,
    const std::vector<nvrhi::BindingSetHandle>& extraTrainingSets)
{
    m_extraTrainingLayouts = extraTrainingLayouts;

    // Re-create host network with random weights
    rtxns::NetworkArchitecture arch;
    arch.numHiddenLayers = m_config.numHiddenLayers;
    arch.inputNeurons = m_config.inputNeurons;
    arch.hiddenNeurons = m_config.hiddenNeurons;
    arch.outputNeurons = m_config.outputNeurons;
    arch.weightPrecision = rtxns::Precision::F16;
    arch.biasPrecision = rtxns::Precision::F16;

    m_network = std::make_unique<rtxns::HostNetwork>(m_networkUtils);
    if (!m_network->Initialise(arch))
    {
        log::error("[%s] Failed to create MLP for training.", m_config.debugPrefix.c_str());
        return;
    }

    // Get training-optimal layout
    m_deviceLayout = m_networkUtils->GetNewMatrixLayout(
        m_network->GetNetworkLayout(), rtxns::MatrixLayout::TrainingOptimal);

    auto hostBufferSize = m_network->GetNetworkLayout().networkByteSize;
    auto deviceBufferSize = m_deviceLayout.networkByteSize;

    assert((deviceBufferSize % sizeof(uint16_t)) == 0);
    m_totalParams = uint32_t(deviceBufferSize / sizeof(uint16_t));

    // Create GPU buffers
    {
        nvrhi::BufferDesc desc;
        const std::string& pfx = m_config.debugPrefix;

        desc.debugName = (pfx + "HostBuffer").c_str();
        desc.initialState = nvrhi::ResourceStates::CopyDest;
        desc.byteSize = hostBufferSize;
        desc.canHaveUAVs = true;
        desc.keepInitialState = true;
        m_hostBuffer = m_device->createBuffer(desc);

        desc.debugName = (pfx + "DeviceBuffer").c_str();
        desc.initialState = nvrhi::ResourceStates::UnorderedAccess;
        desc.byteSize = deviceBufferSize;
        desc.canHaveRawViews = true;
        desc.canHaveTypedViews = true;
        desc.format = nvrhi::Format::R16_FLOAT;
        m_deviceBuffer = m_device->createBuffer(desc);

        desc.debugName = (pfx + "FP32Buffer").c_str();
        desc.canHaveRawViews = false;
        desc.byteSize = m_totalParams * sizeof(float);
        desc.format = nvrhi::Format::R32_FLOAT;
        m_fp32Buffer = m_device->createBuffer(desc);

        desc.debugName = (pfx + "GradientsBuffer").c_str();
        desc.byteSize = (m_totalParams * sizeof(uint16_t) + 3) & ~3;
        desc.canHaveRawViews = true;
        desc.structStride = sizeof(uint16_t);
        desc.format = nvrhi::Format::R16_FLOAT;
        m_gradientsBuffer = m_device->createBuffer(desc);

        desc.debugName = (pfx + "Moments1").c_str();
        desc.byteSize = m_totalParams * sizeof(float);
        desc.format = nvrhi::Format::R32_FLOAT;
        desc.canHaveRawViews = false;
        desc.structStride = 0;
        m_moments1 = m_device->createBuffer(desc);

        desc.debugName = (pfx + "Moments2").c_str();
        m_moments2 = m_device->createBuffer(desc);

        desc.debugName = (pfx + "LossBuffer").c_str();
        desc.byteSize = m_config.batchSize * sizeof(float);
        desc.format = nvrhi::Format::R32_FLOAT;
        desc.canHaveTypedViews = true;
        m_lossBuffer = m_device->createBuffer(desc);
    }

    // Upload initial weights and convert layout
    {
        auto cmdList = m_device->createCommandList();
        cmdList->open();

        cmdList->writeBuffer(m_hostBuffer,
            m_network->GetNetworkParams().data(),
            m_network->GetNetworkParams().size());

        m_networkUtils->ConvertWeights(
            m_network->GetNetworkLayout(), m_deviceLayout,
            m_hostBuffer, 0, m_deviceBuffer, 0,
            m_device, cmdList);

        // Clear gradient + moment buffers
        cmdList->beginTrackingBufferState(m_gradientsBuffer, nvrhi::ResourceStates::UnorderedAccess);
        cmdList->clearBufferUInt(m_gradientsBuffer, 0);
        cmdList->beginTrackingBufferState(m_moments1, nvrhi::ResourceStates::UnorderedAccess);
        cmdList->clearBufferUInt(m_moments1, 0);
        cmdList->beginTrackingBufferState(m_moments2, nvrhi::ResourceStates::UnorderedAccess);
        cmdList->clearBufferUInt(m_moments2, 0);

        cmdList->close();
        m_device->executeCommandList(cmdList);
    }

    // Create training binding layout + set
    {
        nvrhi::BindingLayoutDesc trainLayoutDesc;
        trainLayoutDesc.visibility = nvrhi::ShaderType::Compute;
        trainLayoutDesc.registerSpaceIsDescriptorSet = true;
        trainLayoutDesc.bindings = {
            nvrhi::BindingLayoutItem::ConstantBuffer(0),
            nvrhi::BindingLayoutItem::RawBuffer_SRV(0),
            nvrhi::BindingLayoutItem::RawBuffer_UAV(0),
            nvrhi::BindingLayoutItem::TypedBuffer_UAV(1)
        };
        m_trainingLayout = m_device->createBindingLayout(trainLayoutDesc);

        nvrhi::BindingSetDesc trainSetDesc;
        trainSetDesc.bindings = {
            nvrhi::BindingSetItem::ConstantBuffer(0, m_trainingCB),
            nvrhi::BindingSetItem::RawBuffer_SRV(0, m_deviceBuffer),
            nvrhi::BindingSetItem::RawBuffer_UAV(0, m_gradientsBuffer),
            nvrhi::BindingSetItem::TypedBuffer_UAV(1, m_lossBuffer)
        };
        m_trainingSet = m_device->createBindingSet(trainSetDesc, m_trainingLayout);

        // Training pipeline: our layout + any extra layouts (e.g. cubemap bindings)
        nvrhi::ComputePipelineDesc cpDesc;
        cpDesc.CS = m_trainingCS;
        cpDesc.bindingLayouts = { m_trainingLayout };
        for (auto& layout : extraTrainingLayouts)
            cpDesc.bindingLayouts.push_back(layout);
        m_trainingPipeline = m_device->createComputePipeline(cpDesc);
    }

    // Create optimizer binding layout + set (identical structure for all MLPs)
    {
        nvrhi::BindingLayoutDesc optLayoutDesc;
        optLayoutDesc.visibility = nvrhi::ShaderType::Compute;
        optLayoutDesc.registerSpaceIsDescriptorSet = true;
        optLayoutDesc.bindings = {
            nvrhi::BindingLayoutItem::ConstantBuffer(0),
            nvrhi::BindingLayoutItem::TypedBuffer_UAV(0),
            nvrhi::BindingLayoutItem::TypedBuffer_UAV(1),
            nvrhi::BindingLayoutItem::TypedBuffer_UAV(2),
            nvrhi::BindingLayoutItem::TypedBuffer_UAV(3),
            nvrhi::BindingLayoutItem::TypedBuffer_UAV(4)
        };
        m_optimizerLayout = m_device->createBindingLayout(optLayoutDesc);

        nvrhi::BindingSetDesc optSetDesc;
        optSetDesc.bindings = {
            nvrhi::BindingSetItem::ConstantBuffer(0, m_trainingCB),
            nvrhi::BindingSetItem::TypedBuffer_UAV(0, m_deviceBuffer),
            nvrhi::BindingSetItem::TypedBuffer_UAV(1, m_fp32Buffer),
            nvrhi::BindingSetItem::TypedBuffer_UAV(2, m_gradientsBuffer),
            nvrhi::BindingSetItem::TypedBuffer_UAV(3, m_moments1),
            nvrhi::BindingSetItem::TypedBuffer_UAV(4, m_moments2)
        };
        m_optimizerSet = m_device->createBindingSet(optSetDesc, m_optimizerLayout);

        nvrhi::ComputePipelineDesc cpDesc;
        cpDesc.CS = m_optimizerCS;
        cpDesc.bindingLayouts = { m_optimizerLayout };
        m_optimizerPipeline = m_device->createComputePipeline(cpDesc);
    }

    m_trainingStep = 0;
    m_epoch = 0;
    m_ready = false;

    log::info("[%s] Training resources created: %d params (%d bytes fp16)",
              m_config.debugPrefix.c_str(), m_totalParams, m_totalParams * 2);
}

// =============================================================================
// TrainStep: Run one epoch (batchCount batches) of training + optimizer
// =============================================================================
void NeuralMLP::TrainStep(nvrhi::ICommandList* cmdList,
                          const std::vector<nvrhi::BindingSetHandle>& extraTrainingSets)
{
    std::uniform_int_distribution<uint64_t> ldist;
    uint64_t seed = ldist(m_rd);

    // Build training-layout offset arrays
    uint4 trainWeightOffsets[MAX_TRANSITIONS_ALIGN4] = {};
    uint4 trainBiasOffsets[MAX_TRANSITIONS_ALIGN4] = {};;
    FillTrainOffsets(trainWeightOffsets, trainBiasOffsets);

    for (int batch = 0; batch < m_config.batchCount; ++batch)
    {
        ++m_trainingStep;

        // Write training constants to the CB.
        // Use MAX_TRANSITIONS_ALIGN4 so this struct is always ≤ CB size.
        struct GenericTrainingConstants
        {
            uint4 weightOffsets[MAX_TRANSITIONS_ALIGN4];
            uint4 biasOffsets[MAX_TRANSITIONS_ALIGN4];
            uint32_t maxParamSize;
            float learningRate;
            float currentStep;
            uint32_t batchSize;
            uint64_t seed;
            uint2 _pad;
        } tc = {};

        for (int i = 0; i < MAX_TRANSITIONS_ALIGN4; ++i)
        {
            tc.weightOffsets[i] = trainWeightOffsets[i];
            tc.biasOffsets[i] = trainBiasOffsets[i];
        }
        tc.maxParamSize = m_totalParams;
        tc.learningRate = m_config.learningRate;
        tc.currentStep = float(m_trainingStep);
        tc.batchSize = m_config.batchSize;
        tc.seed = seed + batch;

        cmdList->writeBuffer(m_trainingCB, &tc, sizeof(tc));

        // Training dispatch
        nvrhi::ComputeState trainState;
        trainState.pipeline = m_trainingPipeline;
        trainState.bindings = { m_trainingSet };
        for (auto& extraSet : extraTrainingSets)
            trainState.bindings.push_back(extraSet);
        cmdList->setComputeState(trainState);
        cmdList->dispatch(m_config.batchSize / m_config.threadsPerGroup, 1, 1);

        // Optimizer dispatch
        nvrhi::ComputeState optState;
        optState.pipeline = m_optimizerPipeline;
        optState.bindings = { m_optimizerSet };
        cmdList->setComputeState(optState);
        cmdList->dispatch(
            (m_totalParams + m_config.threadsPerGroupOpt - 1) / m_config.threadsPerGroupOpt, 1, 1);
    }
}

// =============================================================================
// ConvertToInferencing: Training weights → inference-optimal layout
// =============================================================================
void NeuralMLP::ConvertToInferencing()
{
    auto inferLayout = m_networkUtils->GetNewMatrixLayout(
        m_network->GetNetworkLayout(), rtxns::MatrixLayout::InferencingOptimal);

    // Update inference offsets
    memset(m_inferWeightOffsets, 0, sizeof(m_inferWeightOffsets));
    memset(m_inferBiasOffsets, 0, sizeof(m_inferBiasOffsets));
    for (int i = 0; i < m_config.numTransitions; ++i)
    {
        reinterpret_cast<uint32_t*>(m_inferWeightOffsets)[i] = inferLayout.networkLayers[i].weightOffset;
        reinterpret_cast<uint32_t*>(m_inferBiasOffsets)[i] = inferLayout.networkLayers[i].biasOffset;
    }

    // Ensure inference buffer is large enough
    if (!m_inferBuffer || m_inferBuffer->getDesc().byteSize < inferLayout.networkByteSize)
    {
        CreateInferBuffer(inferLayout);
        UpdateInferBindingSet();
        if (onPipelineInvalidated) onPipelineInvalidated();
    }

    // Convert
    auto cmdList = m_device->createCommandList();
    cmdList->open();

    cmdList->setBufferState(m_inferBuffer, nvrhi::ResourceStates::UnorderedAccess);
    cmdList->commitBarriers();

    m_networkUtils->ConvertWeights(
        m_deviceLayout, inferLayout,
        m_deviceBuffer, 0, m_inferBuffer, 0,
        m_device, cmdList);

    cmdList->setBufferState(m_inferBuffer, nvrhi::ResourceStates::ShaderResource);
    cmdList->commitBarriers();

    cmdList->close();
    m_device->executeCommandList(cmdList);
}

// =============================================================================
// LoadWeights: from JSON or native binary
// =============================================================================
bool NeuralMLP::LoadWeights(const std::string& path)
{
    auto loaded = std::make_unique<rtxns::HostNetwork>(m_networkUtils);
    bool success = false;

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
        log::error("[%s] Failed to load weights from: %s", m_config.debugPrefix.c_str(), path.c_str());
        return false;
    }

    m_network = std::move(loaded);

    // Convert to inferencing layout and upload
    auto inferLayout = m_networkUtils->GetNewMatrixLayout(
        m_network->GetNetworkLayout(), rtxns::MatrixLayout::InferencingOptimal);

    memset(m_inferWeightOffsets, 0, sizeof(m_inferWeightOffsets));
    memset(m_inferBiasOffsets, 0, sizeof(m_inferBiasOffsets));
    for (int i = 0; i < m_config.numTransitions; ++i)
    {
        reinterpret_cast<uint32_t*>(m_inferWeightOffsets)[i] = inferLayout.networkLayers[i].weightOffset;
        reinterpret_cast<uint32_t*>(m_inferBiasOffsets)[i] = inferLayout.networkLayers[i].biasOffset;
    }

    if (!m_inferBuffer || m_inferBuffer->getDesc().byteSize < inferLayout.networkByteSize)
    {
        CreateInferBuffer(inferLayout);
        UpdateInferBindingSet();
        if (onPipelineInvalidated) onPipelineInvalidated();
    }

    // Upload
    nvrhi::BufferDesc hostDesc;
    hostDesc.byteSize = m_network->GetNetworkParams().size();
    hostDesc.debugName = (m_config.debugPrefix + "LoadHost").c_str();
    hostDesc.initialState = nvrhi::ResourceStates::CopyDest;
    hostDesc.keepInitialState = true;
    auto hostBuf = m_device->createBuffer(hostDesc);

    auto cmdList = m_device->createCommandList();
    cmdList->open();

    cmdList->writeBuffer(hostBuf, m_network->GetNetworkParams().data(),
                         m_network->GetNetworkParams().size());

    cmdList->setBufferState(m_inferBuffer, nvrhi::ResourceStates::UnorderedAccess);
    cmdList->commitBarriers();

    m_networkUtils->ConvertWeights(
        m_network->GetNetworkLayout(), inferLayout,
        hostBuf, 0, m_inferBuffer, 0,
        m_device, cmdList);

    cmdList->setBufferState(m_inferBuffer, nvrhi::ResourceStates::ShaderResource);
    cmdList->commitBarriers();
    cmdList->close();
    m_device->executeCommandList(cmdList);

    m_ready = true;
    log::info("[%s] Loaded weights from: %s", m_config.debugPrefix.c_str(), path.c_str());
    return true;
}

// =============================================================================
// SaveWeights
// =============================================================================
void NeuralMLP::SaveWeights(const std::string& path, nvrhi::CommandListHandle cmdList)
{
    if (!m_network || !m_deviceBuffer)
    {
        log::warning("[%s] No weights to save.", m_config.debugPrefix.c_str());
        return;
    }

    m_network->UpdateFromBufferToFile(
        m_hostBuffer, m_deviceBuffer,
        m_network->GetNetworkLayout(), m_deviceLayout,
        path, m_device, cmdList);

    log::info("[%s] Saved weights to: %s", m_config.debugPrefix.c_str(), path.c_str());
}

// =============================================================================
// FillInferOffsets: Copy inference weight/bias offsets to caller's arrays
// =============================================================================
void NeuralMLP::FillInferOffsets(uint4* weightOffsets, uint4* biasOffsets) const
{
    for (int i = 0; i < m_config.numTransitionsAlign4; ++i)
    {
        weightOffsets[i] = m_inferWeightOffsets[i];
        biasOffsets[i] = m_inferBiasOffsets[i];
    }
}

// =============================================================================
// Private helpers
// =============================================================================
void NeuralMLP::CreateInferBuffer(const rtxns::NetworkLayout& inferLayout)
{
    nvrhi::BufferDesc bufDesc;
    bufDesc.byteSize = inferLayout.networkByteSize;
    bufDesc.canHaveRawViews = true;
    bufDesc.canHaveUAVs = true;
    bufDesc.debugName = (m_config.debugPrefix + "InferBuffer").c_str();
    bufDesc.initialState = nvrhi::ResourceStates::UnorderedAccess;
    bufDesc.keepInitialState = true;
    m_inferBuffer = m_device->createBuffer(bufDesc);
}

void NeuralMLP::UpdateInferBindingSet()
{
    nvrhi::BindingSetDesc setDesc;
    setDesc.bindings = {
        nvrhi::BindingSetItem::RawBuffer_SRV(0, m_inferBuffer)
    };
    m_inferSet = m_device->createBindingSet(setDesc, m_inferLayout);
}

void NeuralMLP::FillTrainOffsets(uint4* weightOffsets, uint4* biasOffsets) const
{
    memset(weightOffsets, 0, m_config.numTransitionsAlign4 * sizeof(uint4));
    memset(biasOffsets, 0, m_config.numTransitionsAlign4 * sizeof(uint4));
    for (int i = 0; i < m_config.numTransitions; ++i)
    {
        reinterpret_cast<uint32_t*>(weightOffsets)[i] = m_deviceLayout.networkLayers[i].weightOffset;
        reinterpret_cast<uint32_t*>(biasOffsets)[i] = m_deviceLayout.networkLayers[i].biasOffset;
    }
}
