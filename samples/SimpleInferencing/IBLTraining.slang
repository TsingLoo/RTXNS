/*
 * IBLTraining.slang
 *
 * Training compute shader for the IBL Sampler MLP.
 * Learns GGX importance sampling directions + LOD levels.
 * Ground truth is 100% mathematical — no cubemap needed.
 */

#include "NetworkConfig.h"
#include <donut/shaders/binding_helpers.hlsli>

import CooperativeVectorAutoDiff;
import CooperativeVectorFunctions;
import Utils;
import Activation;
import MLP;
import Loss;
import PCG32;

// =============================================================================
// Bindings (no cubemap — pure math training)
// =============================================================================
DECLARE_CBUFFER(GenericTrainingConstants, gConst, 0, 0);
ByteAddressBuffer gMLPParams               : REGISTER_SRV(0, 0);
RWByteAddressBuffer gMLPParamsGradients    : REGISTER_UAV(0, 0);
RWBuffer<float>     gLossBuffer            : REGISTER_UAV(1, 0);

// =============================================================================
// Math constants and helpers
// =============================================================================
const static float PI = 3.14159265358979323846;

float RadicalInverse_VdC(uint bits)
{
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10;
}

float2 Hammersley(uint i, uint N)
{
    return float2(float(i) / float(N), RadicalInverse_VdC(i));
}

// GGX NDF
float D_GGX(float NdotH, float roughness)
{
    float a = roughness * roughness;
    float a2 = a * a;
    float denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

// ImportanceSampleGGX: returns half-vector H in tangent space (N = [0,0,1])
float3 ImportanceSampleGGX(float2 xi, float roughness)
{
    float a = roughness * roughness;
    float phi = 2.0 * PI * xi.x;
    float cosTheta = sqrt((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y));
    float sinTheta = sqrt(1.0 - cosTheta * cosTheta);

    return float3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
}

// Smith GGX geometry
float GeometrySchlickGGX(float NdotV, float roughness)
{
    float k = (roughness * roughness) / 2.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

float GeometrySmith(float NdotV, float NdotL, float roughness)
{
    return GeometrySchlickGGX(NdotV, roughness) * GeometrySchlickGGX(NdotL, roughness);
}

// =============================================================================
// Main training kernel
// =============================================================================
[shader("compute")]
[numthreads(IBL_THREADS_PER_GROUP, 1, 1)]
void main_cs(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    PCG32 rng = PCG32(gConst.seed, idx);

    // --- Random inputs ---
    float NdotV = rng.nextFloat() * 0.98 + 0.02;    // [0.02, 1.0]
    float roughness = rng.nextFloat() * 0.98 + 0.02; // [0.02, 1.0]

    // V in tangent space (N = [0,0,1])
    float sinTheta_v = sqrt(1.0 - NdotV * NdotV);
    float3 V = float3(sinTheta_v, 0.0, NdotV);
    float3 N = float3(0, 0, 1);

    // =================================================================
    // Ground truth 1: K=4 sampling directions + LOD + weights
    // Sample indices evenly spaced in 1024-sample Hammersley sequence
    // =================================================================
    float gt_cosTheta[IBL_NUM_SAMPLES];
    float gt_phi[IBL_NUM_SAMPLES];
    float gt_lod[IBL_NUM_SAMPLES];
    float gt_weight[IBL_NUM_SAMPLES];
    float weightSum = 0.0;

    float a = roughness * roughness;

    for (int k = 0; k < IBL_NUM_SAMPLES; ++k)
    {
        // Evenly spaced Hammersley indices: 0, 256, 512, 768
        uint sampleIdx = k * (1024 / IBL_NUM_SAMPLES);
        float2 xi = Hammersley(sampleIdx, 1024);

        // Half-vector from GGX importance sampling
        float3 H = ImportanceSampleGGX(xi, roughness);

        // Record H in spherical coordinates (ground truth for MLP output)
        gt_cosTheta[k] = H.z;
        gt_phi[k] = atan2(H.y, H.x); // [-PI, PI]
        // Normalize phi to [0, 1] for training stability
        gt_phi[k] = (gt_phi[k] + PI) / (2.0 * PI);

        // Compute L (cubemap sampling direction)
        float VdotH = max(dot(V, H), 0.0001);
        float3 L = 2.0 * VdotH * H - V;
        float NdotL = max(L.z, 0.0);

        // LOD ground truth — use IBL_NUM_SAMPLES (4), NOT 1024!
        // Each of our K=4 samples covers 256× more solid angle than K=1024,
        // so LOD must be higher to properly blur each sample's contribution.
        float NdotH = max(H.z, 0.0);
        float D = D_GGX(NdotH, roughness);
        float pdf = D * NdotH / (4.0 * VdotH + 0.0001);
        float resolution = IBL_CUBEMAP_SIZE;
        float saTexel = 4.0 * PI / (6.0 * resolution * resolution);
        float saSample = 1.0 / (float(IBL_NUM_SAMPLES) * pdf + 0.0001);
        gt_lod[k] = roughness == 0.0 ? 0.0 : 0.5 * log2(saSample / saTexel);
        gt_lod[k] = max(gt_lod[k], 0.0); // Clamp negative LODs
        // Normalize LOD to [0, 1] range (max mip ~ 10)
        gt_lod[k] = gt_lod[k] / 10.0;

        // Weight = NdotL
        gt_weight[k] = NdotL;
        weightSum += NdotL;
    }

    // Normalize weights
    if (weightSum > 1e-8)
    {
        for (int k = 0; k < IBL_NUM_SAMPLES; ++k)
            gt_weight[k] /= weightSum;
    }

    // =================================================================
    // Ground truth 2: BRDF LUT (Monte Carlo BRDF integration)
    // =================================================================
    float gt_scale = 0.0;
    float gt_bias = 0.0;

    for (uint mc = 0; mc < IBL_MC_SAMPLES; ++mc)
    {
        float2 xi = Hammersley(mc, IBL_MC_SAMPLES);
        float3 H = ImportanceSampleGGX(xi, roughness);
        float3 L = 2.0 * dot(V, H) * H - V;
        float NdotL = max(L.z, 0.0);

        if (NdotL > 0.0)
        {
            float NdotH = max(H.z, 0.0);
            float VdotH = max(dot(V, H), 0.0);
            float G = GeometrySmith(NdotV, NdotL, roughness);
            float G_Vis = G * VdotH / (NdotH * NdotV + 1e-5);
            float Fc = pow(1.0 - VdotH, 5.0);
            gt_scale += (1.0 - Fc) * G_Vis;
            gt_bias += Fc * G_Vis;
        }
    }
    gt_scale /= float(IBL_MC_SAMPLES);
    gt_bias /= float(IBL_MC_SAMPLES);

    // =================================================================
    // MLP Forward
    // =================================================================
    float params[IBL_INPUT_FEATURES] = { NdotV, roughness };
    CoopVec<half, IBL_INPUT_NEURONS> inputParams;
    [ForceUnroll]
    for (int i = 0; i < IBL_INPUT_FEATURES; ++i)
        inputParams[i] = (half)params[i];

    var model = rtxns::mlp::TrainingMLP<half,
        IBL_NUM_HIDDEN_LAYERS, IBL_INPUT_NEURONS, IBL_HIDDEN_NEURONS, IBL_OUTPUT_NEURONS,
        CoopVecMatrixLayout::TrainingOptimal, CoopVecComponentType::Float16>(
        gMLPParams,
        gMLPParamsGradients,
        rtxns::UnpackArray<MAX_TRANSITIONS_ALIGN4, IBL_NUM_TRANSITIONS>(gConst.weightOffsets),
        rtxns::UnpackArray<MAX_TRANSITIONS_ALIGN4, IBL_NUM_TRANSITIONS>(gConst.biasOffsets));

    var hiddenActivation = rtxns::mlp::ReLUAct<half, IBL_HIDDEN_NEURONS>();
    var finalActivation = rtxns::mlp::NoneAct<half, IBL_OUTPUT_NEURONS>();

    var predicted = model.forward(inputParams, hiddenActivation, finalActivation);

    // =================================================================
    // Loss + Backprop
    // =================================================================
    float lossScalePerSample = float(IBL_LOSS_SCALE) / (gConst.batchSize * IBL_OUTPUT_NEURONS);

    float totalError = 0.0;
    rtxns::HCoopVec<IBL_OUTPUT_NEURONS> lossGrad;

    // Per-sample outputs: (cosTheta, phi, lod, weight) × K
    for (int k = 0; k < IBL_NUM_SAMPLES; ++k)
    {
        int base = k * 4;

        float diff_cosTheta = float(predicted[base + 0]) - gt_cosTheta[k];
        float diff_phi      = float(predicted[base + 1]) - gt_phi[k];
        float diff_lod      = float(predicted[base + 2]) - gt_lod[k];
        float diff_weight   = float(predicted[base + 3]) - gt_weight[k];

        // Direction errors are more important than LOD/weight
        totalError += diff_cosTheta * diff_cosTheta * 2.0;
        totalError += diff_phi * diff_phi * 2.0;
        totalError += diff_lod * diff_lod;
        totalError += diff_weight * diff_weight;

        lossGrad[base + 0] = (half)(diff_cosTheta * 4.0 * lossScalePerSample);
        lossGrad[base + 1] = (half)(diff_phi * 4.0 * lossScalePerSample);
        lossGrad[base + 2] = (half)(diff_lod * 2.0 * lossScalePerSample);
        lossGrad[base + 3] = (half)(diff_weight * 2.0 * lossScalePerSample);
    }

    // BRDF LUT outputs (last 2)
    int brdfBase = IBL_NUM_SAMPLES * 4;
    float diff_scale = float(predicted[brdfBase]) - gt_scale;
    float diff_bias  = float(predicted[brdfBase + 1]) - gt_bias;
    totalError += diff_scale * diff_scale + diff_bias * diff_bias;
    lossGrad[brdfBase]     = (half)(diff_scale * 2.0 * lossScalePerSample);
    lossGrad[brdfBase + 1] = (half)(diff_bias * 2.0 * lossScalePerSample);

    model.backward(inputParams, hiddenActivation, finalActivation, lossGrad);

    // Store loss
    gLossBuffer[idx] = sqrt(totalError);
}
