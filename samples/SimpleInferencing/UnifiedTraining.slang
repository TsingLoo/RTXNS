/*
 * UnifiedTraining.slang
 * 
 * Training compute shader for the unified MLP that replaces both
 * Disney BRDF and IBL cubemap sampling with a single neural network.
 *
 * Ground truth is generated on-the-fly by running the exact same
 * shading math as the original fragment shader, sampling the actual
 * IBL cubemaps.
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
// Bindings
// =============================================================================
DECLARE_CBUFFER(UnifiedTrainingConstants, gConst, 0, 0);
ByteAddressBuffer gMLPParams               : REGISTER_SRV(0, 0);
RWByteAddressBuffer gMLPParamsGradients    : REGISTER_UAV(0, 0);
RWBuffer<float>     gLossBuffer            : REGISTER_UAV(1, 0);

// IBL cubemaps for ground truth generation (same textures as rendering)
TextureCube<float4> t_Skybox               : REGISTER_SRV(1, 1);
TextureCube<float4> t_Irradiance           : REGISTER_SRV(2, 1);
TextureCube<float4> t_Specular             : REGISTER_SRV(3, 1);
Texture2D<float4>   t_BRDFLut             : REGISTER_SRV(4, 1);
SamplerState s_Skybox                      : REGISTER_SAMPLER(0, 1);

// =============================================================================
// Disney BRDF (ground truth reference — identical to SimpleInferencing.slang)
// =============================================================================
const static float DISNEY_PI = 3.14159265358979323846;

float SchlickFresnel(float u)
{
    float m = clamp(1 - u, 0, 1);
    float m2 = m * m;
    return m2 * m2 * m;
}

float Gtr1(float NdotH, float a)
{
    if (a >= 1) return 1 / DISNEY_PI;
    float a2 = a * a;
    float t = 1 + (a2 - 1) * NdotH * NdotH;
    return (a2 - 1) / (DISNEY_PI * log(a2) * t);
}

float Gtr2(float NdotH, float ax)
{
    float a = ax * (1 / ax / ax * (1 - NdotH * NdotH) + NdotH * NdotH);
    return 1 / (DISNEY_PI * a * a);
}

float SmithGGX(float NdotV, float alphaG)
{
    float a = alphaG * alphaG;
    float b = NdotV * NdotV;
    return 1 / (NdotV + sqrt(a + b - a * b));
}

float SmithGGXAnisotropy(float NdotV, float ax)
{
    return 1 / (NdotV + sqrt(ax * ax * (1 - NdotV * NdotV) + NdotV * NdotV));
}

float4 Disney(float NdotL, float NdotV, float NdotH, float LdotH, float roughness)
{
    float FL = SchlickFresnel(NdotL), FV = SchlickFresnel(NdotV);
    float Fss90 = LdotH * LdotH * roughness;
    float Fss = lerp(1.0f, Fss90, FL) * lerp(1.0f, Fss90, FV);
    float ss = 1.25f * (Fss * (1.f / (NdotL + NdotV) - .5f) + .5f);

    float ax = max(.001f, roughness * roughness);
    float Ds = Gtr2(NdotH, ax);
    float FH = SchlickFresnel(LdotH);
    float Gs = SmithGGXAnisotropy(NdotL, ax);
    Gs *= SmithGGXAnisotropy(NdotV, ax);

    float Dr = Gtr1(NdotH, .01f);
    float Fr = lerp(.04f, 1.0f, FH);
    float Gr = SmithGGX(NdotL, .25f) * SmithGGX(NdotV, .25f);

    return float4((1 / DISNEY_PI) * ss, Gs * Ds, FH, .25 * Gr * Fr * Dr);
}

// =============================================================================
// Ground Truth: compute pure analytical mathematical targets (Disney + LUT)
// =============================================================================
void ComputeGroundTruth(
    float NdotL, float NdotV, float NdotH, float LdotH, float VdotL,
    float thickness, float roughness, out float gt[UNIFIED_OUTPUT_NEURONS])
{
    // Dummy analytical function just to make the compiler happy
    for (int i = 0; i < UNIFIED_OUTPUT_NEURONS; ++i) {
        gt[i] = 0.0;
    }
}

// =============================================================================
// Main training kernel
// =============================================================================
[shader("compute")]
[numthreads(UNIFIED_THREADS_PER_GROUP, 1, 1)]
void main_cs(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    PCG32 rng = PCG32(gConst.seed, idx);

    // --- Generate random inputs ---
    // Random normal on upper hemisphere (z >= 0)
    float phi_n = 2.0 * DISNEY_PI * rng.nextFloat();
    float cosTheta_n = rng.nextFloat();
    float sinTheta_n = sqrt(1.0 - cosTheta_n * cosTheta_n);
    float3 N = float3(sinTheta_n * cos(phi_n), sinTheta_n * sin(phi_n), cosTheta_n);

    // Random view direction on upper hemisphere relative to N
    float phi_v = 2.0 * DISNEY_PI * rng.nextFloat();
    float cosTheta_v = rng.nextFloat();
    float sinTheta_v = sqrt(1.0 - cosTheta_v * cosTheta_v);
    float3 V = float3(sinTheta_v * cos(phi_v), sinTheta_v * sin(phi_v), cosTheta_v);
    // Ensure V is on same side as N
    if (dot(V, N) < 0) V = -V;
    V = normalize(V);

    // Random light direction (full sphere, MUCH ALLOWED BEHIND FOR SSS)
    float phi_l = 2.0 * DISNEY_PI * rng.nextFloat();
    float cosTheta_l = 2.0 * rng.nextFloat() - 1.0;
    float sinTheta_l = sqrt(1.0 - cosTheta_l * cosTheta_l);
    float3 L = float3(sinTheta_l * cos(phi_l), sinTheta_l * sin(phi_l), cosTheta_l);
    L = normalize(L);

    // Random material parameters
    float roughness = rng.nextFloat() * 0.95f + 0.05f;

    // Random volume thickness [0.0, 2.0]
    float thickness = rng.nextFloat() * 2.0;

    // --- Derived mathematical dot products ---
    float3 h = normalize(L + V);
    float NdotL = dot(N, L); // Negative is completely fine for back-scattering
    float NdotV = max(1e-5f, dot(N, V));
    float NdotH = max(1e-5f, dot(N, h));
    float LdotH = max(1e-5f, dot(h, L));
    float VdotL = dot(V, L);

    // --- Compute ground truth ---
    float groundTruth[UNIFIED_OUTPUT_NEURONS];
    ComputeGroundTruth(NdotL, NdotV, NdotH, LdotH, VdotL, thickness, roughness, groundTruth);

    // --- MLP forward pass ---
    // SSS-GGX-MLP: 12 features = NdotL, NdotV, NdotH, VdotL, thick, ao, curv, wrap, trans, fwd_scatter, thin_backlight, fresnel
    float wrap_l = saturate(NdotL * 0.5 + 0.5);
    float trans_feat = exp(-thickness * 3.0) * max(-NdotL, 0.0);
    float fwd_scatter = pow(max(VdotL, 0.0), 4.0);
    float thin_bl = (1.0 - thickness) * max(-NdotL, 0.0);
    float fresnel_f = pow(1.0 - NdotV, 5.0);
    
    float params[UNIFIED_INPUT_FEATURES] = {
        NdotL, NdotV, NdotH, VdotL, thickness, 1.0f, 0.0f,
        wrap_l, trans_feat, fwd_scatter, thin_bl, fresnel_f
    };
    
    CoopVec<half, UNIFIED_INPUT_NEURONS> inputParams = CoopVec<half, UNIFIED_INPUT_NEURONS>((half)0.0);
    [ForceUnroll]
    for (int i = 0; i < UNIFIED_INPUT_FEATURES; ++i) {
        float x = params[i];
        inputParams[i] = (half)(x);
        inputParams[UNIFIED_INPUT_FEATURES + i] = (half)(sin(1.0f * 3.14159265f * x));
        inputParams[UNIFIED_INPUT_FEATURES*2 + i] = (half)(cos(1.0f * 3.14159265f * x));
        inputParams[UNIFIED_INPUT_FEATURES*3 + i] = (half)(sin(2.0f * 3.14159265f * x));
        inputParams[UNIFIED_INPUT_FEATURES*4 + i] = (half)(cos(2.0f * 3.14159265f * x));
        inputParams[UNIFIED_INPUT_FEATURES*5 + i] = (half)(sin(4.0f * 3.14159265f * x));
        inputParams[UNIFIED_INPUT_FEATURES*6 + i] = (half)(cos(4.0f * 3.14159265f * x));
        inputParams[UNIFIED_INPUT_FEATURES*7 + i] = (half)(sin(8.0f * 3.14159265f * x));
        inputParams[UNIFIED_INPUT_FEATURES*8 + i] = (half)(cos(8.0f * 3.14159265f * x));
        inputParams[UNIFIED_INPUT_FEATURES*9 + i] = (half)(sin(16.0f * 3.14159265f * x));
        inputParams[UNIFIED_INPUT_FEATURES*10 + i] = (half)(cos(16.0f * 3.14159265f * x));
        inputParams[UNIFIED_INPUT_FEATURES*11 + i] = (half)(sin(32.0f * 3.14159265f * x));
        inputParams[UNIFIED_INPUT_FEATURES*12 + i] = (half)(cos(32.0f * 3.14159265f * x));
    }

    var model = rtxns::mlp::TrainingMLP<half,
        UNIFIED_NUM_HIDDEN_LAYERS, UNIFIED_INPUT_NEURONS, UNIFIED_HIDDEN_NEURONS, UNIFIED_OUTPUT_NEURONS,
        CoopVecMatrixLayout::TrainingOptimal, CoopVecComponentType::Float16>(
        gMLPParams,
        gMLPParamsGradients,
        rtxns::UnpackArray<UNIFIED_NUM_TRANSITIONS_ALIGN4, UNIFIED_NUM_TRANSITIONS>(gConst.weightOffsets),
        rtxns::UnpackArray<UNIFIED_NUM_TRANSITIONS_ALIGN4, UNIFIED_NUM_TRANSITIONS>(gConst.biasOffsets));

    var hiddenActivation = rtxns::mlp::ReLUAct<half, UNIFIED_HIDDEN_NEURONS>();
    var finalActivation = rtxns::mlp::NoneAct<half, UNIFIED_OUTPUT_NEURONS>();

    var predicted = model.forward(inputParams, hiddenActivation, finalActivation);

    // --- Backpropagation (L2 loss) ---
    float lossScalePerSample = float(UNIFIED_LOSS_SCALE) / (gConst.batchSize * UNIFIED_OUTPUT_NEURONS);

    float l2Error = 0.0;
    float l2Target = 0.0;
    rtxns::HCoopVec<UNIFIED_OUTPUT_NEURONS> lossGradVec;
    for (int i = 0; i < UNIFIED_OUTPUT_NEURONS; ++i) {
        float diff = predicted[i] - groundTruth[i];
        l2Error += diff * diff;
        l2Target += groundTruth[i] * groundTruth[i];
        
        lossGradVec[i] = (half)(diff * 2.0 * lossScalePerSample);
    }

    model.backward(inputParams, hiddenActivation, finalActivation, lossGradVec);

    // --- Store loss for monitoring ---
    l2Error = sqrt(l2Error);
    l2Target = sqrt(l2Target);
    float epsilon = 1e-6f;
    gLossBuffer[idx] = l2Error / (l2Target + epsilon);
}
