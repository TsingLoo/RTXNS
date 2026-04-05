/*
 * Copyright (c) 2015 - 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

import CooperativeVectorFunctions;
import Utils;
import LinearOps;

#include "NetworkConfig.h"
#include <donut/shaders/binding_helpers.hlsli>

// =============================================================================
// Binding Set 0: Neural network data
// =============================================================================
DECLARE_CBUFFER(NeuralConstants, gConst, 0, 0);
ByteAddressBuffer gMLPParams            :REGISTER_SRV(0, 0);

// =============================================================================
// Binding Set 1: IBL cubemaps 
// =============================================================================
TextureCube<float4> t_Skybox            : REGISTER_SRV(1, 1);
TextureCube<float4> t_Irradiance        : REGISTER_SRV(2, 1);
TextureCube<float4> t_Specular          : REGISTER_SRV(3, 1);
Texture2D<float4>   t_BRDFLut           : REGISTER_SRV(4, 1);

SamplerState s_Skybox                   : REGISTER_SAMPLER(0, 1);

// =============================================================================
// Binding Set 2: Material textures (per-slot bindless)
// =============================================================================
StructuredBuffer<GpuMaterialParams> g_Materials : REGISTER_SRV(0, 2);

// Material textures are bound at t1..tN in space 2, each occupying one SRV slot.
// Slang supports indexing into an unbounded array of textures.
Texture2D<float4> g_Textures[] : REGISTER_SRV(1, 2);

SamplerState s_Material                 : REGISTER_SAMPLER(0, 2);

// =============================================================================
// Binding Set 3: Unified MLP weights (Disney + IBL baked)
// =============================================================================
ByteAddressBuffer gUnifiedMLPParams     : REGISTER_SRV(0, 3);

// =============================================================================
// Vertex structures
// =============================================================================
struct VertexIn
{
    float3 pos : POSITION; 
    float3 norm : NORMAL; 
    float2 uv : TEXCOORD;
    float4 tangent : TANGENT;
    uint materialIndex : MATERIAL_IDX;
};

struct VertexOut
{
    float4 pos : SV_Position;
    float3 norm : NORMAL;
    float3 view : VIEW;
    float2 uv : TEXCOORD0;
    float4 tangent : TANGENT;
    nointerpolation uint materialIndex : MATERIAL_IDX;
}

// =============================================================================
// Vertex Shader
// =============================================================================
[shader("vertex")]
void main_vs(
    VertexIn vIn,
    out VertexOut vOut)
{
    vOut.pos = mul(float4(vIn.pos, 1), gConst.viewProject);
    vOut.norm = vIn.norm;
    vOut.view = gConst.cameraPos.xyz - vIn.pos;
    vOut.uv = vIn.uv;
    vOut.tangent = vIn.tangent;
    vOut.materialIndex = vIn.materialIndex;
}

// =============================================================================
// Disney BRDF
// =============================================================================
const static float DISNEY_PI = 3.14159265358979323846;

float SchlickFresnel(float u)
{
    float m = clamp(1 - u, 0, 1);
    float m2 = m * m;
    return m2 * m2 * m; // pow(m,5)
}

float Gtr1(float NdotH, float a)
{
    if (a >= 1)
    {
        return 1 / DISNEY_PI;
    }
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

    // specular
    float ax = max(.001f, roughness * roughness);
    float Ds = Gtr2(NdotH, ax);
    float FH = SchlickFresnel(LdotH);
    float Gs = SmithGGXAnisotropy(NdotL, ax);
    Gs *= SmithGGXAnisotropy(NdotV, ax);

    // clearcoat (ior = 1.5 -> F0 = 0.04)
    float Dr = Gtr1(NdotH, .01f);
    float Fr = lerp(.04f, 1.0f, FH);
    float Gr = SmithGGX(NdotL, .25f) * SmithGGX(NdotV, .25f);

    return float4((1 / DISNEY_PI) * ss, Gs * Ds, FH, .25 * Gr * Fr * Dr);
}

float4 DisneyMLP(float NdotL, float NdotV, float NdotH, float LdotH, float roughness)
{   
    uint4 weightOffsets = gConst.weightOffsets; 
    uint4 biasOffsets = gConst.biasOffsets;  

   CoopVec<VECTOR_FORMAT, INPUT_NEURONS> inputParams;
   CoopVec<VECTOR_FORMAT, HIDDEN_NEURONS> hiddenParams;
   CoopVec<VECTOR_FORMAT, OUTPUT_NEURONS> outputParams;

   // Encode input parameters, 5 inputs to 30 parameters 
   float params[INPUT_FEATURES] = { NdotL, NdotV, NdotH, LdotH, roughness };
   inputParams = rtxns::EncodeFrequency<half, INPUT_FEATURES>(params);
   
   // Forward propagation through the neural network
   hiddenParams = rtxns::LinearOp<VECTOR_FORMAT, HIDDEN_NEURONS, INPUT_NEURONS>(
       inputParams, gMLPParams, weightOffsets[0], biasOffsets[0], CoopVecMatrixLayout::InferencingOptimal, TYPE_INTERPRETATION);
   hiddenParams = rtxns::relu(hiddenParams);
   
   hiddenParams = rtxns::LinearOp<VECTOR_FORMAT, HIDDEN_NEURONS, HIDDEN_NEURONS>(
       hiddenParams, gMLPParams, weightOffsets[1], biasOffsets[1], CoopVecMatrixLayout::InferencingOptimal, TYPE_INTERPRETATION);
   hiddenParams = rtxns::relu(hiddenParams);
   
   hiddenParams = rtxns::LinearOp<VECTOR_FORMAT, HIDDEN_NEURONS, HIDDEN_NEURONS>(
       hiddenParams, gMLPParams, weightOffsets[2], biasOffsets[2], CoopVecMatrixLayout::InferencingOptimal, TYPE_INTERPRETATION);
   hiddenParams = rtxns::relu(hiddenParams);
   
   outputParams = rtxns::LinearOp<VECTOR_FORMAT, OUTPUT_NEURONS, HIDDEN_NEURONS>(
       hiddenParams, gMLPParams, weightOffsets[3], biasOffsets[3], CoopVecMatrixLayout::InferencingOptimal, TYPE_INTERPRETATION);
   outputParams = exp(outputParams);

    return float4(outputParams[0], outputParams[1], outputParams[2], outputParams[3]);
}

// =============================================================================
// Unified MLP: pure PyTorch offline equivalent (3 layers 64 width)
// =============================================================================
void UnifiedMLP(float NdotL, float NdotV, float VdotL, float thickness, float ao, float roughness, float metallic, out float results[UNIFIED_OUTPUT_NEURONS])
{
    float params[UNIFIED_INPUT_FEATURES] = {
        NdotL, NdotV, VdotL, thickness, ao, roughness, metallic
    };

    CoopVec<VECTOR_FORMAT, UNIFIED_INPUT_NEURONS> inputParams = CoopVec<VECTOR_FORMAT, UNIFIED_INPUT_NEURONS>(VECTOR_FORMAT(0.0));
    CoopVec<VECTOR_FORMAT, UNIFIED_HIDDEN_NEURONS> hiddenParams;
    CoopVec<VECTOR_FORMAT, UNIFIED_OUTPUT_NEURONS> outputParams;

    // Custom Frequency Encoding matching FrequencyEncoder(input_dim=7, freq_bands=2) in PyTorch
    // PyTorch torch.cat(encodings, dim=-1) puts all original vars first, then all sin(1v), then all cos(1v)...
    [ForceUnroll]
    for (int i = 0; i < UNIFIED_INPUT_FEATURES; ++i) {
        float x = params[i];
        inputParams[i] = VECTOR_FORMAT(x);
        inputParams[UNIFIED_INPUT_FEATURES + i] = VECTOR_FORMAT(sin(1.0f * 3.14159265f * x));
        inputParams[UNIFIED_INPUT_FEATURES*2 + i] = VECTOR_FORMAT(cos(1.0f * 3.14159265f * x));
        inputParams[UNIFIED_INPUT_FEATURES*3 + i] = VECTOR_FORMAT(sin(2.0f * 3.14159265f * x));
        inputParams[UNIFIED_INPUT_FEATURES*4 + i] = VECTOR_FORMAT(cos(2.0f * 3.14159265f * x));
    }

    // Layer 0: 35 -> 64
    hiddenParams = rtxns::LinearOp<VECTOR_FORMAT, UNIFIED_HIDDEN_NEURONS, UNIFIED_INPUT_NEURONS>(
        inputParams, gUnifiedMLPParams, gConst.uniWeightOffsets[0].x, gConst.uniBiasOffsets[0].x,
        CoopVecMatrixLayout::InferencingOptimal, TYPE_INTERPRETATION);
    hiddenParams = rtxns::relu(hiddenParams);

    // Layer 1: 64 -> 64
    hiddenParams = rtxns::LinearOp<VECTOR_FORMAT, UNIFIED_HIDDEN_NEURONS, UNIFIED_HIDDEN_NEURONS>(
        hiddenParams, gUnifiedMLPParams, gConst.uniWeightOffsets[0].y, gConst.uniBiasOffsets[0].y,
        CoopVecMatrixLayout::InferencingOptimal, TYPE_INTERPRETATION);
    hiddenParams = rtxns::relu(hiddenParams);

    // Layer 2: 64 -> 64
    hiddenParams = rtxns::LinearOp<VECTOR_FORMAT, UNIFIED_HIDDEN_NEURONS, UNIFIED_HIDDEN_NEURONS>(
        hiddenParams, gUnifiedMLPParams, gConst.uniWeightOffsets[0].z, gConst.uniBiasOffsets[0].z,
        CoopVecMatrixLayout::InferencingOptimal, TYPE_INTERPRETATION);
    hiddenParams = rtxns::relu(hiddenParams);

    // Layer 3 (output): 64 -> 3
    outputParams = rtxns::LinearOp<VECTOR_FORMAT, UNIFIED_OUTPUT_NEURONS, UNIFIED_HIDDEN_NEURONS>(
        hiddenParams, gUnifiedMLPParams, gConst.uniWeightOffsets[0].w, gConst.uniBiasOffsets[0].w,
        CoopVecMatrixLayout::InferencingOptimal, TYPE_INTERPRETATION);

    // PyTorch uses Sigmoid activation on output colors!
    for (int i = 0; i < UNIFIED_OUTPUT_NEURONS; ++i) {
        float rawOutput = float(outputParams[i]);
        results[i] = 1.0f / (1.0f + exp(-rawOutput));
    }
}

// =============================================================================
// Helper: sample a material texture by index, return white if index < 0
// =============================================================================
float4 SampleMaterialTexture(int texIdx, float2 uv)
{
    if (texIdx < 0 || texIdx >= (int)gConst.textureCount)
        return float4(1, 1, 1, 1);
    return g_Textures[texIdx].Sample(s_Material, uv);
}

// =============================================================================
// Fragment Shader
// =============================================================================
[shader("fragment")]
void main_ps( 
    VertexOut vOut,
    out float4 o_color : SV_Target0)
{
    // ---- Resolve material parameters ----
    float4 baseColor;
    float roughness;
    float metallic;
    float specular;
    float thickness = 0.5; // Default thickness
    float3 emissive = float3(0, 0, 0);
    float ao = 1.0;

    if (gConst.usePerVertexMaterial != 0 && gConst.materialCount > 0)
    {
        uint matIdx = min(vOut.materialIndex, gConst.materialCount - 1);
        GpuMaterialParams mat = g_Materials[matIdx];
        
        // Base color: factor * texture
        baseColor = mat.baseColor;
        float4 baseColorTex = SampleMaterialTexture(mat.baseColorTexIdx, vOut.uv);
        baseColor *= baseColorTex;
        
        // Alpha test
        if (mat.alphaMode == 1 && baseColor.a < mat.alphaCutoff)
            discard;
        
        // Metallic-roughness texture (glTF spec: B = metallic, G = roughness)
        roughness = mat.roughness;
        metallic = mat.metallic;
        if (mat.metallicRoughnessTexIdx >= 0)
        {
            float4 mrTex = SampleMaterialTexture(mat.metallicRoughnessTexIdx, vOut.uv);
            roughness *= mrTex.g;
            metallic *= mrTex.b;
        }
        
        specular = mat.specular;
        
        // Occlusion (R channel)
        if (mat.occlusionTexIdx >= 0)
        {
            float aoTex = SampleMaterialTexture(mat.occlusionTexIdx, vOut.uv).r;
            ao = lerp(1.0, aoTex, mat.occlusionStrength);
        }
        
        // Thickness map (KHR_materials_volume) (G channel according to spec or R, cgltf treats it as R usually but let's take G like typical ORM or full RGB)
        if (mat.thicknessTexIdx >= 0)
        {
            thickness = SampleMaterialTexture(mat.thicknessTexIdx, vOut.uv).g;
        }
        
        // Emissive
        emissive = mat.emissiveFactor;
        if (mat.emissiveTexIdx >= 0)
        {
            float3 emTex = SampleMaterialTexture(mat.emissiveTexIdx, vOut.uv).rgb;
            emissive *= emTex;
        }
    }
    else
    {
        // Fallback to global UI parameters
        baseColor = gConst.baseColor;
        roughness = gConst.roughness;
        metallic = gConst.metallic;
        specular = gConst.specular;
    }

    // ---- Normal mapping ----
    float3 N = normalize(vOut.norm);
    float3 view = normalize(vOut.view);

    if (gConst.usePerVertexMaterial != 0 && gConst.materialCount > 0)
    {
        uint matIdx = min(vOut.materialIndex, gConst.materialCount - 1);
        GpuMaterialParams mat = g_Materials[matIdx];
        
        if (mat.normalTexIdx >= 0)
        {
            float3 T = normalize(vOut.tangent.xyz);
            float3 B = cross(N, T) * vOut.tangent.w;
            float3x3 TBN = float3x3(T, B, N);
            
            float3 normalMap = SampleMaterialTexture(mat.normalTexIdx, vOut.uv).rgb;
            normalMap = normalMap * 2.0 - 1.0;
            normalMap.xy *= mat.normalScale;
            N = normalize(mul(normalMap, TBN));
        }
    }

    // ---- Lighting setup ----
    float4 lightIntensity = gConst.lightIntensity;
    float4 lightDir = gConst.lightDir;

    // ---- Final Color Output Variables ----
    float4 outParams;
    float3 iblDiffuse;
    float3 iblSpecular;
    float2 envBRDF;

    float3 h = normalize(-lightDir.xyz + view);

    // Allow NdotL to be negative to evaluate backlight scattering
    float NdotL = dot(N, -lightDir.xyz);
    float NdotV = max(1e-5f, dot(N, view));
    float NdotH = max(1e-5f, dot(N, h));
    float LdotH = max(1e-5f, dot(h, -lightDir.xyz));
    float VdotL = dot(view, -lightDir.xyz);

    // ---- IBL Texture Sampling (Always evaluated but mathematical LUT is predicted by MLP if enabled) ----
    float3 R = reflect(-view, N);
    const float MAX_REFLECTION_LOD = 10.0;
    
    // Diffuse
    iblDiffuse = t_Irradiance.SampleLevel(s_Skybox, N, 0.0).rgb;
    
    // Specular
    float lod = roughness * MAX_REFLECTION_LOD;
    iblSpecular = t_Specular.SampleLevel(s_Skybox, R, lod).rgb;
    if (roughness < 0.04) {
        float3 sharpSky = t_Skybox.SampleLevel(s_Skybox, R, lod).rgb;
        iblSpecular = lerp(sharpSky, iblSpecular, clamp(roughness / 0.04, 0.0, 1.0));
    }

    // ---- Check for unified neural shading mode ----
    float3 sssTransmittance = float3(0, 0, 0);

    // Only apply to objects that have a thickness map, or fallback to true for procedural sphere
    bool isNeuralTarget = true;
    if (gConst.usePerVertexMaterial != 0 && gConst.materialCount > 0) {
        uint matIdx = min(vOut.materialIndex, gConst.materialCount - 1);
        isNeuralTarget = (g_Materials[matIdx].thicknessTexIdx >= 0);
    }

    if (gConst.enableNeuralIBL != 0 && isNeuralTarget)
    {
        // UNIVERSAL MLP: perfectly replaces rendering logic predicting the pure RGB
        float mlpResults[UNIFIED_OUTPUT_NEURONS];
        // The Python dataset was strictly trained with roughness=0.15 and metallic=0.0
        // Feeding anything else (like Metallic 1.0 from the UI) triggers untrained explosive noise!
        UnifiedMLP(NdotL, NdotV, VdotL, thickness, ao, 0.15f, 0.0f, mlpResults);
        
        // Output from MLP is Sigmoid bound [0, 1].
        // Scale with the recent 5.0 clipped y_max, BUT ALSO multiply by UI light intensity!
        float3 y_scale = float3(5.0, 5.0, 5.0); 
        float3 neuralDiffuse = float3(mlpResults[0], mlpResults[1], mlpResults[2]) * y_scale; 
        
        // Allow the UI Light Intensity (and an ambient floor of 0.2) to scale the brightness!
        neuralDiffuse *= (gConst.lightIntensity.rgb + 0.2);
        
        // -------------------------------------------------------------
        // THE GOLDEN RULE OF NEURAL RENDERING: Neural Diffuse + Physical Specular
        // -------------------------------------------------------------
        // We use the Neural Network's prediction purely as the "Subsurface DiffuseBase".
        // Now we calculate the exact same mathematical sharp SPECULAR highlight as the engine uses!
        
        // Directional Light Specular (Microfacet BRDF math)
        float3 H = normalize(V + L);
        float NdotH = saturate(dot(N, H));
        float LdotH = saturate(dot(L, H));
        
        float a = roughness * roughness;
        float a2 = a * a;
        float d = (NdotH * NdotH) * (a2 - 1.0) + 1.0;
        float D = a2 / (3.14159265 * d * d);
        
        float k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
        float GV = NdotV / (NdotV * (1.0 - k) + k);
        float GL = NdotL / (NdotL * (1.0 - k) + k);
        float G = GV * GL;
        
        float3 F0 = lerp(float3(0.04, 0.04, 0.04), baseColor.rgb, metallic);
        float3 F = F0 + (1.0 - F0) * pow(1.0 - LdotH, 5.0);
        
        float3 specularTerm = (D * G * F) / max(4.0 * NdotV * NdotL, 0.001) * NdotL;
        // Multiply by the raw light energy
        float3 analyticalSpecular = specularTerm * gConst.lightIntensity.rgb * 3.14159f;
        
        // Combine them: The beautiful MLP subsurface flow + the razor-sharp physical specular
        float3 finalColor = neuralDiffuse + analyticalSpecular;
        
        // Add IBL Specular Reflections for environmental wetness
        finalColor += iblSpecular * F0 * ao * gConst.lightIntensity.rgb;

        // ACES Film Tonemapping to safely compress blinding highlights into smooth white
        float3 x = max(0, finalColor - 0.004);
        finalColor = (x * (6.2 * x + 0.5)) / (x * (6.2 * x + 1.7) + 0.06);

        // Override Final Output
        o_color = float4(finalColor, baseColor.a);
        return;
    }
    else
    {
        // ---- Disney BRDF ----
        if (gConst.enableNeuralShading != 0) {
            outParams = DisneyMLP(NdotL, NdotV, NdotH, LdotH, roughness);
        } else {
            outParams = Disney(NdotL, NdotV, NdotH, LdotH, roughness);
        }

        // BRDF LUT
        envBRDF = t_BRDFLut.SampleLevel(s_Skybox, clamp(float2(NdotV, roughness), 0.01, 0.99), 0).rg;
        
        // ---- Analytical Sub-Surface Scattering Base Math (Ground Truth Fallback) ----
        float3 SCATTER_COLOR = float3(1.5, 4.0, 1.0);
        float wrap = 0.8;
        float wrappedL = max(0.0, (NdotL + wrap) / (1.0 + wrap));
        float phase = max(0.0, VdotL * 0.5 + 0.5);
        float depth = max(0.01, thickness * 2.0); 
        sssTransmittance = exp(-depth / SCATTER_COLOR) * wrappedL * phase * 3.5;
    }

    // ---- Arithmetic Composition (Linear Albedo * Lighting) ----
    float3 Cdlin = float3(pow(max(0.0, baseColor.r), 2.2), pow(max(0.0, baseColor.g), 2.2), pow(max(0.0, baseColor.b), 2.2));
    float3 Cspec0 = lerp(specular * .08f * float3(1,1,1), Cdlin, metallic);
    
    // Directional (Forward + Backlight SSS)
    // For standard BRDF, clamp NdotL. For SSS, use raw light direction.
    float forwardNdotL = max(0.0, NdotL);
    float3 brdfn = outParams.x * Cdlin * (1 - metallic) + outParams.y * lerp(Cspec0, float3(1), outParams.z) + outParams.w;
    float3 colorh = brdfn * forwardNdotL * lightIntensity.rgb;
    
    // Add Subsurface Scattering Glow
    float3 sssBaseColor = float3(0.6, 0.9, 0.4); // Jade hue
    float3 effectiveGlowColor = lerp(Cdlin, sssBaseColor, 0.8); // Trust the SSS base color heavily
    
    // Greatly boost the transmittence glow so it feels like soft jade instead of hard ceramic
    float3 sssGlow = sssTransmittance * effectiveGlowColor * lightIntensity.rgb * 5.0;
    colorh += sssGlow;

    // Ambient / IBL
    float3 envSpecularFinal = iblSpecular * (Cspec0 * envBRDF.x + envBRDF.y);
    float3 fresnelRoughness = Cspec0 + (max(float3(1.0 - roughness), Cspec0) - Cspec0) * pow(1.0 - NdotV, 5.0);
    float3 kD = (1.0 - fresnelRoughness) * (1.0 - metallic);
    float3 PBR_IBL = kD * Cdlin * iblDiffuse + envSpecularFinal;

    // Apply AO to ambient/IBL
    PBR_IBL *= ao;

    // Combine: directional light + IBL + emissive
    o_color = float4(colorh + PBR_IBL + emissive, baseColor.a);
 }
