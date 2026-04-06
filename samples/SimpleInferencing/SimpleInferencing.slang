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

// =============================================================================
// Unified MLP: Residual SSS network (5 layers: input_proj + 3 blocks + output)
// Residual skip connections are fused into exported weights (W_eff = W + I),
// so inference remains a simple sequential feedforward pass.
// =============================================================================
void UnifiedMLP(float NdotL, float NdotV, float VdotL, float thickness, float ao, float curvature, out float results[UNIFIED_OUTPUT_NEURONS])
{
    // Compute 5 SSS-physics-informed features from existing inputs
    float wrap_lighting = saturate(NdotL * 0.5 + 0.5);
    float transmission_feat = exp(-thickness * 3.0) * max(-NdotL, 0.0);
    float forward_scatter = pow(max(VdotL, 0.0), 4.0);
    float thin_backlight = (1.0 - thickness) * max(-NdotL, 0.0);
    float fresnel = pow(1.0 - NdotV, 5.0);

    float params[UNIFIED_INPUT_FEATURES] = {
        NdotL, NdotV, VdotL, thickness, ao, curvature,
        wrap_lighting, transmission_feat, forward_scatter, thin_backlight, fresnel
    };

    CoopVec<VECTOR_FORMAT, UNIFIED_INPUT_NEURONS> inputParams;
    CoopVec<VECTOR_FORMAT, UNIFIED_HIDDEN_NEURONS> hiddenParams;
    CoopVec<VECTOR_FORMAT, UNIFIED_OUTPUT_NEURONS> outputParams;

    [ForceUnroll]
    for (int i = 0; i < UNIFIED_INPUT_FEATURES; ++i) {
        inputParams[i] = VECTOR_FORMAT(params[i]);
    }

    // Layer 0 (input_proj): 11 -> 256
    hiddenParams = rtxns::LinearOp<VECTOR_FORMAT, UNIFIED_HIDDEN_NEURONS, UNIFIED_INPUT_NEURONS>(
        inputParams, gUnifiedMLPParams, gConst.uniWeightOffsets[0].x, gConst.uniBiasOffsets[0].x,
        CoopVecMatrixLayout::InferencingOptimal, TYPE_INTERPRETATION);
    [ForceUnroll] for (int act_i = 0; act_i < UNIFIED_HIDDEN_NEURONS; ++act_i) {
        hiddenParams[act_i] = log(VECTOR_FORMAT(1.0) + exp(hiddenParams[act_i]));
    }

    // Layer 1 (block1 + residual skip): 256 -> 256
    CoopVec<VECTOR_FORMAT, UNIFIED_HIDDEN_NEURONS> savedHidden1 = hiddenParams;
    hiddenParams = rtxns::LinearOp<VECTOR_FORMAT, UNIFIED_HIDDEN_NEURONS, UNIFIED_HIDDEN_NEURONS>(
        hiddenParams, gUnifiedMLPParams, gConst.uniWeightOffsets[0].y, gConst.uniBiasOffsets[0].y,
        CoopVecMatrixLayout::InferencingOptimal, TYPE_INTERPRETATION);
    [ForceUnroll] for (int act_i = 0; act_i < UNIFIED_HIDDEN_NEURONS; ++act_i) {
        hiddenParams[act_i] = log(VECTOR_FORMAT(1.0) + exp(hiddenParams[act_i])) + savedHidden1[act_i];
    }

    // Layer 2 (block2 + residual skip): 256 -> 256
    CoopVec<VECTOR_FORMAT, UNIFIED_HIDDEN_NEURONS> savedHidden2 = hiddenParams;
    hiddenParams = rtxns::LinearOp<VECTOR_FORMAT, UNIFIED_HIDDEN_NEURONS, UNIFIED_HIDDEN_NEURONS>(
        hiddenParams, gUnifiedMLPParams, gConst.uniWeightOffsets[0].z, gConst.uniBiasOffsets[0].z,
        CoopVecMatrixLayout::InferencingOptimal, TYPE_INTERPRETATION);
    [ForceUnroll] for (int act_i = 0; act_i < UNIFIED_HIDDEN_NEURONS; ++act_i) {
        hiddenParams[act_i] = log(VECTOR_FORMAT(1.0) + exp(hiddenParams[act_i])) + savedHidden2[act_i];
    }

    // Layer 3 (block3 + residual skip): 256 -> 256
    CoopVec<VECTOR_FORMAT, UNIFIED_HIDDEN_NEURONS> savedHidden3 = hiddenParams;
    hiddenParams = rtxns::LinearOp<VECTOR_FORMAT, UNIFIED_HIDDEN_NEURONS, UNIFIED_HIDDEN_NEURONS>(
        hiddenParams, gUnifiedMLPParams, gConst.uniWeightOffsets[0].w, gConst.uniBiasOffsets[0].w,
        CoopVecMatrixLayout::InferencingOptimal, TYPE_INTERPRETATION);
    [ForceUnroll] for (int act_i = 0; act_i < UNIFIED_HIDDEN_NEURONS; ++act_i) {
        hiddenParams[act_i] = log(VECTOR_FORMAT(1.0) + exp(hiddenParams[act_i])) + savedHidden3[act_i];
    }

    // Layer 4 (output): 256 -> 3
    outputParams = rtxns::LinearOp<VECTOR_FORMAT, UNIFIED_OUTPUT_NEURONS, UNIFIED_HIDDEN_NEURONS>(
        hiddenParams, gUnifiedMLPParams, gConst.uniWeightOffsets[1].x, gConst.uniBiasOffsets[1].x,
        CoopVecMatrixLayout::InferencingOptimal, TYPE_INTERPRETATION);

    // The model outputs raw unbounded HDR predictions. We only prevent negative values.
    for (int i = 0; i < UNIFIED_OUTPUT_NEURONS; ++i) {
        float rawOutput = float(outputParams[i]);
        results[i] = max(rawOutput, 0.0f);
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
    float curvature = 0.0;

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
        
        // Thickness map (standalone PNG: grayscale in R channel)
        if (mat.thicknessTexIdx >= 0)
        {
            thickness = SampleMaterialTexture(mat.thicknessTexIdx, vOut.uv).r;
        }

        // Curvature map (hijacked from clearcoat)
        if (mat.curvatureTexIdx >= 0)
        {
            curvature = SampleMaterialTexture(mat.curvatureTexIdx, vOut.uv).r;
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

    // Apply neural shading universally when enabled
    bool isNeuralTarget = true;
    
    if (gConst.enableNeuralIBL != 0 && isNeuralTarget)
    {
        // UNIVERSAL MLP: perfectly replaces rendering logic predicting the pure RGB
        float mlpResults[UNIFIED_OUTPUT_NEURONS];
        
        UnifiedMLP(NdotL, NdotV, VdotL, thickness, ao, curvature, mlpResults);
        
        // Output from MLP is unit-intensity SSS transmittance (trained with Sun energy=1.0).
        // Multiply by base color to get the material-colored SSS response.
        float3 hdrTransmittance = max(float3(mlpResults[0], mlpResults[1], mlpResults[2]), 0.0f);
        float3 neuralDiffuse = hdrTransmittance * baseColor.rgb; 
        
        // Scale purely by user light intensity — NO ambient floor.
        // When lightIntensity=0, directional SSS should be zero (environment light handled by IBL).
        neuralDiffuse *= gConst.lightIntensity.rgb;
        
        // -------------------------------------------------------------
        // THE GOLDEN RULE OF NEURAL RENDERING: Neural Diffuse + Physical Specular
        // -------------------------------------------------------------
        // We use the Neural Network's prediction purely as the "Subsurface DiffuseBase".
        // Now we calculate the exact same mathematical sharp SPECULAR highlight as the engine uses!
        
        // Directional Light Specular (Microfacet BRDF math)
        // Specular only exists when the surface faces the light (NdotL > 0).
        // The neural MLP handles back-scattering through its SSS diffuse output.
        float3 F0 = lerp(float3(0.04, 0.04, 0.04), baseColor.rgb, metallic);
        float specNdotL = max(NdotL, 0.0);
        float3 analyticalSpecular = float3(0, 0, 0);
        
        if (specNdotL > 1e-5f)
        {
            float3 V_dir = view;
            float3 L_dir = normalize(-lightDir.xyz);
            float3 H = normalize(V_dir + L_dir);
            float NdotH_spec = saturate(dot(N, H));
            float LdotH_spec = saturate(dot(L_dir, H));
            
            float a = roughness * roughness;
            float a2 = a * a;
            float d = (NdotH_spec * NdotH_spec) * (a2 - 1.0) + 1.0;
            float D = a2 / (3.14159265 * d * d);
            
            float k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
            float GV = NdotV / (NdotV * (1.0 - k) + k);
            float GL = specNdotL / (specNdotL * (1.0 - k) + k);
            float G = GV * GL;
            
            float3 F = F0 + (1.0 - F0) * pow(1.0 - LdotH_spec, 5.0);
            
            float3 specularTerm = (D * G * F) / max(4.0 * NdotV * specNdotL, 0.001) * specNdotL;
            analyticalSpecular = specularTerm * gConst.lightIntensity.rgb * 3.14159f;
        }
        
        // Combine: neural SSS diffuse + physical specular (both scaled by directional light)
        float3 finalColor = neuralDiffuse + analyticalSpecular;
        
        // ----- IBL (Environment lighting) — independent of directional light -----
        // This provides ambient illumination so the object isn't black when lightIntensity=0.
        
        // IBL Diffuse: ambient SSS/diffuse lighting from environment irradiance
        // Same logic as non-neural path: kD * baseColor * irradiance * ao
        float3 fresnelRoughness_n = F0 + (max(float3(1.0 - roughness), F0) - F0) * pow(1.0 - NdotV, 5.0);
        float3 kD_n = (1.0 - fresnelRoughness_n) * (1.0 - metallic);
        float3 Cdlin_n = float3(pow(max(0.0, baseColor.r), 2.2), pow(max(0.0, baseColor.g), 2.2), pow(max(0.0, baseColor.b), 2.2));
        finalColor += kD_n * Cdlin_n * iblDiffuse * ao;
        
        // IBL Specular: environment reflections using the BRDF LUT
        float2 envBRDF_n = t_BRDFLut.SampleLevel(s_Skybox, clamp(float2(NdotV, roughness), 0.01, 0.99), 0).rg;
        finalColor += iblSpecular * (F0 * envBRDF_n.x + envBRDF_n.y) * ao;

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
        // Disney BRDF math collapses (division by zero / NaNs) if NdotL is negative.
        // Neural network supports negative NdotL, but the analytical fallback must be clamped.
        outParams = Disney(max(1e-5f, NdotL), NdotV, clamp(NdotH, 0.0, 1.0), LdotH, roughness);

        // BRDF LUT
        envBRDF = t_BRDFLut.SampleLevel(s_Skybox, clamp(float2(NdotV, roughness), 0.01, 0.99), 0).rg;
    }

    // ---- Arithmetic Composition (Linear Albedo * Lighting) ----
    float3 Cdlin = float3(pow(max(0.0, baseColor.r), 2.2), pow(max(0.0, baseColor.g), 2.2), pow(max(0.0, baseColor.b), 2.2));
    float3 Cspec0 = lerp(specular * .08f * float3(1,1,1), Cdlin, metallic);
    
    // Directional
    float forwardNdotL = max(0.0, NdotL);
    float3 brdfn = outParams.x * Cdlin * (1 - metallic) + outParams.y * lerp(Cspec0, float3(1), outParams.z) + outParams.w;
    float3 colorh = brdfn * forwardNdotL * lightIntensity.rgb;

    // Ambient / IBL
    float3 envSpecularFinal = iblSpecular * (Cspec0 * envBRDF.x + envBRDF.y);
    float3 fresnelRoughness = Cspec0 + (max(float3(1.0 - roughness), Cspec0) - Cspec0) * pow(1.0 - NdotV, 5.0);
    float3 kD = (1.0 - fresnelRoughness) * (1.0 - metallic);
    float3 PBR_IBL = kD * Cdlin * iblDiffuse + envSpecularFinal;

    // Apply AO to ambient/IBL
    PBR_IBL *= ao;

    // Combine: directional light + IBL + emissive
    float3 finalColor = colorh + PBR_IBL + emissive;
    
    // ACES Film Tonemapping to safely compress blinding highlights into smooth gradient
    float3 t_x = max(0, finalColor - 0.004);
    finalColor = (t_x * (6.2 * t_x + 0.5)) / (t_x * (6.2 * t_x + 1.7) + 0.06);

    o_color = float4(finalColor, baseColor.a);
 }
