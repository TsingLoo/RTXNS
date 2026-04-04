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

    float3 h = normalize(-lightDir.xyz + view);

    float NdotL = max(0.f, dot(N, -lightDir.xyz));
    float NdotV = max(0.f, dot(N, view));
    float NdotH = max(0.f, dot(N, h));
    float LdotH = max(0.f, dot(h, -lightDir.xyz));

    // ---- Disney BRDF ----
    float4 outParams;
    if (gConst.enableNeuralShading != 0)
    {
        outParams = DisneyMLP(NdotL, NdotV, NdotH, LdotH, roughness);
    }
    else
    {
        outParams = Disney(NdotL, NdotV, NdotH, LdotH, roughness);
    }

    // ---- Final color ----
    float3 Cdlin = float3(pow(max(0.0, baseColor.r), 2.2), pow(max(0.0, baseColor.g), 2.2), pow(max(0.0, baseColor.b), 2.2));
    float3 Cspec0 = lerp(specular * .08f * float3(1,1,1), Cdlin, metallic);
    float3 brdfn = outParams.x * Cdlin * (1 - metallic) + outParams.y * lerp(Cspec0, float3(1), outParams.z) + outParams.w;
    float3 colorh = brdfn * float3(NdotL) * lightIntensity.rgb;

    // ---- Image-Based Lighting ----
    float3 R = reflect(-view, N);

    const float MAX_REFLECTION_LOD = 10.0;
    float3 iblDiffuse = t_Irradiance.SampleLevel(s_Skybox, N, 0.0).rgb;
    
    float lod = roughness * MAX_REFLECTION_LOD;
    float3 iblSpecular = t_Specular.SampleLevel(s_Skybox, R, lod).rgb;
    if (roughness < 0.04) {
        float3 sharpSky = t_Skybox.SampleLevel(s_Skybox, R, lod).rgb;
        iblSpecular = lerp(sharpSky, iblSpecular, clamp(roughness / 0.04, 0.0, 1.0));
    }

    float2 envBRDF = t_BRDFLut.SampleLevel(s_Skybox, clamp(float2(NdotV, roughness), 0.01, 0.99), 0).rg;
    float3 envSpecular = iblSpecular * (Cspec0 * envBRDF.x + envBRDF.y);

    float3 fresnelRoughness = Cspec0 + (max(float3(1.0 - roughness), Cspec0) - Cspec0) * pow(1.0 - NdotV, 5.0);
    float3 kD = (1.0 - fresnelRoughness) * (1.0 - metallic);
    
    float3 PBR_IBL = kD * Cdlin * iblDiffuse + envSpecular;

    // Apply AO to ambient/IBL
    PBR_IBL *= ao;

    // Combine: directional light + IBL + emissive
    o_color = float4(colorh + PBR_IBL + emissive, baseColor.a);
 }
