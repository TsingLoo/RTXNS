/*
 * Copyright (c) 2015 - 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef __NETWORK_CONFIG_H__
#define __NETWORK_CONFIG_H__

#define VECTOR_FORMAT half
#define TYPE_INTERPRETATION CoopVecComponentType::Float16

// When loading a model from file, these parameters must match
#define INPUT_FEATURES 5
#define INPUT_NEURONS (INPUT_FEATURES * 6) // Frequency encoding increases the input by 6 for each input
#define OUTPUT_NEURONS 4
#define HIDDEN_NEURONS 32

// --- Universal Math MLP (PyTorch Equivalent) ---
// Input: NdotL, NdotV, NdotH, VdotL, roughness, metallic, specular, thick, ao, curv, wrap, trans, fwd_scatter, thin_backlight, fresnel = 15
#define UNIFIED_INPUT_FEATURES 15
#define UNIFIED_INPUT_NEURONS 15
#define UNIFIED_OUTPUT_NEURONS 3 // RGB output
#define UNIFIED_HIDDEN_NEURONS 64
#define UNIFIED_NUM_HIDDEN_LAYERS 4                        // 5 transitions total (input_proj + 3 residual blocks + output)
#define UNIFIED_NUM_TRANSITIONS (UNIFIED_NUM_HIDDEN_LAYERS + 1) // 5
#define UNIFIED_NUM_TRANSITIONS_ALIGN4 ((UNIFIED_NUM_TRANSITIONS + 3) / 4) // 2

// Training constants for unified MLP
#define UNIFIED_BATCH_SIZE (1 << 16)
#define UNIFIED_BATCH_COUNT 50
#define UNIFIED_THREADS_PER_GROUP 64
#define UNIFIED_THREADS_PER_GROUP_OPT 32
#define UNIFIED_LEARNING_RATE 0.001f
#define UNIFIED_LOSS_SCALE 128.0

// Maximum number of material textures supported in the bindless texture array
#define MAX_MATERIAL_TEXTURES 64

struct NeuralConstants
{
    // Scene setup
    float4x4 viewProject;
    float4x4 inverseViewProject;
    float4x4 inverseViewProjectNoTranslation; // Rotation-only inverse VP for skybox
    float4x4 view;
    float4 cameraPos;

    // Light setup
    float4 lightDir;
    float4 lightIntensity;

    // Material props (used as global fallback when no per-vertex material)
    float4 baseColor;
    float specular;
    float roughness;
    float metallic;
    uint enableNeuralShading;

    // Neural weight & bias offsets
    uint4 weightOffsets; // Offsets to weight matrices in bytes.
    uint4 biasOffsets; // Offsets to bias vectors in bytes.

    // Material system
    uint usePerVertexMaterial;
    uint materialCount;
    uint textureCount;

    // Unified neural shading (Disney + IBL baked)
    uint enableNeuralIBL;
    uint enableIBL;
    uint2 _pad0;
    uint4 uniWeightOffsets[UNIFIED_NUM_TRANSITIONS_ALIGN4];
    uint4 uniBiasOffsets[UNIFIED_NUM_TRANSITIONS_ALIGN4];
};

// GPU material parameters — must match C++ MaterialParams exactly
struct GpuMaterialParams
{
    float4 baseColor;
    float roughness;
    float metallic;
    float specular;
    float normalScale;

    float3 emissiveFactor;
    float occlusionStrength;

    int baseColorTexIdx;
    int normalTexIdx;
    int metallicRoughnessTexIdx;
    int occlusionTexIdx;

    int emissiveTexIdx;
    int alphaMode;
    float alphaCutoff;
    int thicknessTexIdx; // Replaced _pad0
    int curvatureTexIdx; // Hijacked from GLTF clearcoat_texture
};

// Training constant buffer for unified MLP
struct UnifiedTrainingConstants
{
    uint4 weightOffsets[UNIFIED_NUM_TRANSITIONS_ALIGN4];
    uint4 biasOffsets[UNIFIED_NUM_TRANSITIONS_ALIGN4];

    uint32_t maxParamSize;
    float learningRate;
    float currentStep;
    uint32_t batchSize;

    uint64_t seed;
    uint2 _pad;
};

#endif //__NETWORK_CONFIG_H__