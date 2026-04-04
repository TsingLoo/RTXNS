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

// Maximum number of material textures supported in the bindless texture array
#define MAX_MATERIAL_TEXTURES 64

struct NeuralConstants
{
    // Scene setup
    float4x4 viewProject;
    float4x4 inverseViewProject;
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
    uint _matPad0; // padding to 16-byte boundary
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
    float _pad0;
};

#endif //__NETWORK_CONFIG_H__