/*
 * Copyright (c) 2015 - 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once

#include <donut/core/math/math.h>

#include <vector>
#include <string>

struct Vertex
{
    dm::float3 position;
    dm::float3 normal;
};

struct MaterialParams
{
    dm::float4 baseColor;
    float roughness; 
    float metallic;
    float specular;
    float padding; // for 16-byte alignment
};

struct TextureData
{
    std::string path;
    int width = 0;
    int height = 0;
    int channels = 0;
    std::vector<float> data_float; // Raw HDR floating point pixel data
};

// Load an HDR or EXR environment map
bool LoadHDRI(const std::string& path, TextureData& outTexture);

std::pair<std::vector<Vertex>, std::vector<uint32_t>> GenerateSphere(float radius, uint32_t segmentsU, uint32_t segmentsV);

// Loads a GLTF/GLB file, baking node transforms, and extracting vertices, indices, and materials.
bool LoadGLTF(
    const std::string& path, 
    std::vector<Vertex>& outVertices, 
    std::vector<uint32_t>& outIndices, 
    std::vector<MaterialParams>& outMaterials,
    std::vector<uint32_t>& outMaterialIndices
);
