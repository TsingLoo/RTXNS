/*
 * Copyright (c) 2015 - 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "GeometryUtils.h"

using namespace dm;

std::pair<std::vector<Vertex>, std::vector<uint32_t>> GenerateSphere(float radius, uint32_t segmentsU, uint32_t segmentsV)
{
    std::vector<Vertex> vs;
    std::vector<uint32_t> indices;

    // Create vertices.
    for (uint32_t v = 0; v <= segmentsV; ++v)
    {
        for (uint32_t u = 0; u <= segmentsU; ++u)
        {
            float2 uv = float2(u / float(segmentsU), v / float(segmentsV));
            float theta = uv.x * 2.f * PI_f;
            float phi = uv.y * PI_f;
            float3 dir = float3(std::cos(theta) * std::sin(phi), std::cos(phi), std::sin(theta) * std::sin(phi));
            vs.push_back({ dir * radius, dir });
        }
    }

    // Create indices.
    for (uint32_t v = 0; v < segmentsV; ++v)
    {
        for (uint32_t u = 0; u < segmentsU; ++u)
        {
            uint32_t i0 = v * (segmentsU + 1) + u;
            uint32_t i1 = v * (segmentsU + 1) + (u + 1) % (segmentsU + 1);
            uint32_t i2 = (v + 1) * (segmentsU + 1) + u;
            uint32_t i3 = (v + 1) * (segmentsU + 1) + (u + 1) % (segmentsU + 1);

            indices.emplace_back(i0);
            indices.emplace_back(i1);
            indices.emplace_back(i2);

            indices.emplace_back(i2);
            indices.emplace_back(i1);
            indices.emplace_back(i3);
        }
    }

    return { vs, indices };
}

#include <cgltf.h>

bool LoadGLTF(
    const std::string& path, 
    std::vector<Vertex>& outVertices, 
    std::vector<uint32_t>& outIndices, 
    std::vector<MaterialParams>& outMaterials,
    std::vector<uint32_t>& outMaterialIndices
)
{
    cgltf_options options = {};
    cgltf_data* data = nullptr;
    cgltf_result result = cgltf_parse_file(&options, path.c_str(), &data);
    if (result != cgltf_result_success) return false;
    
    result = cgltf_load_buffers(&options, data, path.c_str());
    if (result != cgltf_result_success)
    {
        cgltf_free(data);
        return false;
    }
    
    outMaterials.clear();
    for (size_t i = 0; i < data->materials_count; ++i)
    {
        cgltf_material* mat = &data->materials[i];
        MaterialParams p;
        p.baseColor = float4(1.f, 1.f, 1.f, 1.f);
        p.metallic = 0.0f;
        p.roughness = 0.5f;
        p.specular = 0.5f;
        
        if (mat->has_pbr_metallic_roughness)
        {
            p.baseColor = float4(
                mat->pbr_metallic_roughness.base_color_factor[0],
                mat->pbr_metallic_roughness.base_color_factor[1],
                mat->pbr_metallic_roughness.base_color_factor[2],
                mat->pbr_metallic_roughness.base_color_factor[3]
            );
            p.metallic = mat->pbr_metallic_roughness.metallic_factor;
            p.roughness = mat->pbr_metallic_roughness.roughness_factor;
        }
        outMaterials.push_back(p);
    }
    
    if (outMaterials.empty())
    {
        MaterialParams p;
        p.baseColor = float4(1.f, 1.f, 1.f, 1.f);
        p.metallic = 0.0f;
        p.roughness = 0.5f;
        p.specular = 0.5f;
        outMaterials.push_back(p);
    }
    
    for (size_t n = 0; n < data->nodes_count; ++n)
    {
        cgltf_node* node = &data->nodes[n];
        if (!node->mesh) continue;
        
        cgltf_float mat[16];
        cgltf_node_transform_world(node, mat);
        
        float4x4 transform;
        transform.row0 = float4(mat[0], mat[4], mat[8], mat[12]);
        transform.row1 = float4(mat[1], mat[5], mat[9], mat[13]);
        transform.row2 = float4(mat[2], mat[6], mat[10], mat[14]);
        transform.row3 = float4(mat[3], mat[7], mat[11], mat[15]);
        
        float3x3 normal_matrix = float3x3(transform.row0.xyz(), transform.row1.xyz(), transform.row2.xyz());
        
        cgltf_mesh* mesh = node->mesh;
        for (size_t p = 0; p < mesh->primitives_count; ++p)
        {
            cgltf_primitive* prim = &mesh->primitives[p];
            uint32_t mat_idx = 0;
            if (prim->material) {
                mat_idx = (uint32_t)(prim->material - data->materials);
            }
            
            uint32_t vertexOffset = (uint32_t)outVertices.size();
            
            cgltf_accessor* pos_acc = nullptr;
            cgltf_accessor* norm_acc = nullptr;
            for (size_t a = 0; a < prim->attributes_count; ++a) {
                if (prim->attributes[a].type == cgltf_attribute_type_position) pos_acc = prim->attributes[a].data;
                if (prim->attributes[a].type == cgltf_attribute_type_normal) norm_acc = prim->attributes[a].data;
            }
            
            if (!pos_acc) continue;
            
            size_t vCount = pos_acc->count;
            for (size_t v = 0; v < vCount; ++v)
            {
                Vertex vert;
                cgltf_float pos[3] = {0,0,0};
                cgltf_accessor_read_float(pos_acc, v, pos, 3);
                
                float4 world_pos = transform * float4(pos[0], pos[1], pos[2], 1.f);
                vert.position = world_pos.xyz();
                
                if (norm_acc) {
                    cgltf_float norm[3] = {0,0,0};
                    cgltf_accessor_read_float(norm_acc, v, norm, 3);
                    float3 world_norm = normalize(normal_matrix * float3(norm[0], norm[1], norm[2]));
                    vert.normal = world_norm;
                } else {
                    vert.normal = float3(0,1,0); 
                }
                
                outVertices.push_back(vert);
                outMaterialIndices.push_back(mat_idx);
            }
            
            if (prim->indices) {
                size_t iCount = prim->indices->count;
                for (size_t i = 0; i < iCount; ++i) {
                    outIndices.push_back(vertexOffset + (uint32_t)cgltf_accessor_read_index(prim->indices, i));
                }
            } else {
                for (size_t i = 0; i < vCount; ++i) {
                    outIndices.push_back(vertexOffset + (uint32_t)i);
                }
            }
        }
    }
    
    cgltf_free(data);
    return true;
}
