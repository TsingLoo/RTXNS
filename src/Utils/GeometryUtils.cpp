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

#include <fstream>
#include <sstream>
#include <algorithm>
#include <filesystem>
#include <cstring>

#include <stb_image.h>

extern "C" {
#include <mikktspace.h>
}

#include <cgltf.h>

using namespace dm;

// =============================================================================
// Sphere generation
// =============================================================================

std::pair<std::vector<Vertex>, std::vector<uint32_t>> GenerateSphere(float radius, uint32_t segmentsU, uint32_t segmentsV)
{
    std::vector<Vertex> vs;
    std::vector<uint32_t> indices;

    for (uint32_t v = 0; v <= segmentsV; ++v)
    {
        for (uint32_t u = 0; u <= segmentsU; ++u)
        {
            float2 uv = float2(u / float(segmentsU), v / float(segmentsV));
            float theta = uv.x * 2.f * PI_f;
            float phi = uv.y * PI_f;
            float3 dir = float3(std::cos(theta) * std::sin(phi), std::cos(phi), std::sin(theta) * std::sin(phi));

            Vertex vert = {};
            vert.position = dir * radius;
            vert.normal = dir;
            vert.uv = uv;
            vert.tangent = float4(0, 0, 0, 1);
            vert.materialIndex = 0;
            vs.push_back(vert);
        }
    }

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

// =============================================================================
// OBJ loader
// =============================================================================

bool LoadOBJ(
    const std::string& path, 
    std::vector<Vertex>& outVertices, 
    std::vector<uint32_t>& outIndices, 
    std::vector<MaterialParams>& outMaterials,
    std::vector<uint32_t>& outMaterialIndices
)
{
    std::ifstream file(path);
    if (!file.is_open()) return false;

    std::vector<dm::float3> temp_vertices;
    std::vector<dm::float3> temp_normals;

    outVertices.clear();
    outIndices.clear();

    MaterialParams p = {};
    p.baseColor = dm::float4(1.f, 1.f, 1.f, 1.f);
    p.metallic = 0.0f;
    p.roughness = 0.5f;
    p.specular = 0.5f;
    p.normalScale = 1.0f;
    p.occlusionStrength = 1.0f;
    p.emissiveFactor = dm::float3(0, 0, 0);
    p.baseColorTexIdx = -1;
    p.normalTexIdx = -1;
    p.metallicRoughnessTexIdx = -1;
    p.occlusionTexIdx = -1;
    p.emissiveTexIdx = -1;
    p.alphaMode = 0;
    p.alphaCutoff = 0.5f;
    p.thicknessTexIdx = -1;
    p.curvatureTexIdx = -1;
    outMaterials.push_back(p);

    std::string line;
    while (std::getline(file, line))
    {
        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        std::string type;
        iss >> type;
        
        if (type == "v")
        {
            dm::float3 v;
            iss >> v.x >> v.y >> v.z;
            temp_vertices.push_back(v);
        }
        else if (type == "vn")
        {
            dm::float3 vn;
            iss >> vn.x >> vn.y >> vn.z;
            temp_normals.push_back(vn);
        }
        else if (type == "f")
        {
            std::string vt1, vt2, vt3;
            if (!(iss >> vt1 >> vt2 >> vt3)) continue;

            auto parseVertex = [&](const std::string& v_str) {
                std::string token;
                std::istringstream tokenStream(v_str);
                
                int v_idx = 0, vt_idx = 0, vn_idx = 0;
                
                std::getline(tokenStream, token, '/');
                if (!token.empty()) v_idx = std::stoi(token);
                
                std::getline(tokenStream, token, '/');
                if (!token.empty()) vt_idx = std::stoi(token);
                
                std::getline(tokenStream, token, '/');
                if (!token.empty()) vn_idx = std::stoi(token);

                Vertex vert = {};
                if (v_idx > 0 && v_idx <= (int)temp_vertices.size())
                    vert.position = temp_vertices[v_idx - 1];
                else
                    vert.position = dm::float3(0,0,0);
                    
                if (vn_idx > 0 && vn_idx <= (int)temp_normals.size())
                    vert.normal = normalize(temp_normals[vn_idx - 1]);
                else
                    vert.normal = dm::float3(0,1,0);

                vert.uv = dm::float2(0, 0);
                vert.tangent = dm::float4(1, 0, 0, 1);

                outVertices.push_back(vert);
                outIndices.push_back((uint32_t)outVertices.size() - 1);
                outMaterialIndices.push_back(0);
            };

            parseVertex(vt1);
            parseVertex(vt2);
            parseVertex(vt3);
            
            std::string vt4;
            if (iss >> vt4)
            {
                outIndices.push_back(outIndices[outIndices.size() - 3]);
                outIndices.push_back(outIndices[outIndices.size() - 2]);
                parseVertex(vt4);
            }
        }
    }
    
    return true;
}

// =============================================================================
// MikkTSpace tangent generation for GLTF meshes
// =============================================================================

struct MikkTSpaceUserData
{
    std::vector<Vertex>* vertices;
    std::vector<uint32_t>* indices;
};

static int mikkGetNumFaces(const SMikkTSpaceContext* pContext)
{
    auto* data = (MikkTSpaceUserData*)pContext->m_pUserData;
    return (int)(data->indices->size() / 3);
}

static int mikkGetNumVerticesOfFace(const SMikkTSpaceContext* /*pContext*/, const int /*iFace*/)
{
    return 3;
}

static void mikkGetPosition(const SMikkTSpaceContext* pContext, float fvPosOut[], const int iFace, const int iVert)
{
    auto* data = (MikkTSpaceUserData*)pContext->m_pUserData;
    uint32_t idx = (*data->indices)[iFace * 3 + iVert];
    const Vertex& v = (*data->vertices)[idx];
    fvPosOut[0] = v.position.x;
    fvPosOut[1] = v.position.y;
    fvPosOut[2] = v.position.z;
}

static void mikkGetNormal(const SMikkTSpaceContext* pContext, float fvNormOut[], const int iFace, const int iVert)
{
    auto* data = (MikkTSpaceUserData*)pContext->m_pUserData;
    uint32_t idx = (*data->indices)[iFace * 3 + iVert];
    const Vertex& v = (*data->vertices)[idx];
    fvNormOut[0] = v.normal.x;
    fvNormOut[1] = v.normal.y;
    fvNormOut[2] = v.normal.z;
}

static void mikkGetTexCoord(const SMikkTSpaceContext* pContext, float fvTexcOut[], const int iFace, const int iVert)
{
    auto* data = (MikkTSpaceUserData*)pContext->m_pUserData;
    uint32_t idx = (*data->indices)[iFace * 3 + iVert];
    const Vertex& v = (*data->vertices)[idx];
    fvTexcOut[0] = v.uv.x;
    fvTexcOut[1] = v.uv.y;
}

static void mikkSetTSpaceBasic(const SMikkTSpaceContext* pContext, const float fvTangent[], const float fSign, const int iFace, const int iVert)
{
    auto* data = (MikkTSpaceUserData*)pContext->m_pUserData;
    uint32_t idx = (*data->indices)[iFace * 3 + iVert];
    Vertex& v = (*data->vertices)[idx];
    v.tangent = dm::float4(fvTangent[0], fvTangent[1], fvTangent[2], fSign);
}

static void GenerateMikkTSpaceTangents(std::vector<Vertex>& vertices, std::vector<uint32_t>& indices)
{
    MikkTSpaceUserData userData;
    userData.vertices = &vertices;
    userData.indices = &indices;

    SMikkTSpaceInterface iface = {};
    iface.m_getNumFaces = mikkGetNumFaces;
    iface.m_getNumVerticesOfFace = mikkGetNumVerticesOfFace;
    iface.m_getPosition = mikkGetPosition;
    iface.m_getNormal = mikkGetNormal;
    iface.m_getTexCoord = mikkGetTexCoord;
    iface.m_setTSpaceBasic = mikkSetTSpaceBasic;
    iface.m_setTSpace = nullptr;

    SMikkTSpaceContext context = {};
    context.m_pInterface = &iface;
    context.m_pUserData = &userData;

    genTangSpaceDefault(&context);
}

// =============================================================================
// GLTF texture loading helpers
// =============================================================================

static bool LoadGltfImage(
    const cgltf_image* image,
    const std::string& gltfDir,
    const cgltf_data* data,
    GltfTextureData& outTex)
{
    int w, h, ch;
    unsigned char* pixels = nullptr;

    if (image->buffer_view)
    {
        // Embedded texture — data is in the buffer_view
        const uint8_t* bufData = cgltf_buffer_view_data(image->buffer_view);
        if (!bufData) return false;
        pixels = stbi_load_from_memory(bufData, (int)image->buffer_view->size, &w, &h, &ch, 4);
    }
    else if (image->uri)
    {
        // External file reference
        std::string uri = image->uri;
        // Handle data URI
        if (uri.rfind("data:", 0) == 0)
        {
            // base64 encoded data URI
            auto commaPos = uri.find(',');
            if (commaPos == std::string::npos) return false;
            // We can't easily decode base64 here without a helper, so skip
            return false;
        }

        std::string texPath = gltfDir + "/" + uri;
        pixels = stbi_load(texPath.c_str(), &w, &h, &ch, 4);
    }

    if (!pixels) return false;

    outTex.width = w;
    outTex.height = h;
    outTex.channels = 4;
    outTex.pixels.resize(w * h * 4);
    memcpy(outTex.pixels.data(), pixels, w * h * 4);
    outTex.name = image->name ? image->name : "";

    stbi_image_free(pixels);
    return true;
}

static int ResolveTextureIndex(
    const cgltf_texture_view& texView,
    const cgltf_data* data,
    const std::vector<int>& imageToTextureIdx)
{
    if (!texView.texture) return -1;
    if (!texView.texture->image) return -1;
    size_t imgIdx = cgltf_image_index(data, texView.texture->image);
    if (imgIdx >= imageToTextureIdx.size()) return -1;
    return imageToTextureIdx[imgIdx];
}

// =============================================================================
// GLTF loader
// =============================================================================

bool LoadGLTF(
    const std::string& path, 
    std::vector<Vertex>& outVertices, 
    std::vector<uint32_t>& outIndices, 
    std::vector<MaterialParams>& outMaterials,
    std::vector<GltfTextureData>& outTextures
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
    
    // Get directory of GLTF file for resolving relative texture paths
    std::string gltfDir = std::filesystem::path(path).parent_path().string();

    // ---- Load all images ----
    outTextures.clear();
    std::vector<int> imageToTextureIdx(data->images_count, -1);

    for (size_t i = 0; i < data->images_count; ++i)
    {
        GltfTextureData tex;
        if (LoadGltfImage(&data->images[i], gltfDir, data, tex))
        {
            imageToTextureIdx[i] = (int)outTextures.size();
            outTextures.push_back(std::move(tex));
        }
    }

    // ---- Load materials ----
    outMaterials.clear();
    for (size_t i = 0; i < data->materials_count; ++i)
    {
        cgltf_material* mat = &data->materials[i];
        MaterialParams p = {};
        p.baseColor = float4(1.f, 1.f, 1.f, 1.f);
        p.metallic = 0.0f;
        p.roughness = 0.5f;
        p.specular = 0.5f;
        p.normalScale = 1.0f;
        p.occlusionStrength = 1.0f;
        p.emissiveFactor = float3(mat->emissive_factor[0], mat->emissive_factor[1], mat->emissive_factor[2]);

        // Alpha mode
        if (mat->alpha_mode == cgltf_alpha_mode_mask)
            p.alphaMode = 1;
        else if (mat->alpha_mode == cgltf_alpha_mode_blend)
            p.alphaMode = 2;
        else
            p.alphaMode = 0;
        p.alphaCutoff = mat->alpha_cutoff;

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

            p.baseColorTexIdx = ResolveTextureIndex(mat->pbr_metallic_roughness.base_color_texture, data, imageToTextureIdx);
            p.metallicRoughnessTexIdx = ResolveTextureIndex(mat->pbr_metallic_roughness.metallic_roughness_texture, data, imageToTextureIdx);
        }
        else
        {
            p.baseColorTexIdx = -1;
            p.metallicRoughnessTexIdx = -1;
        }

        p.normalTexIdx = ResolveTextureIndex(mat->normal_texture, data, imageToTextureIdx);
        if (mat->normal_texture.texture)
            p.normalScale = mat->normal_texture.scale;

        p.occlusionTexIdx = ResolveTextureIndex(mat->occlusion_texture, data, imageToTextureIdx);
        if (mat->occlusion_texture.texture)
            p.occlusionStrength = mat->occlusion_texture.scale;

        p.emissiveTexIdx = ResolveTextureIndex(mat->emissive_texture, data, imageToTextureIdx);

        if (mat->has_volume) {
            p.thicknessTexIdx = ResolveTextureIndex(mat->volume.thickness_texture, data, imageToTextureIdx);
            if (p.thicknessTexIdx >= 0)
                printf("GLTF material %zu: thickness texture loaded via KHR_materials_volume (idx=%d)\n", i, p.thicknessTexIdx);
        } else {
            p.thicknessTexIdx = -1;
        }

        if (mat->has_clearcoat) {
            p.curvatureTexIdx = ResolveTextureIndex(mat->clearcoat.clearcoat_texture, data, imageToTextureIdx);
            if (p.curvatureTexIdx >= 0)
                printf("GLTF material %zu: curvature texture loaded via KHR_materials_clearcoat (idx=%d)\n", i, p.curvatureTexIdx);
        } else {
            p.curvatureTexIdx = -1;
        }

        outMaterials.push_back(p);
    }
    
    if (outMaterials.empty())
    {
        MaterialParams p = {};
        p.baseColor = float4(1.f, 1.f, 1.f, 1.f);
        p.metallic = 0.0f;
        p.roughness = 0.5f;
        p.specular = 0.5f;
        p.normalScale = 1.0f;
        p.occlusionStrength = 1.0f;
        p.emissiveFactor = float3(0, 0, 0);
        p.baseColorTexIdx = -1;
        p.normalTexIdx = -1;
        p.metallicRoughnessTexIdx = -1;
        p.occlusionTexIdx = -1;
        p.emissiveTexIdx = -1;
        p.alphaMode = 0;
        p.alphaCutoff = 0.5f;
        p.thicknessTexIdx = -1;
        p.curvatureTexIdx = -1;
        outMaterials.push_back(p);
    }

    // ---- Fallback: load SSS companion textures from same directory ----
    // If the GLB didn't contain thickness/AO/curvature via extensions,
    // try loading them as standalone PNGs next to the GLB file.
    auto tryLoadFallbackTexture = [&](const std::string& filename) -> int {
        std::string texPath = gltfDir + "/" + filename;
        if (!std::filesystem::exists(texPath)) return -1;
        
        int w, h, ch;
        unsigned char* pixels = stbi_load(texPath.c_str(), &w, &h, &ch, 4);
        if (!pixels) return -1;
        
        GltfTextureData tex;
        tex.width = w;
        tex.height = h;
        tex.channels = 4;
        tex.pixels.resize(w * h * 4);
        memcpy(tex.pixels.data(), pixels, w * h * 4);
        tex.name = filename;
        stbi_image_free(pixels);
        
        int idx = (int)outTextures.size();
        outTextures.push_back(std::move(tex));
        printf("Loaded fallback SSS texture: %s (%dx%d)\n", texPath.c_str(), w, h);
        return idx;
    };

    for (auto& mat : outMaterials)
    {
        if (mat.thicknessTexIdx < 0)
            mat.thicknessTexIdx = tryLoadFallbackTexture("thickness_tex.png");
        if (mat.occlusionTexIdx < 0)
            mat.occlusionTexIdx = tryLoadFallbackTexture("ao_tex.png");
        if (mat.curvatureTexIdx < 0)
            mat.curvatureTexIdx = tryLoadFallbackTexture("curvature_tex.png");
    }
    
    // ---- Load geometry ----
    bool hasTangentsInFile = false;

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
            cgltf_accessor* uv_acc = nullptr;
            cgltf_accessor* tan_acc = nullptr;

            for (size_t a = 0; a < prim->attributes_count; ++a) {
                if (prim->attributes[a].type == cgltf_attribute_type_position) pos_acc = prim->attributes[a].data;
                if (prim->attributes[a].type == cgltf_attribute_type_normal)   norm_acc = prim->attributes[a].data;
                if (prim->attributes[a].type == cgltf_attribute_type_texcoord && prim->attributes[a].index == 0) uv_acc = prim->attributes[a].data;
                if (prim->attributes[a].type == cgltf_attribute_type_tangent)  tan_acc = prim->attributes[a].data;
            }
            
            if (!pos_acc) continue;

            if (tan_acc) hasTangentsInFile = true;
            
            size_t vCount = pos_acc->count;
            for (size_t v = 0; v < vCount; ++v)
            {
                Vertex vert = {};
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

                if (uv_acc) {
                    cgltf_float uv[2] = {0,0};
                    cgltf_accessor_read_float(uv_acc, v, uv, 2);
                    vert.uv = float2(uv[0], uv[1]);
                } else {
                    vert.uv = float2(0, 0);
                }

                if (tan_acc) {
                    cgltf_float tan[4] = {0,0,0,1};
                    cgltf_accessor_read_float(tan_acc, v, tan, 4);
                    float3 world_tan = normalize(normal_matrix * float3(tan[0], tan[1], tan[2]));
                    vert.tangent = float4(world_tan.x, world_tan.y, world_tan.z, tan[3]);
                } else {
                    vert.tangent = float4(1, 0, 0, 1); // Will be overwritten by MikkTSpace
                }
                
                vert.materialIndex = mat_idx;
                outVertices.push_back(vert);
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

    // Generate tangents via MikkTSpace if none were present in the file
    if (!hasTangentsInFile && !outVertices.empty() && !outIndices.empty())
    {
        GenerateMikkTSpaceTangents(outVertices, outIndices);
    }
    
    cgltf_free(data);
    return true;
}
