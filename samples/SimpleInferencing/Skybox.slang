

Texture2D<float4> t_Skybox : register(t0);
SamplerState s_Skybox : register(s0);

#include "NetworkConfig.h"

cbuffer SkyboxConstants : register(b0)
{
    NeuralConstants constants;
};

struct VSOutput
{
    float4 position : SV_Position;
    float3 viewDir  : TEXCOORD0;
};

[shader("vertex")]
VSOutput main_vs(uint vertexID : SV_VertexID)
{
    VSOutput output;
    
    // Generate a full screen triangle
    float2 uv = float2((vertexID << 1) & 2, vertexID & 2);
    float4 posClip = float4(uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 1.0, 1.0); // Z=1.0 for reversed Z, drawn behind everything
    
    output.position = posClip;

    // Calculate view direction for this pixel by unprojecting clip space coordinates
    float4 posWorld = mul(posClip, constants.inverseViewProject);
    output.viewDir = normalize(posWorld.xyz / posWorld.w);

    return output;
}

[shader("fragment")]
float4 main_ps(VSOutput input) : SV_Target
{
    float3 d = normalize(input.viewDir);
    
    // Convert 3D direction to 2D equirectangular UVs
    float2 uv;
    uv.x = atan2(d.z, d.x) * 0.15915494309 + 0.5; // (1 / 2*PI)
    uv.y = asin(-d.y) * 0.31830988618 + 0.5;      // (1 / PI)
    
    float4 color = t_Skybox.SampleLevel(s_Skybox, uv, 0); // No mipmaps since we don't calculate analytic gradients here
    return float4(color.rgb, 1.0);
}
