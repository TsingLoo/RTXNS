

TextureCube<float4> t_Skybox : register(t0);
SamplerState s_Skybox : register(s0);

#include "NetworkConfig.h"

cbuffer SkyboxConstants : register(b0)
{
    NeuralConstants constants;
};

struct VSOutput
{
    float4 position : SV_Position;
    float2 clipXY   : TEXCOORD0;
};

[shader("vertex")]
VSOutput main_vs(uint vertexID : SV_VertexID)
{
    VSOutput output;
    
    // Generate a full screen triangle
    float2 uv = float2((vertexID << 1) & 2, vertexID & 2);
    float4 posClip = float4(uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 1.0, 1.0);
    
    output.position = posClip;
    output.clipXY = posClip.xy;

    return output;
}

[shader("fragment")]
float4 main_ps(VSOutput input) : SV_Target
{
    // Unproject clip-space to world-space direction using rotation-only inverse VP.
    // Z=0 maps to the near plane, which gives view-space direction (cx/xScale, cy/yScale, 1).
    // The perspective divide and normalize produce the correct world direction.
    float4 posClip = float4(input.clipXY, 0.0, 1.0);
    float4 worldDir = mul(posClip, constants.inverseViewProjectNoTranslation);
    float3 d = normalize(worldDir.xyz / worldDir.w);
    
    float4 color = t_Skybox.SampleLevel(s_Skybox, d, 0); 
    return float4(color.rgb, 1.0);
}
