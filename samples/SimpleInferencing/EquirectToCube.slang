/*
 * EquirectToCube.slang - converts a 2D equirectangular panorama into a Cubemap.
 */

Texture2D<float4> t_equirect : register(t0);
SamplerState s_linear : register(s0);
RWTexture2DArray<float4> u_cubeMap : register(u0);

const static float DISNEY_PI = 3.14159265358979323846;

[numthreads(8, 8, 1)]
void main_cs(uint3 DTid : SV_DispatchThreadID)
{
    uint width, height, depths;
    u_cubeMap.GetDimensions(width, height, depths);

    if (DTid.x >= width || DTid.y >= height) return;

    // Convert pixel coordinate to [-1, 1] for the face
    float2 uv = float2(DTid.x + 0.5f, DTid.y + 0.5f) / float2(width, height);
    float2 xy = uv * 2.0f - 1.0f;
    xy.y = -xy.y; 

    float3 dir;
    switch (DTid.z)
    {
        case 0: dir = normalize(float3(1.0f, xy.y, -xy.x)); break; // +X
        case 1: dir = normalize(float3(-1.0f, xy.y, xy.x)); break; // -X
        case 2: dir = normalize(float3(xy.x, 1.0f, -xy.y)); break; // +Y
        case 3: dir = normalize(float3(xy.x, -1.0f, xy.y)); break; // -Y
        case 4: dir = normalize(float3(xy.x, xy.y, 1.0f)); break;  // +Z
        case 5: dir = normalize(float3(-xy.x, xy.y, -1.0f)); break; // -Z
        default: dir = float3(1,0,0); break;
    }

    dir = normalize(dir);

    // Convert 3D direction to lat/long equirectangular 2D coordinates
    float2 skyUV;
    skyUV.x = atan2(dir.z, dir.x) * 0.15915494309f + 0.5f;
    skyUV.y = asin(clamp(-dir.y, -1.0f, 1.0f)) * 0.31830988618f + 0.5f;

    // Sample the panorama
    float4 color = t_equirect.SampleLevel(s_linear, skyUV, 0);

    u_cubeMap[DTid] = color;
}
