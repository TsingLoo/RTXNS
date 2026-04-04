/*
 * IrradianceConvolution.slang - convolutes an Environment Mapped cubemap to a diffuse irradiance map 
 */

TextureCube<float4> u_EnvironmentMap : register(t0);
SamplerState s_Linear                 : register(s0);
RWTexture2DArray<float4> u_IrradianceMap : register(u0);

const static float PI = 3.14159265359;

[numthreads(8, 8, 1)]
void main_cs(uint3 DTid : SV_DispatchThreadID)
{
    uint width, height, depths;
    u_IrradianceMap.GetDimensions(width, height, depths);

    if (DTid.x >= width || DTid.y >= height) return;

    // Convert pixel coordinate to [-1, 1] for the face
    float2 uv = float2(DTid.x + 0.5f, DTid.y + 0.5f) / float2(width, height);
    float2 xy = uv * 2.0f - 1.0f;
    xy.y = -xy.y; 

    float3 dir;
    switch (DTid.z)
    {
        case 0: dir = float3( 1.0f,  xy.y, -xy.x); break; // +X
        case 1: dir = float3(-1.0f,  xy.y,  xy.x); break; // -X
        case 2: dir = float3( xy.x,  1.0f, -xy.y); break; // +Y
        case 3: dir = float3( xy.x, -1.0f,  xy.y); break; // -Y
        case 4: dir = float3( xy.x,  xy.y,  1.0f); break; // +Z
        case 5: dir = float3(-xy.x,  xy.y, -1.0f); break; // -Z
        default: dir = float3(1,0,0); break;
    }
    dir = normalize(dir);

    // Mathematics directly from the user's Monte-Carlo reference integration
    float3 normal = dir;
    float3 N = normal;
    float3 irradiance = float3(0.0, 0.0, 0.0);

    float3 up    = float3(0.0, 1.0, 0.0);
    // Determine right and up vectors in world space
    // Standard exception handling if normal is completely vertical
    if (abs(normal.y) > 0.999) {
        up = float3(0.0, 0.0, 1.0);
    }
    float3 right = normalize(cross(up, normal));
    up           = normalize(cross(normal, right));

    float sampleDelta = 0.025;
    float nrSamples = 0.0;
    
    for(float phi = 0.0; phi < 2.0 * PI; phi += sampleDelta)
    {
        for(float theta = 0.0; theta < 0.5 * PI; theta += sampleDelta)
        {
            // spherical to cartesian (in tangent space)
            float3 tangentSample = float3(sin(theta) * cos(phi),  sin(theta) * sin(phi), cos(theta));
            
            // tangent space to world
            float3 sampleVec = tangentSample.x * right + tangentSample.y * up + tangentSample.z * N;

            irradiance += u_EnvironmentMap.SampleLevel(s_Linear, sampleVec, 0).rgb * cos(theta) * sin(theta);
            nrSamples++;
        }
    }
    irradiance = PI * irradiance * (1.0 / nrSamples);

    u_IrradianceMap[DTid] = float4(irradiance, 1.0);
}
