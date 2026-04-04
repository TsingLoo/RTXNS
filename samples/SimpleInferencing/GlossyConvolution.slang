/*
 * GlossyConvolution.slang - convolutes an Environment Mapped cubemap to a specular prefiltered map 
 */

TextureCube<float4> u_EnvironmentMap     : register(t0);
SamplerState s_Linear                    : register(s0);
RWTexture2DArray<float4> u_SpecularMap   : register(u0);

cbuffer ConvolutionConstants : register(b0)
{
    float u_Roughness;
    float u_CubeSize;
    float pad0;
    float pad1;
};

const static float PI = 3.14159265359;

float RadicalInverse_VdC(uint bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10; // / 0x100000000
}

float2 Hammersley(uint i, uint N) {
    return float2(float(i)/float(N), RadicalInverse_VdC(i));
}

float distribFunc(float3 N, float3 wh, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, wh), 0.0);
    float denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

float3 ImportanceSampleGGX(float2 xi, float3 N, float roughness) {
    float a = roughness * roughness;

    float phi = 2.0 * PI * xi.x;
    float cosTheta = sqrt((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y));
    float sinTheta = sqrt(1.0 - cosTheta*cosTheta);

    // from spherical coordinates to cartesian coordinates - halfway vector
    float3 wh;
    wh.x = cos(phi) * sinTheta;
    wh.y = sin(phi) * sinTheta;
    wh.z = cosTheta;

    // from tangent-space H vector to world-space sample vector
    float3 up        = abs(N.z) < 0.999 ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0);
    float3 tangent   = normalize(cross(up, N));
    float3 bitangent = cross(N, tangent);

    float3 whW = tangent * wh.x + bitangent * wh.y + N * wh.z;
    return normalize(whW);
}

[numthreads(8, 8, 1)]
void main_cs(uint3 DTid : SV_DispatchThreadID)
{
    uint width, height, depths;
    u_SpecularMap.GetDimensions(width, height, depths);

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

    float3 N = dir;
    float3 V = N; // Assume viewing angle equals normal

    float3 Li = float3(0.0, 0.0, 0.0);
    float totalWeight = 0.0;
    const uint SAMPLE_COUNT = 1024u;

    for(uint i = 0u; i < SAMPLE_COUNT; ++i) {
        float2 xi = Hammersley(i, SAMPLE_COUNT);
        float3 wh = ImportanceSampleGGX(xi, N, u_Roughness);
        float3 L  = normalize(2.0 * dot(V, wh) * wh - V);

        float NdotL = max(dot(N, L), 0.0);
        if(NdotL > 0.0) {
            float D       = distribFunc(N, wh, u_Roughness);
            float nDotwh  = max(dot(N, wh), 0.0);
            float woDotwh = max(dot(wh, V), 0.0);
            float pdf     = D * nDotwh / (4.0 * woDotwh) + 0.0001;

            float resolution = u_CubeSize; 
            float saTexel    = 4.0 * PI / (6.0 * resolution * resolution);
            float saSample   = 1.0 / (float(SAMPLE_COUNT) * pdf + 0.0001);
            float mipLevel   = u_Roughness == 0.0 ? 0.0 : 0.5 * log2(saSample / saTexel);

            Li += u_EnvironmentMap.SampleLevel(s_Linear, L, mipLevel).rgb * NdotL;
            totalWeight += NdotL;
        }
    }

    if(totalWeight > 0.0) {
        Li /= totalWeight;
    }

    u_SpecularMap[DTid] = float4(Li, 1.0);
}
