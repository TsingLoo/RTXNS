#pragma once
#include <donut/core/math/math.h>

struct UIData
{
    float lightIntensity = 1.f;
    float specular = 0.5f;
    float roughness = 0.4f;
    float metallic = 0.7f;
    donut::math::float3 lightDir = { -0.761f, -0.467f, -0.450f };
    donut::math::float4 baseColor = { 0.82f, 0.67f, 0.16f, 1.f };
    bool enableNeuralShading = true;
    bool enableNeuralSSS = false;  // Use unified MLP instead of cubemap IBL
    bool trainUnified = false;     // Start/stop unified MLP training
    bool enableNeuralIBL = false;  // Use IBL Sampler MLP for specular IBL
    bool trainIBL = false;         // Start/stop IBL Sampler training
};
