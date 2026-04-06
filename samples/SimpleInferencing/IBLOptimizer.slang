/*
 * IBLOptimizer.slang
 *
 * Adam optimizer for the IBL Sampler MLP weights.
 * Same pattern as UnifiedOptimizer.slang.
 */

#include "NetworkConfig.h"
#include <donut/shaders/binding_helpers.hlsli>

import Optimizers;

DECLARE_CBUFFER(GenericTrainingConstants, gConst, 0, 0);
RWBuffer<half> gMLPParams             : REGISTER_UAV(0, 0);
RWBuffer<float> gMLPParams32          : REGISTER_UAV(1, 0);
RWBuffer<half> gMLPParamsGradients    : REGISTER_UAV(2, 0);
RWBuffer<float> gMoments1             : REGISTER_UAV(3, 0);
RWBuffer<float> gMoments2             : REGISTER_UAV(4, 0);

[numthreads(IBL_THREADS_PER_GROUP_OPT, 1, 1)]
void adam_cs(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    uint i = dispatchThreadID.x;
    if (i >= gConst.maxParamSize)
        return;

    float gradient = (float)gMLPParamsGradients[i];
    gMLPParamsGradients[i] = half(0.0);

    if (isfinite(gradient))
    {
        float weightBias = gConst.currentStep == 1 ? (float)gMLPParams[i] : gMLPParams32[i];

        optimizers::Adam optimizer = optimizers::Adam(gMoments1, gMoments2, gConst.learningRate, IBL_LOSS_SCALE);

        float adjustedWeightBias = optimizer.step(weightBias, i, gradient, gConst.currentStep);

        gMLPParams32[i] = adjustedWeightBias;
        gMLPParams[i] = (half) adjustedWeightBias;
    }
}
