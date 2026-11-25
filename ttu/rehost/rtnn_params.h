#pragma once
#include <optix_types.h>
// Offset of `params` in OptiX's launch constant bank (c[3]) on SM89, driver 580 (as ttu/harness/setup3/setup3_params.h).
#define PARAMS_C3_OFFSET 0x140
struct RParams {
    const float4 *q;              // query points, one per launch index / lane
    const float4 *pts;            // data points, primitive i = box [p_i - r, p_i + r]
    unsigned *nbr;                // per query 65 words: [0] neighbours found, [1..64] their primitive indices (first 64), in intersection call order
    unsigned *ncand;              // per query: intersection program calls (candidates)
    unsigned long long *inst;     // [0] OptiX's instance node address, [1..16] its first words, [17] c[3][0x34]
    OptixTraversableHandle handle;
    float r2;
    unsigned n;
    unsigned *candlog;            // per query 256 words: the primitive index of every intersection program call, in call order (first 256)
};
