#pragma once
#include <optix_types.h>
struct Params {
    OptixTraversableHandle handle;
    const float3 *pts;       // all points, packed (also the queries: every point is a query)
    float *ld; int *li;      // per query K slot lists (squared distance, id), nearest first
    unsigned *nout, *cand;   // per query: list size, candidate callbacks (cand only when count != 0)
    float r2, L, rr;         // radius squared, ray length (sph arm), ray length of the cus arm
    float3 dir;
    unsigned qoff, qn, count;
};
