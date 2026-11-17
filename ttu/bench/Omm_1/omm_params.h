#pragma once
#include <optix_types.h>
// Omm_1 (opacity micromaps). OptiX only, no TTU code.
struct Ray8 { float o[3], tmin, d[3], tmax; };
struct Out { unsigned hit, tbits, prim, calls; };   // calls = any hit invocations of this ray (counting launch only)
struct Params {
    const Ray8 *rays; Out *out; OptixTraversableHandle handle;
    unsigned n, cnt;      // cnt != 0: any hit programs count their invocations into out[ray].calls (atomics, counting launch only)
    unsigned ahw;         // dependent FMA iterations added to every any hit invocation (cost of a heavier alpha test), 0 = none
    float r2[4];          // squared hole radius around barycentrics (1/3, 1/3) per pattern (primitive id mod 4)
};
