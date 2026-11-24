#pragma once
#include <optix_types.h>
struct Ray8 { float o[3], tmin, d[3], tmax; };
struct RefOut { unsigned hit; float t; unsigned prim; float u, v; unsigned front, gas_index, pad; };
struct Params {
    const Ray8 *rays; RefOut *out;
    OptixTraversableHandle handle;               // full GAS (root start)
    const OptixTraversableHandle *tl_handle;     // one GAS per treelet, entry [C] = the full GAS (rays without a treelet)
    const unsigned *tl;                          // treelet of every ray, in the order of rays
    const unsigned *n_dev; unsigned use_tl;      // n_dev: number of rays, read on the device (launches may be larger)
    unsigned *head; const unsigned *qend; unsigned nq, pinned;   // persistent kernel
    unsigned *stats;                             // 0 own queue chunks, 1 stolen chunks, 2 lane mismatches, 3 chunks total
    unsigned *smhist; unsigned *warp_sm;
    OptixTraversableHandle coarse; const float *boxes; const unsigned *excl;   // inspector: one box per treelet (6 floats), box id to skip per ray (0xffffffff: none, tie break on)
};
