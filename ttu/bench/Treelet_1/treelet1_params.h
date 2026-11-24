#pragma once
#include <optix_types.h>
struct Ray8 { float o[3], tmin, d[3], tmax; };
struct RefOut { unsigned hit; float t; unsigned prim; float u, v; unsigned front, gas_index, pad; };
struct Params {
    const Ray8 *rays; RefOut *out;
    OptixTraversableHandle handle;               // full GAS (root start)
    const OptixTraversableHandle *tl_handle;     // one GAS per treelet (treelet root start)
    const unsigned *tl;                          // treelet of every ray, in the order of rays
    unsigned n, use_tl, chunk;                   // chunk: rays claimed per lane per atomic (persistent arms)
    unsigned *head; const unsigned *qend; unsigned nq, pinned;   // persistent kernel: queue heads and ends (ray index), TPC queues if pinned
    unsigned *stats;                             // 0 own queue chunks, 1 stolen chunks, 2 lane mismatches, 3 chunks total
    unsigned *smhist;                            // 256: chunks claimed per SM id
    unsigned *warp_sm;                           // one entry per persistent warp (launch index / 32): its SM id
};
