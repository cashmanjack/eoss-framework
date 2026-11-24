#pragma once
#include <optix_types.h>

// Offset of `params` in OptiX's launch constant bank (c[3]) on SM89, driver 580.
// The host checks the instance node it finds this way before using it.
#define PARAMS_C3_OFFSET 0x140

struct Ray8 { float o[3], tmin, d[3], tmax; };
struct RefOut { unsigned hit; float t; unsigned prim; float u, v; unsigned front, gas_index, pad; };

struct Params {
    const Ray8 *rays;
    RefOut *out;
    OptixTraversableHandle handle;
    unsigned long long *inst;   // [0] node address, [1..16] its first 16 words, [17] c[3][0x34]
    unsigned n;
    unsigned ias;               // 1: the handle is an IAS; the raygen does not read OptiX's instance node (the GAS only pipeline has one at c[3][0x58] - 0x100)
    unsigned oclk;              // HG_1 (hg1_dev.cu only): 1 = timed launch, no instance node read, 4 clock64 probes per launch index into clk
    unsigned long long *clk;    // HG_1: hg1_dev.cu: [4 x launch index] = ta, tb, tc, td; hg1b_dev.cu: [8 x launch index] = ta, tb0, tc0, tb1, tc1, tb2, tc2, td
    unsigned nrep;              // HG_1b (hg1b_dev.cu): traces per raygen, 1 to 3
    unsigned rot;               // HG_1b: 1 = trace k uses the ray of launch index (i + k) mod n
    unsigned *islog;            // rehosting (setup3_aabb_dev.cu): per ray 65 words: [0] number of intersection program calls, then the primitive indices of the first 64
};
