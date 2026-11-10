#pragma once
#include <optix_types.h>

// Offset of `params` in OptiX's launch constant bank (c[3]) on SM89, driver 580.
// The host checks the instance node it finds this way before using it.
#define PARAMS_C3_OFFSET 0x140

struct Ray8 { float o[3], tmin, d[3], tmax; };
struct RefOut { unsigned hit; float t; unsigned prim; float u, v; unsigned pad[3]; };

struct Params {
    const Ray8 *rays;
    RefOut *out;
    OptixTraversableHandle handle;
    unsigned long long *inst;   // [0] = node address, [1..16] = its first 16 words
    unsigned n;
};
