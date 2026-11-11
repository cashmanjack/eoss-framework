#pragma once
#include <optix_types.h>
struct Ray8 { float o[3], tmin, d[3], tmax; };
struct RefOut { unsigned hit; float t; unsigned prim; float u, v; unsigned front, gas_index, pad; };
struct Params {
    const Ray8 *rays;
    RefOut *out;
    const unsigned *hints;
    OptixTraversableHandle handle;
    unsigned n, bits;
};
