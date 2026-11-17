#pragma once
#include <optix_types.h>
// Struct_2 (build time, many instances, partial warps). OptiX only, no TTU code.
struct Ray8 { float o[3], tmin, d[3], tmax; };
struct Out { unsigned hit, tbits, prim, inst; };
struct Params { const Ray8 *rays; Out *out; OptixTraversableHandle handle; unsigned n; unsigned active; unsigned spread; };   // active lanes per warp (32 = all), spread 1 = every (32/active)th lane
