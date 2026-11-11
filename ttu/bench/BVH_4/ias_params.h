#pragma once
#include <optix_types.h>
struct Ray8 { float o[3], tmin, d[3], tmax; };
struct RefOut { unsigned hit; float t; unsigned prim; float u, v; unsigned front, instance_id, gas_index; };
struct Params { const Ray8 *rays; RefOut *out; OptixTraversableHandle handle; unsigned long long *c3; unsigned n; };
#define PARAMS_C3_OFFSET 0x140
