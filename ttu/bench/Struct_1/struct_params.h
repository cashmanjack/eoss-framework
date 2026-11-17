#pragma once
#include <optix_types.h>
// Struct_1 (build flags, compaction, instance levels, box quantization). OptiX only, no TTU code.
struct Ray8 { float o[3], tmin, d[3], tmax; };
struct Out { unsigned hit, tbits, prim, calls, inst, pad0, pad1, pad2; };   // hit | instance id << 1; calls = intersection program calls (custom primitives)
struct Params {
    const Ray8 *rays; Out *out; OptixTraversableHandle handle;
    unsigned n, nrep;
    unsigned long long *clk;   // non null: probe launch, [8 x launch index] = ta, (tb_k, tc_k) x 3, td
};
