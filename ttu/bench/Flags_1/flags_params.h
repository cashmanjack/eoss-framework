#pragma once
#include <optix_types.h>
struct Ray8 { float o[3], tmin, d[3], tmax; };
struct Out { unsigned hit, tbits, prim, ah, front, inst, pad0, pad1; };
struct Params {
    const Ray8 *rays; Out *out; OptixTraversableHandle handle;
    unsigned n, flags, mask, nrep;
    unsigned long long *clk;   // non null: probe launch, [8 x launch index] = ta, (tb_k, tc_k) x 3, td
};
