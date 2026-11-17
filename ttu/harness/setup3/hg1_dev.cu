// HG_1: the Setup_3 reference raygen plus four clock64 probes (params.oclk = 1: timed launch, no instance node read).
// read from a buffer. Lane 0 also reports OptiX's own instance node and the
// launch constant the store words depend on (see README).
#include <optix.h>
#include "setup3_params.h"

extern "C" __constant__ Params params;

extern "C" __global__ void __raygen__rg()
{
    const unsigned i = optixGetLaunchIndex().x;
    const unsigned long long ta = clock64();
    if (i == 0 && !params.ias && !params.oclk) {
        // OptiX's launch constants sit below params in the same constant bank;
        // the word 0x58 into the bank points 0x100 past the instance node.
        const unsigned long long t =
            ((const volatile unsigned long long *)&params)[(int)(0x58 - PARAMS_C3_OFFSET) / 8];
        params.inst[0] = t - 0x100;
        for (int k = 0; k < 16; ++k)
            params.inst[1 + k] = ((const volatile unsigned *)(t - 0x100))[k];
        params.inst[17] = ((const volatile unsigned *)&params)[(int)(0x34 - PARAMS_C3_OFFSET) / 4];
    }
    if (i == 0 && params.ias && !params.oclk) params.inst[17] = ((const volatile unsigned *)&params)[(int)(0x34 - PARAMS_C3_OFFSET) / 4];
    if (i >= params.n) return;
    const Ray8 r = params.rays[i];
    unsigned hit = 0, tb = 0, prim = 0, u = 0, v = 0, front = 0, gi = 0;
    // the ray words come from a load: make the clock read wait for them (a real MOV reading the loaded registers), so TRACE does not contain the ray load
    unsigned dm; asm volatile("mov.b32 %0, %1;" : "=r"(dm) : "r"(__float_as_uint(r.o[0]) ^ __float_as_uint(r.o[2]) ^ __float_as_uint(r.d[0]) ^ __float_as_uint(r.tmax)));
    const unsigned long long pb = clock64();
    optixTrace(params.handle, make_float3(r.o[0], r.o[1], r.o[2]), make_float3(r.d[0], r.d[1], r.d[2]),
               r.tmin, r.tmax, 0.f, OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE, 0, 1, 0,
               hit, tb, prim, u, v, front, gi);
    const unsigned long long pc = clock64();
    RefOut o; o.hit = hit; o.t = __uint_as_float(tb); o.prim = prim;
    o.u = __uint_as_float(u); o.v = __uint_as_float(v); o.front = front; o.gas_index = gi; o.pad = 0;
    params.out[i] = o;
    if (params.oclk) {
        const unsigned long long pd = clock64();
        params.clk[4 * i] = ta; params.clk[4 * i + 1] = pb; params.clk[4 * i + 2] = pc; params.clk[4 * i + 3] = pd;
    }
}

extern "C" __global__ void __miss__ms() { optixSetPayload_0(0u); }

extern "C" __global__ void __closesthit__ch()
{
    const float2 b = optixGetTriangleBarycentrics();
    optixSetPayload_0(1u);
    optixSetPayload_1(__float_as_uint(optixGetRayTmax()));
    optixSetPayload_2(optixGetPrimitiveIndex());
    optixSetPayload_3(__float_as_uint(b.x));
    optixSetPayload_4(__float_as_uint(b.y));
    optixSetPayload_5(optixIsTriangleFrontFaceHit() ? 1u : 0u);
    optixSetPayload_6(optixGetSbtGASIndex());
}
