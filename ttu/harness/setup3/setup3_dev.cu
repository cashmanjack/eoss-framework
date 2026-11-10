// OptiX programs for the Setup_3 reference launch: one ray per launch index,
// read from a buffer. Lane 0 also reports OptiX's own instance node (see README).
#include <optix.h>
#include "setup3_params.h"

extern "C" __constant__ Params params;

extern "C" __global__ void __raygen__rg()
{
    const unsigned i = optixGetLaunchIndex().x;
    if (i == 0) {
        // OptiX's launch constants sit below params in the same constant bank;
        // the word 0x58 into the bank points 0x100 past the instance node.
        const unsigned long long t =
            ((const volatile unsigned long long *)&params)[(int)(0x58 - PARAMS_C3_OFFSET) / 8];
        params.inst[0] = t - 0x100;
        for (int k = 0; k < 16; ++k)
            params.inst[1 + k] = ((const volatile unsigned *)(t - 0x100))[k];
    }
    if (i >= params.n) return;
    const Ray8 r = params.rays[i];
    unsigned hit = 0, tb = 0, prim = 0, u = 0, v = 0;
    optixTrace(params.handle, make_float3(r.o[0], r.o[1], r.o[2]), make_float3(r.d[0], r.d[1], r.d[2]),
               r.tmin, r.tmax, 0.f, OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE, 0, 1, 0,
               hit, tb, prim, u, v);
    RefOut o; o.hit = hit; o.t = __uint_as_float(tb); o.prim = prim;
    o.u = __uint_as_float(u); o.v = __uint_as_float(v);
    params.out[i] = o;
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
}
