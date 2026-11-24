// OptiX programs for the rehosting reference launch (setup3 --aabb): the same raygen, miss and closest hit as setup3_dev.cu, plus an
// intersection program that never reports and logs every call (primitive index) into params.islog, 65 words per ray.
#include <optix.h>
#include "setup3_params.h"

extern "C" __constant__ Params params;

extern "C" __global__ void __raygen__rg()
{
    const unsigned i = optixGetLaunchIndex().x;
    if (i == 0 && !params.ias) {
        const unsigned long long t =
            ((const volatile unsigned long long *)&params)[(int)(0x58 - PARAMS_C3_OFFSET) / 8];
        params.inst[0] = t - 0x100;
        for (int k = 0; k < 16; ++k)
            params.inst[1 + k] = ((const volatile unsigned *)(t - 0x100))[k];
        params.inst[17] = ((const volatile unsigned *)&params)[(int)(0x34 - PARAMS_C3_OFFSET) / 4];
    }
    if (i >= params.n) return;
    const Ray8 r = params.rays[i];
    unsigned hit = 0, tb = 0, prim = 0, u = 0, v = 0, front = 0, gi = 0;
    optixTrace(params.handle, make_float3(r.o[0], r.o[1], r.o[2]), make_float3(r.d[0], r.d[1], r.d[2]),
               r.tmin, r.tmax, 0.f, OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE, 0, 1, 0,
               hit, tb, prim, u, v, front, gi);
    RefOut o; o.hit = hit; o.t = __uint_as_float(tb); o.prim = prim;
    o.u = __uint_as_float(u); o.v = __uint_as_float(v); o.front = front; o.gas_index = gi; o.pad = 0;
    params.out[i] = o;
}

extern "C" __global__ void __miss__ms() { optixSetPayload_0(0u); }

extern "C" __global__ void __closesthit__ch() { optixSetPayload_0(1u); }

extern "C" __global__ void __intersection__is()
{
    unsigned *log = params.islog + (size_t)optixGetLaunchIndex().x * 65;   // one thread per ray: no atomic needed
    const unsigned k = log[0];
    log[0] = k + 1;
    if (k < 64) log[1 + k] = optixGetPrimitiveIndex();
}
