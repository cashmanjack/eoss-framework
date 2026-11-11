// OptiX programs for Morton / SER tests (no TTU code anywhere). One ray per launch index.
//   plain    optixTrace
//   ser      optixReorder(hint, bits) before optixTrace (hint per ray from a host file; no hit object yet)
//   serpost  optixTraverse, optixReorder(), optixInvoke (reorder after traversal: cannot change traversal order)
#include <optix.h>
#include "morton_params.h"

extern "C" __constant__ Params params;

extern "C" __global__ void __raygen__plain()
{
    const unsigned i = optixGetLaunchIndex().x;
    if (i >= params.n) return;
    const Ray8 r = params.rays[i];
    unsigned hit = 0, tb = 0, prim = 0, u = 0, v = 0, front = 0, gi = 0;
    optixTrace(params.handle, make_float3(r.o[0], r.o[1], r.o[2]), make_float3(r.d[0], r.d[1], r.d[2]),
               r.tmin, r.tmax, 0.f, OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE, 0, 1, 0, hit, tb, prim, u, v, front, gi);
    RefOut o; o.hit = hit; o.t = __uint_as_float(tb); o.prim = prim; o.u = __uint_as_float(u); o.v = __uint_as_float(v); o.front = front; o.gas_index = gi; o.pad = 0;
    params.out[i] = o;
}

extern "C" __global__ void __raygen__ser()
{
    const unsigned i = optixGetLaunchIndex().x;
    if (i >= params.n) return;
    const Ray8 r = params.rays[i];
    optixReorder(params.hints[i], params.bits);
    unsigned hit = 0, tb = 0, prim = 0, u = 0, v = 0, front = 0, gi = 0;
    optixTrace(params.handle, make_float3(r.o[0], r.o[1], r.o[2]), make_float3(r.d[0], r.d[1], r.d[2]),
               r.tmin, r.tmax, 0.f, OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE, 0, 1, 0, hit, tb, prim, u, v, front, gi);
    RefOut o; o.hit = hit; o.t = __uint_as_float(tb); o.prim = prim; o.u = __uint_as_float(u); o.v = __uint_as_float(v); o.front = front; o.gas_index = gi; o.pad = 0;
    params.out[i] = o;
}

extern "C" __global__ void __raygen__serpost()
{
    const unsigned i = optixGetLaunchIndex().x;
    if (i >= params.n) return;
    const Ray8 r = params.rays[i];
    unsigned hit = 0, tb = 0, prim = 0, u = 0, v = 0, front = 0, gi = 0;
    optixTraverse(params.handle, make_float3(r.o[0], r.o[1], r.o[2]), make_float3(r.d[0], r.d[1], r.d[2]),
                  r.tmin, r.tmax, 0.f, OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE, 0, 1, 0, hit, tb, prim, u, v, front, gi);
    optixReorder();
    optixInvoke(hit, tb, prim, u, v, front, gi);
    RefOut o; o.hit = hit; o.t = __uint_as_float(tb); o.prim = prim; o.u = __uint_as_float(u); o.v = __uint_as_float(v); o.front = front; o.gas_index = gi; o.pad = 0;
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
    optixSetPayload_5(optixIsTriangleFrontFaceHit() ? 1u : 0u);
    optixSetPayload_6(optixGetSbtGASIndex());
}
