// Omm_1 OptiX programs. Closest hit, alpha tested any hit (disc hole in barycentric space, pattern = primitive id mod 4), optional invocation counting.
#include <optix.h>
#include "omm_params.h"
extern "C" __constant__ Params params;
extern "C" __global__ void __raygen__rg()
{
    const unsigned i = optixGetLaunchIndex().x;
    if (i >= params.n) return;
    const Ray8 r = params.rays[i];
    unsigned p0 = 0, p1 = 0, p2 = 0;
    optixTrace(params.handle, make_float3(r.o[0], r.o[1], r.o[2]), make_float3(r.d[0], r.d[1], r.d[2]), r.tmin, r.tmax, 0.f,
               OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE, 0, 1, 0, p0, p1, p2);
    params.out[i].hit = p0; params.out[i].tbits = p1; params.out[i].prim = p2;
}
extern "C" __global__ void __miss__ms() { optixSetPayload_0(0u); }
extern "C" __global__ void __closesthit__ch()
{
    optixSetPayload_0(1u);
    optixSetPayload_1(__float_as_uint(optixGetRayTmax()));
    optixSetPayload_2(optixGetPrimitiveIndex());
}
extern "C" __global__ void __anyhit__ah()
{
    if (params.cnt) atomicAdd(&params.out[optixGetLaunchIndex().x].calls, 1u);
    const float2 b = optixGetTriangleBarycentrics();
    const float du = b.x - (1.f / 3.f), dv = b.y - (1.f / 3.f);
    if (params.ahw) { float x = b.x + 1.f, y = b.y + 0.5f; for (unsigned k = 0; k < params.ahw; ++k) { x = fmaf(x, 1.0000001f, y); y = fmaf(y, 0.9999999f, x); } if (x == 12345.f) optixIgnoreIntersection(); }
    if (du * du + dv * dv <= params.r2[optixGetPrimitiveIndex() & 3]) optixIgnoreIntersection();
}
