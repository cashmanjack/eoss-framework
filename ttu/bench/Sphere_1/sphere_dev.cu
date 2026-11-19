// Sphere_1 OptiX programs (model/predictions_spheres.md). -DCOUNT: the custom intersection program counts its own calls per ray (payload 2).
#include <optix.h>
#include "sphere_params.h"
extern "C" __constant__ Params params;

extern "C" __global__ void __raygen__rg()
{
    const unsigned i = optixGetLaunchIndex().x;
    unsigned long long c0 = 0, g0 = 0;
    if (params.mode == 0 && params.cal) { c0 = clock64(); asm volatile("mov.u64 %0, %globaltimer;" : "=l"(g0)); }
    if (i >= params.n) return;
    float o[3], d[3]; make_ray(params.mode == 1 ? params.ray_id : i, o, d);
    if (params.scene == 1) { const unsigned h0 = hash32((params.mode == 1 ? params.ray_id : i) * 4u + 1u), h1 = hash32((params.mode == 1 ? params.ray_id : i) * 4u + 2u); o[0] = params.jit * u01(h0); o[1] = params.jit * u01(h1); o[2] = -2.f; d[0] = 0.f; d[1] = 0.f; d[2] = 1.f; }
    unsigned p0 = 0, p1 = 0, p2 = 0, p3 = 0;
    const long long t0 = clock64();
    optixTrace(params.handle, make_float3(o[0], o[1], o[2]), make_float3(d[0], d[1], d[2]), 0.f, 1e16f, 0.f, OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE, 0, 1, 0, p0, p1, p2, p3);
    const long long t1 = clock64();
    if (params.mode == 1) { params.cyc[i] = (unsigned long long)(t1 - t0); return; }
    if (params.out) params.out[i] = make_uint4(p0, p1, p3, p2);
    unsigned acc = p0 ^ p1 ^ (p3 << 1);
    if (acc == 0x7fc00001u) params.sink[i >> 5] = acc;
    if (params.cal && (i & 31u) == 0) {
        unsigned long long c1 = clock64(), g1; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(g1));
        unsigned long long *w = params.cal + 4ull * (i >> 5); w[0] = c0; w[1] = c1; w[2] = g0; w[3] = g1;
    }
}
extern "C" __global__ void __miss__ms() { optixSetPayload_0(0u); }
extern "C" __global__ void __closesthit__ch()
{
    optixSetPayload_0(1u);
    optixSetPayload_1(__float_as_uint(optixGetRayTmax()));
    optixSetPayload_3(optixGetPrimitiveIndex() / params.tps);
}
// custom primitive arm: the ray sphere test of the spheres (center from global memory, one radius)
extern "C" __global__ void __intersection__sph()
{
#ifdef COUNT
    optixSetPayload_2(optixGetPayload_2() + 1u);
#endif
    const float3 c = params.centers[optixGetPrimitiveIndex()], o = optixGetObjectRayOrigin(), d = optixGetObjectRayDirection();
    const float3 oc = make_float3(o.x - c.x, o.y - c.y, o.z - c.z);
    const float b = oc.x * d.x + oc.y * d.y + oc.z * d.z, cc = oc.x * oc.x + oc.y * oc.y + oc.z * oc.z - params.radius * params.radius, disc = b * b - cc;
    if (disc < 0.f) return;
    const float s = sqrtf(disc); float t = -b - s;
#ifdef ENTRY_ONLY   // pc14 (F428): report only the entry root, as the 5090's built in sphere does (it reports no exit hit for a ray that starts inside)
    if (t < optixGetRayTmin()) return;
#else
    if (t < optixGetRayTmin()) t = -b + s;
#endif
    if (t >= optixGetRayTmin() && t <= optixGetRayTmax()) optixReportIntersection(t, 0);
}
