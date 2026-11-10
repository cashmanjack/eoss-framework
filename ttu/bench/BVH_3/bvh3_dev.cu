// BVH_3 OptiX programs: raygen times optixTrace with clock64; the intersection program never reports.
#include <optix.h>
struct P3 { unsigned long long *cyc; unsigned *hit; unsigned long long *is_calls; OptixTraversableHandle h; int count_is; };
extern "C" __constant__ P3 params;

extern "C" __global__ void __raygen__rg()
{
    const unsigned i = optixGetLaunchIndex().x;
    unsigned p0 = 0;
    const long long t0 = clock64();
    optixTrace(params.h, make_float3(0.f, 0.f, -2.f), make_float3(0.f, 0.f, 1.f), 0.f, 1e16f, 0.f, OptixVisibilityMask(255),
               OPTIX_RAY_FLAG_NONE, 0, 1, 0, p0);
    const long long t1 = clock64();
    params.cyc[i] = (unsigned long long)(t1 - t0); params.hit[i] = p0;
}
extern "C" __global__ void __miss__ms() { optixSetPayload_0(0); }
extern "C" __global__ void __closesthit__ch() { optixSetPayload_0(1); }
extern "C" __global__ void __intersection__is()   // always false: never calls optixReportIntersection
{
    if (params.count_is) atomicAdd(params.is_calls, 1ull);
}
