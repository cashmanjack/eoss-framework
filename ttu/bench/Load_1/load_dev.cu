// Load_1 device code (OptiX only, no TTU code of our own): point queries against a lattice of disjoint boxes, 1 candidate each.
#include <optix.h>
struct Rec { unsigned long long c0, c1, g0, g1; unsigned sm, calls; };
struct Params { const float *boxes; const float *pts; const unsigned *order; Rec *rec; unsigned *hits; OptixTraversableHandle handle; unsigned n; };
extern "C" __constant__ Params params;
__device__ __forceinline__ unsigned long long gt() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ unsigned smid() { unsigned s; asm volatile("mov.u32 %0, %%smid;" : "=r"(s)); return s; }
extern "C" __global__ void __raygen__rg()
{
    const unsigned i = optixGetLaunchIndex().x; if (i >= params.n) return;
    const unsigned long long g0 = gt(), c0 = clock64();
    const unsigned p = params.order[i]; const float3 o = make_float3(params.pts[3 * p], params.pts[3 * p + 1], params.pts[3 * p + 2]);
    unsigned h = 0, calls = 0;
    optixTrace(params.handle, o, make_float3(1.f, 0.f, 0.f), 0.f, 1e-16f, 0.f, OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE, 0, 1, 0, h, calls);
    const unsigned long long c1 = clock64(), g1 = gt();
    params.hits[i] = h;
    const unsigned lane = threadIdx.x & 31;
    // calls: sum over the lanes of the warp (each lane's own count)
    unsigned s = calls; for (int d = 16; d; d >>= 1) s += __shfl_down_sync(__activemask(), s, d);   // threads past n returned: only full warps are exact; the last warp is partial (reported as is)
    if (lane == 0) { Rec r; r.c0 = c0; r.c1 = c1; r.g0 = g0; r.g1 = g1; r.sm = smid(); r.calls = s; params.rec[i >> 5] = r; }
}
extern "C" __global__ void __miss__ms() { optixSetPayload_0(0u); }
extern "C" __global__ void __closesthit__ch() { optixSetPayload_0(1u); }
extern "C" __global__ void __intersection__is()
{
    const unsigned pi = optixGetPrimitiveIndex(); const float *b = params.boxes + 6 * pi; const float3 o = optixGetObjectRayOrigin();
    optixSetPayload_1(optixGetPayload_1() + 1u);
    if (o.x >= b[0] && o.x <= b[3] && o.y >= b[1] && o.y <= b[4] && o.z >= b[2] && o.z <= b[5]) optixReportIntersection(0.f, 0);
}
