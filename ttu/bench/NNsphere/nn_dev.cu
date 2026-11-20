// NNsphere B device programs. Compile with -DK=5 or -DK=50. Predictions: model/predictions_5090_batch10.md part B.
#include <optix.h>
#include "nn_params.h"
extern "C" __constant__ Params params;
// insertion of point id into query q's list (shared by both arms); n is the list size carried in payload 2
static __device__ __forceinline__ void consider(unsigned q, unsigned id, unsigned &n)
{
    const float3 a = params.pts[q], b = params.pts[id];
    const float dx = b.x - a.x, dy = b.y - a.y, dz = b.z - a.z; const float d2 = dx * dx + dy * dy + dz * dz;
    if (d2 > params.r2) return;
#ifdef COUNTONLY   // no list: only the distance test and a count of points inside (isolates traversal and dispatch from the insertion code)
    ++n; return;
#endif
    float *ld = params.ld + (size_t)q * K; int *li = params.li + (size_t)q * K;
    int j;
    if (n == K) { if (d2 >= ld[K - 1]) return; j = K - 1; } else { j = (int)n; ++n; }
    while (j > 0 && ld[j - 1] > d2) { ld[j] = ld[j - 1]; li[j] = li[j - 1]; --j; }
    ld[j] = d2; li[j] = (int)id;
}
extern "C" __global__ void __raygen__sph()   // origin q - L d, tmax L: the entry root of every sphere that contains q lies in (0, L]
{
    const unsigned i = optixGetLaunchIndex().x; if (i >= params.qn) return; const unsigned q = params.qoff + i; const float3 p = params.pts[q];
    unsigned p0 = q, p1 = 0, p2 = 0;
    optixTrace(params.handle, make_float3(p.x - params.L * params.dir.x, p.y - params.L * params.dir.y, p.z - params.L * params.dir.z), params.dir, 0.f, params.L, 0.f, OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT, 0, 1, 0, p0, p1, p2);
    params.nout[q] = p2; if (params.count) params.cand[q] = p1;
}
extern "C" __global__ void __raygen__cus()   // RTNN style: a tiny ray from q, every box that contains q calls the intersection program
{
    const unsigned i = optixGetLaunchIndex().x; if (i >= params.qn) return; const unsigned q = params.qoff + i; const float3 p = params.pts[q];
    unsigned p0 = q, p1 = 0, p2 = 0;
    optixTrace(params.handle, p, params.dir, 0.f, params.rr, 0.f, OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT, 0, 1, 0, p0, p1, p2);
    params.nout[q] = p2; if (params.count) params.cand[q] = p1;
}
extern "C" __global__ void __miss__ms() {}
extern "C" __global__ void __anyhit__sph()
{
    unsigned n = optixGetPayload_2(); consider(optixGetPayload_0(), optixGetPrimitiveIndex(), n);
    optixSetPayload_2(n); optixSetPayload_1(optixGetPayload_1() + 1u);
    optixIgnoreIntersection();
}
extern "C" __global__ void __intersection__cus()
{
    unsigned n = optixGetPayload_2(); consider(optixGetPayload_0(), optixGetPrimitiveIndex(), n);
    optixSetPayload_2(n); optixSetPayload_1(optixGetPayload_1() + 1u);
}
