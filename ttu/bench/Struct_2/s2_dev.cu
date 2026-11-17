#include <optix.h>
#include "s2_params.h"
extern "C" __constant__ Params params;
extern "C" __global__ void __raygen__rg()
{
    const unsigned i = optixGetLaunchIndex().x, lane = i & 31u, warp = i >> 5, A = params.active;
    unsigned rank; bool on;
    if (params.spread) { const unsigned st = 32u / A; on = (lane % st) == 0; rank = lane / st; } else { on = lane < A; rank = lane; }
    if (!on) return;
    const unsigned r = warp * A + rank; if (r >= params.n) return;
    const Ray8 q = params.rays[r]; unsigned p0 = 0, p1 = 0, p2 = 0, p3 = 0;
    optixTrace(params.handle, make_float3(q.o[0], q.o[1], q.o[2]), make_float3(q.d[0], q.d[1], q.d[2]), q.tmin, q.tmax, 0.f, OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_ANYHIT, 0, 1, 0, p0, p1, p2, p3);
    params.out[r].hit = p0; params.out[r].tbits = p1; params.out[r].prim = p2; params.out[r].inst = p3;
}
extern "C" __global__ void __miss__ms() { optixSetPayload_0(0u); }
extern "C" __global__ void __closesthit__ch() { optixSetPayload_0(1u); optixSetPayload_1(__float_as_uint(optixGetRayTmax())); optixSetPayload_2(optixGetPrimitiveIndex()); optixSetPayload_3(optixGetInstanceId()); }
