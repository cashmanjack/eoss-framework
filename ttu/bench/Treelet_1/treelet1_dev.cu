// Treelet_1 OptiX programs (no TTU code anywhere).
//   direct   one ray per launch index, handle = full GAS or the ray's treelet GAS
//   persist  persistent warps: each claims 32 consecutive rays from a queue, pinned = queue of its own TPC (smid / 2), steals only when it is empty
#include <optix.h>
#include "treelet1_params.h"
extern "C" __constant__ Params params;

static __device__ __forceinline__ void trace_store(unsigned idx)
{
    const Ray8 r = params.rays[idx];
    const OptixTraversableHandle h = params.use_tl ? params.tl_handle[params.tl[idx]] : params.handle;
    unsigned hit = 0, tb = 0, prim = 0, u = 0, v = 0, front = 0, gi = 0;
    optixTrace(h, make_float3(r.o[0], r.o[1], r.o[2]), make_float3(r.d[0], r.d[1], r.d[2]),
               r.tmin, r.tmax, 0.f, OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE, 0, 1, 0, hit, tb, prim, u, v, front, gi);
    RefOut o; o.hit = hit; o.t = __uint_as_float(tb); o.prim = prim; o.u = __uint_as_float(u); o.v = __uint_as_float(v); o.front = front; o.gas_index = gi; o.pad = 0;
    params.out[idx] = o;
}

extern "C" __global__ void __raygen__direct()
{
    const unsigned i = optixGetLaunchIndex().x;
    if (i >= params.n) return;
    trace_store(i);
}

extern "C" __global__ void __raygen__persist()
{
    unsigned lane, smid;
    asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane));
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    const unsigned li = optixGetLaunchIndex().x;
    if (lane == 0) params.warp_sm[li >> 5] = smid;
    if ((li & 31u) != lane) atomicAdd(&params.stats[2], 1u);
    const unsigned q0 = params.pinned ? min(smid >> 1, params.nq - 1) : 0u;
    for (;;) {
        unsigned base = 0, qq = 0, ok = 0;
        if (lane == 0) {
            unsigned b = atomicAdd(&params.head[q0], 32u);
            if (b < params.qend[q0]) { base = b; qq = q0; ok = 1; }
            else if (params.pinned) {
                for (unsigned k = 1; k < params.nq; ++k) {
                    const unsigned q = (q0 + k) % params.nq;
                    if (*(volatile unsigned *)&params.head[q] >= params.qend[q]) continue;
                    b = atomicAdd(&params.head[q], 32u);
                    if (b < params.qend[q]) { base = b; qq = q; ok = 1; break; }
                }
            }
            if (ok) { atomicAdd(&params.stats[qq == q0 ? 0 : 1], 1u); atomicAdd(&params.stats[3], 1u); atomicAdd(&params.smhist[smid & 255u], 1u); }
        }
        const unsigned mask = __activemask();
        ok = __shfl_sync(mask, ok, 0);
        if (!ok) break;
        base = __shfl_sync(mask, base, 0); qq = __shfl_sync(mask, qq, 0);
        const unsigned idx = base + lane;
        if (idx < params.qend[qq]) trace_store(idx);
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
