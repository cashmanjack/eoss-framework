// Treelet_3 old pipeline (single GAS graph, as Treelet_2): natural / Morton baselines and the presorted oracle pinned reference. OptiX only.
#include <optix.h>
#include "treelet3_params.h"
extern "C" __constant__ Params params;

static __device__ __forceinline__ void trace_store(unsigned idx, unsigned tag)
{
    const Ray8 r = params.rays[idx];
    const OptixTraversableHandle h = params.use_tl ? params.tl_handle[params.tl[idx]] : params.handle;
    unsigned hit = 0, tb = 0, prim = 0, u = 0, v = 0, front = 0, gi = 0;
    optixTrace(h, make_float3(r.o[0], r.o[1], r.o[2]), make_float3(r.d[0], r.d[1], r.d[2]),
               r.tmin, r.tmax, 0.f, OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE, 0, 1, 0, hit, tb, prim, u, v, front, gi);
    RefOut o; o.hit = hit; o.t = __uint_as_float(tb); o.prim = prim; o.u = __uint_as_float(u); o.v = __uint_as_float(v); o.front = front; o.gas_index = gi; o.pad = tag;
    params.out[idx] = o;
}
extern "C" __global__ void __raygen__direct()
{
    const unsigned i = optixGetLaunchIndex().x;
    if (i >= *params.n_dev) return;
    trace_store(i, 0);
}
// persistent warps, all control flow warp uniform (Treelet_1 v5)
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
        unsigned ok = 0, base = 0, qq = q0;
        unsigned b = 0; if (lane == 0) b = atomicAdd(&params.head[q0], 32u);
        b = __shfl_sync(__activemask(), b, 0);
        if (b < params.qend[q0]) { ok = 1; base = b; }
        else if (params.pinned) {
            for (unsigned k = 1; k < params.nq; ++k) {
                const unsigned q = (q0 + k) % params.nq;
                unsigned h = *(volatile unsigned *)&params.head[q]; h = __shfl_sync(__activemask(), h, 0);
                if (h >= params.qend[q]) continue;
                b = 0; if (lane == 0) b = atomicAdd(&params.head[q], 32u);
                b = __shfl_sync(__activemask(), b, 0);
                if (b < params.qend[q]) { ok = 1; base = b; qq = q; break; }
            }
        }
        if (!ok) break;
        if (lane == 0) { atomicAdd(&params.stats[qq == q0 ? 0 : 1], 1u); atomicAdd(&params.stats[3], 1u); atomicAdd(&params.smhist[smid & 255u], 1u); }
        const unsigned idx = base + lane;
        if (idx < params.qend[qq]) trace_store(idx, 0x80000000u | (li >> 5));
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
