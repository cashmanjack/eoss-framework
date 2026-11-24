// Treelet_2 OptiX programs (no TTU code anywhere). direct, persist (as Treelet_1 v5), coarse (inspector over treelet boxes).
#include <optix.h>
#include "treelet2_params.h"
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

// persistent warps: all control flow is warp uniform (lane 0's value through a shuffle; lane 0's atomic is one predicated instruction). See Treelet_1 log.
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

// inspector: nearest treelet box along the ray (prim = box id), or with excl[i] = a box id: nearest OTHER box within [tmin, tmax]
extern "C" __global__ void __raygen__coarse()
{
    const unsigned i = optixGetLaunchIndex().x;
    if (i >= *params.n_dev) return;
    const Ray8 r = params.rays[i];
    unsigned hit = 0, tb = 0, prim = 0, u = 0, v = 0, front = 0, gi = 0;
    optixTrace(params.coarse, make_float3(r.o[0], r.o[1], r.o[2]), make_float3(r.d[0], r.d[1], r.d[2]),
               r.tmin, r.tmax, 0.f, OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE, 1, 1, 0, hit, tb, prim, u, v, front, gi);
    RefOut o; o.hit = hit; o.t = __uint_as_float(tb); o.prim = prim; o.u = 0.f; o.v = 0.f; o.front = 0; o.gas_index = 0; o.pad = 0;
    params.out[i] = o;
}

extern "C" __global__ void __intersection__box()
{
    const unsigned id = optixGetPrimitiveIndex(), ri = optixGetLaunchIndex().x, ex = params.excl[ri];
    if (id == ex) return;
    const float3 o = optixGetWorldRayOrigin(), d = optixGetWorldRayDirection();
    const float *b = params.boxes + 6 * id;
    float tn = optixGetRayTmin(), tf = optixGetRayTmax();
    const float oo[3] = {o.x, o.y, o.z}, dd[3] = {d.x, d.y, d.z};
    for (int a = 0; a < 3; ++a) {
        const float inv = 1.f / dd[a]; float t0 = (b[a] - oo[a]) * inv, t1 = (b[a + 3] - oo[a]) * inv;
        if (t0 > t1) { const float s = t0; t0 = t1; t1 = s; }
        if (dd[a] == 0.f) { if (oo[a] < b[a] || oo[a] > b[a + 3]) return; continue; }
        tn = fmaxf(tn, t0); tf = fminf(tf, t1);
    }
    if (!(tn <= tf)) return;
    float tr = tn;
    if (ex == 0xffffffffu) {   // inspector tie break: among boxes entered at (nearly) the same distance prefer the one whose centre is nearest to the entry point
        float s = 0.f, dg = 0.f;
        for (int a = 0; a < 3; ++a) { const float c = 0.5f * (b[a] + b[a + 3]), h = 0.5f * (b[a + 3] - b[a]), p = oo[a] + dd[a] * tn; s = fmaxf(s, fabsf(p - c) / fmaxf(h, 1e-12f)); dg += h * h; }
        tr = tn + 1e-5f * sqrtf(dg) * s;
        if (tr > optixGetRayTmax()) tr = tn;
    }
    optixReportIntersection(tr, 0);
}

extern "C" __global__ void __closesthit__box()
{
    optixSetPayload_0(1u);
    optixSetPayload_1(__float_as_uint(optixGetRayTmax()));
    optixSetPayload_2(optixGetPrimitiveIndex());
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
