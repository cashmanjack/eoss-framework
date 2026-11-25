// Treelet_3 programs (OptiX only, no TTU code anywhere). Pipeline with single level instancing.
//   first   : top level step of every ray in natural order (IAS of treelet shells), key + cursor + histogram
//   fused   : persistent pinned warps drain per TPC treelet queues, trace the ray in the treelet GAS, find its next treelet with the top level step, requeue (all in one kernel)
//   replay  : persistent pinned warps over a presorted level list (grouping reused), no top level step, no queue
//   iasnat  : natural order trace of the IAS whose instances are the real treelet GASes (two level overhead baseline)
#include <optix.h>
#include "treelet3_params.h"
extern "C" __constant__ Params params;
#define FULL __activemask()
#define NONE 0xffffffffu

#define HB(code) do { if (params.warp_sm && (lane == 0 || lane == 31)) { const unsigned o_ = wid * 8 + (lane ? 4u : 0u); *(volatile unsigned *)&params.warp_sm[o_] = iter; *(volatile unsigned *)&params.warp_sm[o_ + 1] = (code); } } while (0)
struct Top { unsigned found; unsigned id; float tp; };
// next treelet along the ray after the cursor (tprev, idprev) in order of entry, up to distance tbound. Origin pulled back by params.pull so that the entry face of a box that contains the origin is a front face hit.
static __device__ __forceinline__ Top top_next(const Ray8 &r, float tprev, unsigned i0, unsigned i1, unsigned i2, float tbound)
{
    // i0, i1, i2: the last three treelets visited (16 bit ids, 0xffff none). Boxes that enter the ray within a few ulps of the cursor are found again (tmin is lowered by delta) and the visited ones are dropped by id in the any hit program, so an exact or ulp level tie never loses a box.
    const float s = params.pull, delta = 1e-6f * tprev;
    unsigned p0 = 0, p1 = 0, p2 = 0, p3 = __float_as_uint(tprev), p4 = i0 & 0xffffu, p5 = i1 & 0xffffu, p6 = i2 & 0xffffu;
    optixTrace(params.ias_shell, make_float3(r.o[0] - s * r.d[0], r.o[1] - s * r.d[1], r.o[2] - s * r.d[2]), make_float3(r.d[0], r.d[1], r.d[2]),
               fmaxf(tprev - delta, 0.f), s + tbound, 0.f, OptixVisibilityMask(255), OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES, 1, 1, 0, p0, p1, p2, p3, p4, p5, p6);
    Top o; o.found = p0; o.tp = __uint_as_float(p1); o.id = p2; return o;
}

static __device__ __forceinline__ void trace_tl(unsigned t, const Ray8 &r, float tmax, unsigned &h, unsigned &tb, unsigned &pr, unsigned &u, unsigned &v, unsigned &fr)
{
    unsigned gi = 0; h = 0;
    optixTrace(params.tl_handle[t], make_float3(r.o[0], r.o[1], r.o[2]), make_float3(r.d[0], r.d[1], r.d[2]), r.tmin, tmax, 0.f, OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE, 0, 1, 0, h, tb, pr, u, v, fr, gi);
}

extern "C" __global__ void __raygen__first()
{
    const unsigned i = optixGetLaunchIndex().x;
    if (i >= *params.n_dev) return;
    const Ray8 r = params.rays[i];
    const Top tr = top_next(r, 0.f, 0xffffu, 0xffffu, 0xffffu, r.tmax);
    Cur c; c.t = tr.tp; c.id = tr.found ? tr.id : params.C; c.k = 0; c.pad = 0xffffffffu; params.cur[i] = c;
    RefOut o; o.hit = 0; o.t = r.tmax; o.prim = 0; o.u = 0.f; o.v = 0.f; o.front = 0; o.gas_index = 0; o.pad = 0; params.out[i] = o;
    atomicAdd(&params.hist[c.id], 1u);
}

extern "C" __global__ void __raygen__tltest()   // debug: trace every ray in the treelet params.tl[i] (the oracle), result like iasnat
{
    const unsigned i = optixGetLaunchIndex().x;
    if (i >= *params.n_dev) return;
    const Ray8 r = params.rays[i]; unsigned h, tb, pr, uu, vv, fr; trace_tl(params.tl[i], r, r.tmax, h, tb, pr, uu, vv, fr);
    RefOut o; o.hit = h; o.t = __uint_as_float(tb); o.prim = h ? params.tri_sorted[params.toff[params.tl[i]] + pr] : 0u; o.u = 0.f; o.v = 0.f; o.front = 0; o.gas_index = params.tl[i]; o.pad = 0; params.out[i] = o;
}
extern "C" __global__ void __raygen__iasnat()
{
    const unsigned i = optixGetLaunchIndex().x;
    if (i >= *params.n_dev) return;
    const Ray8 r = params.rays[i];
    unsigned hit = 0, tb = 0, prim = 0, u = 0, v = 0, front = 0, gi = 0;
    optixTrace(params.ias_real, make_float3(r.o[0], r.o[1], r.o[2]), make_float3(r.d[0], r.d[1], r.d[2]), r.tmin, r.tmax, 0.f, OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE, 0, 1, 0, hit, tb, prim, u, v, front, gi);
    RefOut o; o.hit = hit; o.t = __uint_as_float(tb); o.prim = prim; o.u = __uint_as_float(u); o.v = __uint_as_float(v); o.front = front; o.gas_index = gi; o.pad = 0;
    params.out[i] = o;
}

// number of entries of queue t that are safe to claim: all reserved slots when every reserved slot is published (quiescent), else only the first level entries (written before the kernel)
static __device__ __forceinline__ unsigned qlimit(unsigned t)
{
    const unsigned t1 = *(volatile unsigned *)&params.qtail[t], dn = *(volatile unsigned *)&params.qpub[t], t2 = *(volatile unsigned *)&params.qtail[t];
    const unsigned lim = (dn == t1 && t1 == t2) ? t1 : min(params.hist[t], t1);
    return min(lim, params.qcap[t]);
}

// first treelet in the TPC's range at or after its pointer that has unclaimed entries (uniform inside the convergence group am), NONE if the range is empty
static __device__ __forceinline__ unsigned probe(unsigned q, unsigned am, unsigned rank, unsigned gsize)
{
    const unsigned lo = params.tlo[q], hi = params.thi[q], len = hi - lo;
    if (len == 0) return NONE;
    unsigned c = *(volatile unsigned *)&params.tcur[q]; c = __shfl_sync(am, c, __ffs(am) - 1); if (c < lo || c >= hi) c = lo;
    const unsigned sh = c - lo;
    for (unsigned off = 0; off < len; off += gsize) {
        const unsigned k = off + rank; const bool inr = k < len;
        unsigned kk = sh + (inr ? k : 0u); if (kk >= len) kk -= len; const unsigned t = lo + kk;
        const unsigned h = *(volatile unsigned *)&params.qhead[t]; const unsigned tl = qlimit(t); const bool ne = inr && h < tl;
        const unsigned m = __ballot_sync(am, ne);
        if (m) { const unsigned src = __ffs(m) - 1; return __shfl_sync(am, t, src); }
    }
    return NONE;
}

extern "C" __global__ void __raygen__fused()
{
    unsigned lane, smid;
    asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane));
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    const unsigned li = optixGetLaunchIndex().x, wid = li >> 5, nq = params.nq;
    const unsigned q0 = params.pinned ? min(smid >> 1, nq - 1) : (wid % nq);
    unsigned long long c0 = 0, c1 = 0, c2 = 0, c3 = 0, c4 = 0;
    unsigned steal = 0, nchunk = 0, nsteal = 0, kmax = 0, nvis = 0, nreq = 0, nover = 0, nstep = 0;
    for (unsigned iter = 0;; ++iter) {
        if (iter > (1u << 18)) { if (lane == 0) atomicAdd(&params.cyc[5], 1000ull); break; }
        __syncwarp();
        const unsigned am = __activemask(), gsize = __popc(am), rank = __popc(am & ((1u << lane) - 1u)), leader = __ffs(am) - 1u;   // robust to a warp that is split into convergence groups
        const unsigned long long t0 = clock64(); HB(1);
        unsigned tsel = NONE, qsel = q0;
        for (unsigned a = 0; a <= params.steal_tries && tsel == NONE; ++a) {
            const unsigned q = a == 0 ? q0 : (q0 + a + steal) % nq;
            tsel = probe(q, am, rank, gsize); qsel = q;
        }
        unsigned s0 = 0, n = 0; HB(2);
        if (tsel != NONE) {   // branch free claim: lane 0's view and lane 0's single CAS, broadcast (all control flow warp uniform)
            for (int tr = 0; tr < 8; ++tr) {
                const unsigned h = *(volatile unsigned *)&params.qhead[tsel], tl = qlimit(tsel);
                const unsigned hh = __shfl_sync(am, h, leader), tt = __shfl_sync(am, tl, leader);
                if (hh >= tt) break;
                const unsigned m = min(gsize, tt - hh);
                unsigned old = hh; if (lane == leader) old = atomicCAS(&params.qhead[tsel], hh, hh + m);
                old = __shfl_sync(am, old, leader);
                if (old == hh) { s0 = hh; n = m; break; }
            }
            if (lane == leader && n != 0) *(volatile unsigned *)&params.tcur[qsel] = tsel;
        }
        if (n == 0) {   // nothing claimed: finished, or idle. Idle warps still run the straight line body below with dummy rays so that all lanes of a warp keep arriving at the same trace call sites.
            unsigned o = *(volatile unsigned *)params.outstanding; o = __shfl_sync(am, o, leader);
            if (o == 0) break;
            steal += params.steal_tries; __nanosleep(256);
            c4 += clock64() - t0;
        }
        if (n && qsel != q0 && lane == leader) ++nsteal;
        if (n && lane == leader) ++nchunk;
        // Everything below is straight line code for all 32 lanes: inactive lanes trace a dummy zero length ray, the spin is a warp uniform loop. (A warp whose lanes drift apart deadlocks inside optixTrace: seen with lane 31 idling while lane 0 sat in a trace.)
        const unsigned tsel0 = tsel == NONE ? 0u : tsel;   // idle: a valid treelet index for the dummy trace
        const bool valid = rank < n; unsigned x = 0;
        volatile unsigned *pe = params.qbuf + (size_t)params.qbase[tsel0] + s0 + (valid ? rank : 0u);
        for (unsigned sp = 0;; ++sp) {
            x = valid ? *pe : 0u; const bool wait = valid && x == NONE;
            if (!__any_sync(__activemask(), wait)) break;
            if (sp > 4096) { if (wait) atomicAdd(&params.cnt[5], 1u); break; }
            __nanosleep(32);
        }
        const bool ok0 = valid && x != NONE; bool fin = valid && x == NONE;   // a timed out slot ends that ray (counted, the run is then invalid)
        const unsigned idx0 = ok0 ? (x & 0x7fffffu) : 0u; const unsigned ekk = x >> 23; HB(3);
        const unsigned long long t1 = clock64(); c0 += t1 - t0;
        const unsigned idx = idx0;
        const Ray8 r = params.rays[idx];
        const uint4 cv = __ldcg((const uint4 *)&params.cur[idx]); Cur c; c.t = __uint_as_float(cv.x); c.id = cv.y; c.k = cv.z; c.pad = cv.w;
        const bool ok = ok0 && (c.k & 63u) == ekk;   // an entry whose visit number does not match the ray's record is stale (counted)
        if (ok0 && !ok) atomicAdd(&params.cyc[7], 1ull);
        if (ok && params.tl) atomicAdd((unsigned *)&params.tl[idx], 1u);   // debug: times this ray was processed
        const unsigned hit0 = __ldcg(&params.out[idx].hit); const float tt0 = __ldcg(&params.out[idx].t);
        float tbest = hit0 ? tt0 : r.tmax; nvis += ok ? 1u : 0u;
        unsigned h = 0, tb = 0, pr = 0, uu = 0, vv = 0, fr = 0; HB(4);
        trace_tl(tsel0, r, ok ? tbest : 0.f, h, tb, pr, uu, vv, fr);
        if (ok && h) {
            atomicAdd(&params.cyc[6], 1ull);
            tbest = __uint_as_float(tb);
            const unsigned gp = params.tri_sorted[params.toff[tsel0] + pr];
            __stcg((uint4 *)&params.out[idx], make_uint4(1u, tb, gp, uu));
            __stcg(((uint4 *)&params.out[idx]) + 1, make_uint4(vv, fr, tsel0, 0u));
        }
        
        const unsigned long long t2 = clock64(); c1 += t2 - t1;
        if (ok && params.record && c.k < KMAX) params.vis[(size_t)idx * KMAX + c.k] = (unsigned short)tsel0;
        nstep += ok ? 1u : 0u; HB(5);
        const Top nx = top_next(r, c.t, tsel0, c.pad & 0xffffu, c.pad >> 16, ok ? tbest : 0.f);
        const unsigned long long t3 = clock64(); c2 += t3 - t2;
        HB(6);
        if (ok) {
            c.k++; if (c.k > kmax) kmax = c.k;
            bool go = nx.found != 0; if (go && c.k >= 64) { go = false; ++nover; }   // safety: a ray never visits more than 64 treelets
            if (go) { c.pad = (tsel0 & 0xffffu) | ((c.pad & 0xffffu) << 16); c.t = nx.tp; c.id = nx.id; }
            __stcg((uint4 *)&params.cur[idx], make_uint4(__float_as_uint(c.t), c.id, c.k, c.pad));
            if (go) {
                __threadfence();
                const unsigned slot = atomicAdd(&params.qtail[nx.id], 1u); ++nreq;
                if (slot < params.qcap[nx.id]) { *(volatile unsigned *)&params.qbuf[(size_t)params.qbase[nx.id] + slot] = idx | ((c.k & 63u) << 23); __threadfence(); atomicAdd(&params.qpub[nx.id], 1u); }
                else { ++nover; fin = true; }
            } else fin = true;
        }
        { const unsigned am2 = __activemask(); const unsigned fm = __ballot_sync(am2, fin); if (lane == (unsigned)(__ffs(am2) - 1) && fm) atomicSub(params.outstanding, (unsigned)__popc(fm)); }
        c3 += clock64() - t3;
    }
    { const unsigned amE = __activemask(); if (lane == (unsigned)(__ffs(amE) - 1)) { atomicAdd(&params.cyc[0], c0); atomicAdd(&params.cyc[1], c1); atomicAdd(&params.cyc[2], c2); atomicAdd(&params.cyc[3], c3); atomicAdd(&params.cyc[4], c4); atomicAdd(&params.cyc[5], 1ull); } }
    if (nchunk) atomicAdd(&params.cnt[0], nchunk); if (nsteal) atomicAdd(&params.cnt[1], nsteal); atomicMax(&params.cnt[6], kmax);
    // lane counters (per lane sums, written by every lane that has something)
    if (nreq) atomicAdd(&params.cnt[2], nreq);
    if (nvis) atomicAdd(&params.cnt[3], nvis);
    if (nover) atomicAdd(&params.cnt[4], nover);
    if (nstep) atomicAdd(&params.cnt[7], nstep);
}

// replay of one level: entries lvl_idx / lvl_ray / lvl_key (sorted by treelet), per TPC contiguous ranges (head / end per TPC), persistent warps, steal when the own range is empty. Straight line code per chunk (inactive lanes trace a dummy zero length ray).
extern "C" __global__ void __raygen__replay()
{
    unsigned lane, smid;
    asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane));
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    const unsigned q0 = min(smid >> 1, params.nq - 1);
    for (;;) {
        const unsigned am = __activemask(), gsize = __popc(am), rank = __popc(am & ((1u << lane) - 1u)), leader = __ffs(am) - 1u;
        unsigned ok = 0, base = 0, qq = q0;
        unsigned b = 0; if (lane == leader) b = atomicAdd(&params.lvl_head[q0], gsize);
        b = __shfl_sync(am, b, leader);
        if (b < params.lvl_end[q0]) { ok = 1; base = b; }
        else {
            for (unsigned k = 1; k < params.nq; ++k) {
                const unsigned q = (q0 + k) % params.nq;
                unsigned h = *(volatile unsigned *)&params.lvl_head[q]; h = __shfl_sync(am, h, leader);
                if (h >= params.lvl_end[q]) continue;
                b = 0; if (lane == leader) b = atomicAdd(&params.lvl_head[q], gsize);
                b = __shfl_sync(am, b, leader);
                if (b < params.lvl_end[q]) { ok = 1; base = b; qq = q; break; }
            }
        }
        if (!ok) break;
        const unsigned p = base + rank; const bool valid = p < params.lvl_end[qq]; const unsigned pp = valid ? p : 0u;
        const unsigned idx = params.lvl_idx[pp], t = params.lvl_key[pp]; const Ray8 r = params.lvl_ray[pp];
        float tm = r.tmax; unsigned hit0 = 0; RefOut o;
        if (params.lvl_level && valid) { o = params.out[idx]; hit0 = o.hit; if (hit0) tm = o.t; }
        unsigned h, tb, pr, uu, vv, fr;
        trace_tl(t, r, valid ? tm : 0.f, h, tb, pr, uu, vv, fr);
        if (valid) {
            if (h) { o.hit = 1; o.t = __uint_as_float(tb); o.prim = params.tri_sorted[params.toff[t] + pr]; o.u = __uint_as_float(uu); o.v = __uint_as_float(vv); o.front = fr; o.gas_index = t; o.pad = 0; params.out[idx] = o; }
            else if (!params.lvl_level) { o.hit = 0; o.t = r.tmax; o.prim = 0; o.u = 0.f; o.v = 0.f; o.front = 0; o.gas_index = 0; o.pad = 0; params.out[idx] = o; }
        }
    }
}

// ---- shell: entry faces of the treelet boxes (instances of one unit cube GAS, transform = the box), filter by cursor and by "box contains the origin" ----
extern "C" __global__ void __anyhit__shell()
{
    const float t = optixGetRayTmax();
    const unsigned id = optixGetInstanceId();
    if (id == optixGetPayload_4() || id == optixGetPayload_5() || id == optixGetPayload_6()) { optixIgnoreIntersection(); return; }   // already visited
    const float s = params.pull;
    if (t < s) {   // entry face behind the real origin: counts only if the box contains the origin
        const float3 o = optixGetWorldRayOrigin(), d = optixGetWorldRayDirection();
        const float x = o.x + s * d.x, y = o.y + s * d.y, z = o.z + s * d.z; const float *b = params.boxes + 6 * (size_t)id;
        if (!(x >= b[0] && x <= b[3] && y >= b[1] && y <= b[4] && z >= b[2] && z <= b[5])) optixIgnoreIntersection();
    }
}
extern "C" __global__ void __closesthit__shell()
{
    optixSetPayload_0(1u);
    optixSetPayload_1(__float_as_uint(optixGetRayTmax()));
    optixSetPayload_2(optixGetInstanceId());
}
extern "C" __global__ void __raygen__shelltest()   // one thread: out[0].hit/t/prim = found / entry parameter / id of the first box along rays[0] after the cursor (0, none)
{
    if (optixGetLaunchIndex().x != 0) return;
    const Ray8 r = params.rays[0];
    const RefOut in0 = params.out[0]; const bool cont = in0.gas_index == 1;
    const Top t = top_next(r, cont ? in0.t : 0.f, cont ? in0.prim : 0xffffu, cont ? in0.front : 0xffffu, cont ? in0.gas_index >> 1 : 0xffffu, r.tmax);
    RefOut o; o.hit = t.found; o.t = t.tp; o.prim = t.id; o.u = 0; o.v = 0; o.front = 0; o.gas_index = 0; o.pad = 0; params.out[0] = o;
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
    optixSetPayload_6(optixGetInstanceId());
}
