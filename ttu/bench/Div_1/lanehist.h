// Div_1 lane histogram probe (OptiX callbacks, plain CUDA clock64). One slot of LH_STRIDE u64 per warp (warp = launch index x >> 5, 1D launches):
//   [p]       callbacks entered with p active lanes (1..32)          [33 + p]  cycles inside those callbacks (entry to exit)
//   [66 + p]  interval cycles: previous callback exit (or raygen start) to this callback exit, attributed to this callback's lane count
//   [99 + p]  handoffs whose first callback had p lanes (a callback entering at least LH_GAP cycles after the previous exit starts a new handoff)
//   [132] last exit clock   [133] raygen start   [134] raygen end   [135] callbacks of the warp
// Only the leader lane (lowest lane active at entry) writes; callbacks of one warp are sequential, so plain read-modify-write is safe.
#pragma once
#define LH_STRIDE 136
#define LH_GAP 300ull
#if defined(__CUDACC__)
__device__ __forceinline__ unsigned lh_lane() { unsigned l; asm volatile("mov.u32 %0, %%laneid;" : "=r"(l)); return l; }
struct LHE { unsigned long long t0; unsigned mask; };
__device__ __forceinline__ LHE lh_enter(unsigned long long *lh) { LHE e; e.mask = __activemask(); e.t0 = lh ? clock64() : 0ull; return e; }
__device__ __forceinline__ void lh_exit(unsigned long long *lh, unsigned warp, const LHE &e)
{
    if (!lh) return;
    if (lh_lane() != (unsigned)(__ffs(e.mask) - 1)) return;
    unsigned long long *b = lh + (size_t)warp * LH_STRIDE; const unsigned p = __popc(e.mask); const unsigned long long t1 = clock64();
    unsigned long long prev = b[132]; const bool first = prev == 0ull; if (first) prev = b[133];
    b[p] += 1; b[33 + p] += t1 - e.t0; b[66 + p] += t1 - prev; if (first || e.t0 - b[132] >= LH_GAP) b[99 + p] += 1; b[132] = t1; b[135] += 1;
}
__device__ __forceinline__ void lh_rg_start(unsigned long long *lh, unsigned warp) { if (lh && lh_lane() == 0) { unsigned long long *b = lh + (size_t)warp * LH_STRIDE; if (b[133] == 0ull) b[133] = clock64(); } }
__device__ __forceinline__ void lh_rg_end(unsigned long long *lh, unsigned warp) { if (lh) { unsigned long long *b = lh + (size_t)warp * LH_STRIDE; atomicMax(&b[134], (unsigned long long)clock64()); } }   // every lane: lanes of a warp finish raygen at different times
// Cheaper probe modes (Phase 10c): lh_gate returns null for warps outside the sample (warp & smask != 0); with notime the callbacks only count lanes (no clock reads, no raygen stamps).
__device__ __forceinline__ unsigned long long *lh_gate(unsigned long long *lh, unsigned warp, unsigned smask) { return (lh && (warp & smask) == 0u) ? lh : nullptr; }
__device__ __forceinline__ LHE lh_enter2(unsigned long long *lh, unsigned notime) { LHE e; e.mask = __activemask(); e.t0 = (lh && !notime) ? clock64() : 0ull; return e; }
__device__ __forceinline__ void lh_exit2(unsigned long long *lh, unsigned warp, const LHE &e, unsigned notime)
{
    if (!lh) return;
    if (!notime) { lh_exit(lh, warp, e); return; }
    if (lh_lane() != (unsigned)(__ffs(e.mask) - 1)) return;
    unsigned long long *b = lh + (size_t)warp * LH_STRIDE; b[__popc(e.mask)] += 1; b[135] += 1;
}
#endif
