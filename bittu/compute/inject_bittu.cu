// Device functions injected by bittu.cu (compiled with -Xptxas -astoolspatch --keep-device-functions).
#include <stdint.h>
#include <stdarg.h>
#include "utils/utils.h"
#include "bittu_rec.h"
static __device__ __forceinline__ uint64_t clk() { uint64_t t; asm volatile("mov.u64 %0, %%clock64;" : "=l"(t)); return t; }
static __device__ __forceinline__ int tidg() { return get_global_warp_id() * 32 + get_laneid(); }
// calibration probe: only a clock read, placed right before bittu_pre at the same point; (t0b - t0a) is the cost of one probe
extern "C" __device__ __noinline__ void bittu_cal(int pred, uint64_t rec, uint64_t nseen, int nth) {
    const uint64_t t = clk(); const int tid = tidg(); if (!pred || tid >= nth) return;
    const int k = ((int *)nseen)[tid]; if (k < BITTU_INST) ((BittuRec *)rec)[tid * BITTU_INST + k].t0a = t;
}
// before the group: clock, ray words (variadic registers), instance counter
extern "C" __device__ __noinline__ void bittu_pre(int pred, uint64_t rec, uint64_t nseen, int nth, int nr, ...) {
    const uint64_t t = clk(); const int tid = tidg(); if (!pred || tid >= nth) return;
    const int k = ((int *)nseen)[tid]; ((int *)nseen)[tid] = k + 1; if (k >= BITTU_INST) return;
    BittuRec &r = ((BittuRec *)rec)[tid * BITTU_INST + k]; r.t0b = t; r.mask_pre = __activemask();
    va_list vl; va_start(vl, nr); for (int i = 0; i < nr && i < 8; ++i) r.ray[i] = va_arg(vl, uint32_t); va_end(vl);
}
// after the group (after the instruction that waits for the results): clock, result registers
extern "C" __device__ __noinline__ void bittu_post(int pred, uint64_t rec, uint64_t nseen, int nth, int nr, ...) {
    const uint64_t t = clk(); const int tid = tidg(); if (!pred || tid >= nth) return;
    const int k = ((int *)nseen)[tid] - 1; if (k < 0 || k >= BITTU_INST) return;
    BittuRec &r = ((BittuRec *)rec)[tid * BITTU_INST + k]; r.t1 = t; r.mask_post = __activemask();
    va_list vl; va_start(vl, nr); for (int i = 0; i < nr && i < BITTU_MAXR; ++i) r.res[i] = va_arg(vl, uint32_t); va_end(vl);
}
