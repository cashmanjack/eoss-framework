// Device functions injected by kill.cu (compiled with -Xptxas -astoolspatch --keep-device-functions).
#include <stdint.h>
#include <stdarg.h>
#include "utils/utils.h"
#define NREG 12
// Counters and the register buffer live in the tool (managed memory); their addresses come as arguments.
extern "C" __device__ __noinline__ void bittu_before(uint64_t cnt) { atomicAdd((unsigned long long *)cnt, 1ull); }
extern "C" __device__ __noinline__ void bittu_after(uint64_t cnt) { atomicAdd((unsigned long long *)cnt, 1ull); }
extern "C" __device__ __noinline__ void bittu_regs(int pred, uint64_t cnt, uint64_t buf, ...) {
    atomicAdd((unsigned long long *)cnt, 1ull);
    uint32_t (*regbuf)[NREG] = (uint32_t (*)[NREG])buf;
    if (!pred || get_global_warp_id() != 0) return;
    va_list vl; va_start(vl, buf);
    const unsigned lane = get_laneid();
    for (int r = 0; r < NREG; ++r) regbuf[lane][r] = va_arg(vl, uint32_t);
    va_end(vl);
}
