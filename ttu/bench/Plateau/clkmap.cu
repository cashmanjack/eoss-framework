// Plateau (b), idea 2: do the SMs of a GPC share a clock? Each SM (one 1024 thread block per SM) records clock64 and
// the global timer at the start and after ~MS milliseconds; per SM clock rate = delta clock / delta global timer.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
__device__ __forceinline__ unsigned long long gt() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__global__ void __launch_bounds__(1024) k(unsigned long long *o, unsigned long long ns)
{
    if (threadIdx.x) return;
    unsigned s; asm volatile("mov.u32 %0, %%smid;" : "=r"(s));
    const unsigned long long c0 = clock64(), g0 = gt(); unsigned long long c1, g1;
    do { c1 = clock64(); g1 = gt(); } while (g1 - g0 < ns);
    unsigned long long *p = o + 5 * blockIdx.x; p[0] = s; p[1] = c0; p[2] = g0; p[3] = c1; p[4] = g1;
}
int main(int argc, char **argv)
{
    const unsigned long long ns = argc > 1 ? atoll(argv[1]) : 50000000ull;
    unsigned long long *d, h[142 * 5]; cudaMalloc(&d, sizeof h);
    k<<<142, 1024>>>(d, ns); if (cudaDeviceSynchronize()) return 1; cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost);
    for (int b = 0; b < 142; ++b) { const unsigned long long *p = h + 5 * b;
        printf("%u %llu %llu %llu %llu %.6f\n", (unsigned)p[0], p[1], p[2], p[3], p[4], (double)(p[3] - p[1]) / (double)(p[4] - p[2])); }
}
