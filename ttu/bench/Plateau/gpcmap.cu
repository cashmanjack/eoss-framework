// Plateau (b): which SMs share a GPC. A spin kernel runs only on the SMs in a mask (blocks of 1024 threads, one per SM,
// other blocks exit); ncu gpc__cycles_active.sum / one GPC's elapsed cycles = number of GPCs that had an SM running.
//   gpcmap MASK_LIST      e.g. 0,1,12   (SM ids, comma separated)
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
__global__ void __launch_bounds__(1024) spin(const unsigned long long *mask, unsigned long long n, unsigned *sink)
{
    unsigned s; asm volatile("mov.u32 %0, %%smid;" : "=r"(s));
    if (!((mask[s >> 6] >> (s & 63)) & 1)) return;
    if (threadIdx.x) return;
    const unsigned long long t0 = clock64(); unsigned x = s;
    while (clock64() - t0 < n) x = x * 1664525u + 1013904223u;
    if (x == 0x12345678u) sink[0] = x;
}
int main(int argc, char **argv)
{
    unsigned long long m[3] = {0, 0, 0};
    for (char *t = strtok(argv[1], ","); t; t = strtok(nullptr, ",")) { int s = atoi(t); m[s >> 6] |= 1ull << (s & 63); }
    unsigned long long *dm; unsigned *sink; cudaMalloc(&dm, 24); cudaMalloc(&sink, 4); cudaMemcpy(dm, m, 24, cudaMemcpyHostToDevice);
    spin<<<142, 1024>>>(dm, argc > 2 ? atoll(argv[2]) : 2000000ull, sink);
    cudaError_t e = cudaDeviceSynchronize(); printf("%s\n", cudaGetErrorString(e)); return e != cudaSuccess;
}
