// Phase 10i test 1 (a): plain CUDA replica of the Conc_4 stream kernel (setup3.cu compute type 14 l2fp<MB>, with the page spread option p<P>), finite run, no TTU, no OptiX.
// usage: l2probe FOOTPRINT_MB PAGES ITERS   (PAGES 0 = the l2fp hash: every lane its own random line; P >= 1: the 32 lanes of a warp load use P distinct 2 MB pages, 32 / P lines each)
// Launch 1 warms the L2, launch 2 is the one ncu profiles (--launch-skip 1). Prints lines per second of the second launch.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
__global__ void stream(const uint4 *buf, size_t nbuf, int pages, int iters, unsigned long long *sink)
{
    const unsigned NL = (unsigned)(nbuf / 8); const unsigned NP = NL / 16384u ? NL / 16384u : 1u;   // 2 MB page = 16384 lines of 128 B
    uint4 acc = make_uint4(0, 0, 0, 0);
    for (int it = 0; it < iters; ++it) {
        for (int k = 0; k < 32; ++k) {
            unsigned L;
            if (pages == 0) { const unsigned h = ((unsigned)threadIdx.x * 7919u + (unsigned)k * 4099u + (unsigned)it * 1021u + blockIdx.x * 37u) * 2654435761u; L = (h >> 5) % NL; }
            else {
                const unsigned grp = (threadIdx.x & 31) / (32 / pages);   // lanes of a group share a page
                const unsigned hp = ((((unsigned)(threadIdx.x >> 5)) * 131u + grp * 977u + (unsigned)k * 4099u + (unsigned)it * 1021u + blockIdx.x * 37u) * 2654435761u);
                const unsigned hl = ((unsigned)threadIdx.x * 7919u + (unsigned)k * 31u + (unsigned)it * 17u + blockIdx.x * 5u) * 2246822519u;
                L = ((hp >> 7) % NP) * 16384u + ((hl >> 9) % 16384u);
            }
            const uint4 v = __ldcg(buf + ((size_t)L * 8 + (threadIdx.x & 7)));
            acc.x += v.x; acc.y ^= v.y; acc.z += v.z; acc.w ^= v.w;
        }
    }
    if (acc.x == 0xdeadbeef) atomicAdd(sink, 1ull);
}
int main(int argc, char **argv)
{
    const int mb = argc > 1 ? atoi(argv[1]) : 64, pages = argc > 2 ? atoi(argv[2]) : 0, iters = argc > 3 ? atoi(argv[3]) : 2000;
    if (pages && (32 % pages)) { fprintf(stderr, "PAGES must divide 32\n"); return 2; }
    int dev = 0; cudaGetDevice(&dev); cudaDeviceProp pr; cudaGetDeviceProperties(&pr, dev);
    const size_t nbuf = (size_t)mb * 65536, nalloc = (size_t)1 << 26;   // footprint in uint4; the buffer is 1 GB as in setup3
    uint4 *buf; unsigned long long *sink; cudaMalloc((void **)&buf, nalloc * 16); cudaMemset(buf, 1, nalloc * 16); cudaMalloc((void **)&sink, 8); cudaMemset(sink, 0, 8);
    const int grid = pr.multiProcessorCount * 4;   // 4 blocks of 384 threads per SM (the sib arms: lim 4 on half of the SMs in the real runs; here all SMs)
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    for (int l = 0; l < 2; ++l) {
        cudaEventRecord(a); stream<<<grid, 384>>>(buf, nbuf, pages, iters, sink); cudaEventRecord(b); cudaEventSynchronize(b);
        float ms; cudaEventElapsedTime(&ms, a, b); const double lines = (double)grid * 12 * 32 * 32 * iters;
        printf("launch %d: %.3f ms, %.2f G lines/s (%d blocks, footprint %d MB, pages %d)\n", l, ms, lines / ms / 1e6, grid, mb, pages);
    }
    cudaError_t e = cudaGetLastError(); if (e) { fprintf(stderr, "%s\n", cudaGetErrorString(e)); return 1; }
    return 0;
}
