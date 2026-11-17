// Sched_1 (plain CUDA, no TTU): which ready warp issues when several are ready on one scheduler of one SM.
//   sched1 MASK_HEX SEGLEN DEP LAUNCHES OUT.json     one block of 32 warps (1024 threads); warp w runs iff bit w of MASK is set; all warps are released together by one bar.sync.
//   Every active warp runs 64 segments of SEGLEN FMAs (DEP=0: 8 independent chains, always ready; DEP=1: one dependent chain, ready every 4 cycles) and reads the SM clock after the first release and after each segment.
//   Output JSON: per launch the SM id and per active warp its 65 clock values (release, then end of each segment); the order and spacing of the stamps across warps show who issues when.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define NSEG 64
__device__ __forceinline__ unsigned clk() { unsigned c; asm volatile("mov.u32 %0, %%clock;" : "=r"(c)); return c; }
template <int L, int DEP>
__global__ void sched1(unsigned *out, unsigned *smid_out, unsigned mask, float seed)
{
    const int w = threadIdx.x >> 5;
    float a[8]; for (int i = 0; i < 8; ++i) a[i] = seed + i * 1e-3f + threadIdx.x * 1e-6f;
    __shared__ unsigned st[32 * (NSEG + 1)]; unsigned *my = st + w * (NSEG + 1);
    __syncthreads();
    if (!((mask >> w) & 1u)) return;
    if ((threadIdx.x & 31) == 0) my[0] = clk();
#pragma unroll
    for (int s = 0; s < NSEG; ++s) {
        if (DEP) { _Pragma("unroll") for (int k = 0; k < L; ++k) a[0] = fmaf(a[0], 1.0000001f, 0.5f); }
        else { _Pragma("unroll") for (int k = 0; k < L; ++k) a[k & 7] = fmaf(a[k & 7], 1.0000001f, 0.5f); }
        asm volatile("" : "+f"(a[0]), "+f"(a[1]), "+f"(a[2]), "+f"(a[3]), "+f"(a[4]), "+f"(a[5]), "+f"(a[6]), "+f"(a[7]));
        { unsigned c = clk(); if ((threadIdx.x & 31) == 0) my[s + 1] = c; }
    }
    if ((threadIdx.x & 31) == 0) { for (int s = 0; s <= NSEG; ++s) out[w * (NSEG + 1) + s] = my[s]; { unsigned sm; asm volatile("mov.u32 %0, %%smid;" : "=r"(sm)); smid_out[w] = sm; } }
    float t = 0; for (int i = 0; i < 8; ++i) t += a[i]; if (t == 12345.f) out[0] = 1;
}
int main(int argc, char **argv)
{
    if (argc < 6) { fprintf(stderr, "sched1 MASK_HEX SEGLEN(8|32|64) DEP LAUNCHES OUT.json\n"); return 2; }
    unsigned mask = (unsigned)strtoul(argv[1], 0, 16); int L = atoi(argv[2]), dep = atoi(argv[3]), n = atoi(argv[4]);
    FILE *f = fopen(argv[5], "wx"); if (!f) { fprintf(stderr, "exists or cannot open %s\n", argv[5]); return 2; }
    unsigned *d, *ds; cudaMalloc(&d, 32 * (NSEG + 1) * 4); cudaMalloc(&ds, 32 * 4);
    std::vector<unsigned> h(32 * (NSEG + 1)), hs(32);
    fprintf(f, "{\"mask\": %u, \"seglen\": %d, \"dep\": %d, \"launches\": [", mask, L, dep);
    for (int it = 0; it < n; ++it) {
        cudaMemset(d, 0, 32 * (NSEG + 1) * 4);
        if (L == 8) { if (dep) sched1<8, 1><<<1, 1024>>>(d, ds, mask, 1.f); else sched1<8, 0><<<1, 1024>>>(d, ds, mask, 1.f); }
        else if (L == 32) { if (dep) sched1<32, 1><<<1, 1024>>>(d, ds, mask, 1.f); else sched1<32, 0><<<1, 1024>>>(d, ds, mask, 1.f); }
        else { if (dep) sched1<64, 1><<<1, 1024>>>(d, ds, mask, 1.f); else sched1<64, 0><<<1, 1024>>>(d, ds, mask, 1.f); }
        cudaError_t e = cudaDeviceSynchronize(); if (e != cudaSuccess) { fprintf(stderr, "cuda error %s\n", cudaGetErrorString(e)); return 3; }
        cudaMemcpy(h.data(), d, h.size() * 4, cudaMemcpyDeviceToHost); cudaMemcpy(hs.data(), ds, 128, cudaMemcpyDeviceToHost);
        fprintf(f, "%s{\"sm\": %u, \"warps\": {", it ? ", " : "", hs[__builtin_ctz(mask)]);
        bool first = true;
        for (int w = 0; w < 32; ++w) if ((mask >> w) & 1u) { fprintf(f, "%s\"%d\": [", first ? "" : ", ", w); first = false; for (int s = 0; s <= NSEG; ++s) fprintf(f, "%s%u", s ? "," : "", h[w * (NSEG + 1) + s]); fprintf(f, "]"); }
        fprintf(f, "}}");
    }
    fprintf(f, "]}\n"); fclose(f); printf("ok mask %x seglen %d dep %d launches %d\n", mask, L, dep, n); return 0;
}
