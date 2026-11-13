// Plain CUDA (no TTU). Launch SMs blocks of 1024 threads (one block per SM can be resident) and record the SM each block lands on.
// Checks the placement rule the harness assumes: block b on SM 2b for b < SMs/2, then SM 2(b - SMs/2) + 1. Exit 0 if it holds.
#include <cstdio>
#include <cstdlib>
__global__ void k(unsigned *o, unsigned long long spin) {
    if (threadIdx.x == 0) { unsigned s; asm volatile("mov.u32 %0, %%smid;" : "=r"(s)); o[blockIdx.x] = s; }
    unsigned long long t = clock64(); while (clock64() - t < spin) {}
}
int main() {
    int nsm = 0, dv = 0; cudaGetDevice(&dv); cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dv);
    unsigned *d, h[1024]; cudaMalloc(&d, 1024 * 4);
    int bad = 0, pass_bad = 0;
    for (int rep = 0; rep < 5; ++rep) {
        cudaMemset(d, 0xff, 1024 * 4); k<<<nsm, 1024>>>(d, 20000000ull);
        if (cudaDeviceSynchronize()) { printf("cuda error\n"); return 2; }
        cudaMemcpy(h, d, nsm * 4, cudaMemcpyDeviceToHost); int b2 = 0;
        for (int b = 0; b < nsm; ++b) { unsigned want = b < nsm / 2 ? 2 * b : 2 * (b - nsm / 2) + 1; if (h[b] != want) { if (!b2) printf("rep %d: block %d on SM %u, rule says %u\n", rep, b, h[b], want); ++b2; } }
        bad += b2; if (rep == 0) { printf("first blocks:"); for (int b = 0; b < 8; ++b) printf(" %u", h[b]); printf(" ... block %d: %u\n", nsm - 1, h[nsm - 1]); }
    }
    printf("device SMs %d, mismatches over 5 launches: %d\n", nsm, bad);
    return bad ? 1 : 0;
}
