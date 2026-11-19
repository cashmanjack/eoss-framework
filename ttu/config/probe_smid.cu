// Plain CUDA (no TTU). Which SM ids exist and how blocks land on them. Prints SM count, %nsmid, the set of SM ids seen,
// L2 size, clocks, cluster launch support. Many blocks of 1024 threads so every SM gets one; ids are recorded per block.
#include <cstdio>
#include <set>
__global__ void k(unsigned *o, unsigned *nsm, unsigned long long spin) {
    if (threadIdx.x == 0) { unsigned s, n; asm volatile("mov.u32 %0, %%smid;" : "=r"(s)); asm volatile("mov.u32 %0, %%nsmid;" : "=r"(n)); o[blockIdx.x] = s; *nsm = n; }
    unsigned long long t = clock64(); while (clock64() - t < spin) {}
}
int main() {
    int dv = 0; cudaGetDevice(&dv); cudaDeviceProp p; cudaGetDeviceProperties(&p, dv);
    int nsm = p.multiProcessorCount, clk = 0, mclk = 0, cl = 0;
    cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, dv); cudaDeviceGetAttribute(&mclk, cudaDevAttrMemoryClockRate, dv); cudaDeviceGetAttribute(&cl, cudaDevAttrClusterLaunch, dv);
    printf("name %s cc %d.%d SMs %d L2 %d bytes (%.1f MB) clock %d kHz memclock %d kHz bus %d bits regs/SM %d smem/SM %zu maxthreads/SM %d cluster launch %d\n",
           p.name, p.major, p.minor, nsm, p.l2CacheSize, p.l2CacheSize / 1048576.0, clk, mclk, p.memoryBusWidth, p.regsPerMultiprocessor, p.sharedMemPerMultiprocessor, p.maxThreadsPerMultiProcessor, cl);
    unsigned *d, *dn, h[1024], n = 0; cudaMalloc(&d, 1024 * 4); cudaMalloc(&dn, 4);
    std::set<unsigned> seen; int bad = 0, diff0 = 0; unsigned h0[1024];
    for (int rep = 0; rep < 6; ++rep) {
        cudaMemset(d, 0xff, 1024 * 4); k<<<nsm, 1024>>>(d, dn, 20000000ull);
        if (cudaDeviceSynchronize()) { printf("cuda error\n"); return 2; }
        cudaMemcpy(h, d, nsm * 4, cudaMemcpyDeviceToHost); cudaMemcpy(&n, dn, 4, cudaMemcpyDeviceToHost);
        if (rep == 0) for (int b = 0; b < nsm; ++b) h0[b] = h[b];
        for (int b = 0; b < nsm; ++b) { if (h[b] != h0[b]) ++diff0; seen.insert(h[b]); unsigned want = b < nsm / 2 ? 2 * b : 2 * (b - nsm / 2) + 1; if (h[b] != want) ++bad; }
        if (rep == 0) { printf("block->SM, all:"); for (int b = 0; b < nsm; ++b) printf(" %u", h[b]); printf("\n"); }
        if (rep == 0) { printf("block 0..7 on SM:"); for (int b = 0; b < 8; ++b) printf(" %u", h[b]); printf("; blocks %d..%d on SM:", nsm / 2 - 2, nsm / 2 + 3); for (int b = nsm / 2 - 2; b < nsm / 2 + 4; ++b) printf(" %u", h[b]); printf("\n"); }
    }
    printf("%%nsmid %u, distinct SM ids seen %zu (min %u max %u), Ada rule mismatches over 6 launches %d, blocks on a different SM than in launch 0: %d\n", n, seen.size(), *seen.begin(), *seen.rbegin(), bad, diff0);
    printf("ids missing from 0..nsmid-1:"); int miss = 0; for (unsigned i = 0; i < n; ++i) if (!seen.count(i)) { printf(" %u", i); ++miss; } printf(" (%d)\n", miss);
    return 0;
}
