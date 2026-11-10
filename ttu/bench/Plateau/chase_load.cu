// Plateau (a) and (b): L2 load to use latency of a dependent chase while other SMs also chase (plain CUDA, no TTU).
//
//   chase_load SMLIST C N [MB]    SMLIST: comma list of SM ids that run (e.g. 0,2,4), or first:G (SMs 2b, then odd, as blocks land),
//                                 C chasers per SM (1, 32, 256, 1024), N dependent loads each, MB size of the shared cycle (default 6)
//
// One random cycle over MB megabytes (one element per 128 byte line), shared by all chasers; every chaser starts at its own
// random line, so addresses at any moment differ. Loads are ld.global.cg (L2, never L1). One block of 1024 threads per
// SM (other SMs' blocks exit). Chaser k of an SM: C = 1 thread 0; 32 lane 0 of every warp; 256 lanes 0..7; 1024 all.
// Prints: median / p5 / p95 cycles per load over all chasers, loads per microsecond over the whole GPU, per SM median.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>
__device__ __forceinline__ unsigned ldcg(const unsigned *p) { unsigned v; asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(p)); return v; }
__global__ void __launch_bounds__(1024) chase(const unsigned *a, const unsigned *start, const unsigned long long *mask, unsigned C, unsigned n,
                                              unsigned long long *cyc, unsigned *sink)
{
    unsigned s; asm volatile("mov.u32 %0, %%smid;" : "=r"(s));
    if (!((mask[s >> 6] >> (s & 63)) & 1)) return;
    const unsigned t = threadIdx.x, w = t >> 5, l = t & 31;
    const unsigned per = C >= 32 ? C / 32 : 1;            // chasing lanes per warp
    const bool on = C >= 32 ? l < per : t == 0;
    if (C < 32 && t >= 1) { /* only thread 0 */ }
    unsigned j = start[blockIdx.x * 1024 + t];
    __syncthreads();
    if (on) {
        const unsigned long long t0 = clock64();
        for (unsigned i = 0; i < n; ++i) j = ldcg(a + j);
        cyc[blockIdx.x * 1024 + t] = clock64() - t0;
        if (j == 0xfffffffeu) sink[0] = j;
    }
}
int main(int argc, char **argv)
{
    if (argc < 4) return 2;
    unsigned long long m[3] = {0, 0, 0}; std::vector<int> sms;
    if (!strncmp(argv[1], "first:", 6)) { const int g = atoi(argv[1] + 6); for (int b = 0; b < g; ++b) sms.push_back(b < 71 ? 2 * b : 2 * (b - 71) + 1); }
    else for (char *t = strtok(argv[1], ","); t; t = strtok(nullptr, ",")) sms.push_back(atoi(t));
    for (int s : sms) m[s >> 6] |= 1ull << (s & 63);
    const unsigned C = atoi(argv[2]), N = atoi(argv[3]); const double MB = argc > 4 ? atof(argv[4]) : 6.0;
    const size_t lines = (size_t)(MB * 1048576.0) / 128, words = lines * 32;
    std::vector<unsigned> perm(lines), h(words, 0); std::iota(perm.begin(), perm.end(), 0u);
    std::mt19937 g(2026); std::shuffle(perm.begin() + 1, perm.end(), g);
    for (size_t i = 0; i < lines; ++i) h[(size_t)perm[i] * 32] = perm[(i + 1) % lines] * 32;   // one cycle through all lines
    std::vector<unsigned> st(142 * 1024); for (auto &x : st) x = (unsigned)(g() % lines) * 32;
    unsigned *da, *dst, *sink; unsigned long long *dm, *dc;
    cudaMalloc(&da, words * 4); cudaMalloc(&dst, st.size() * 4); cudaMalloc(&sink, 4); cudaMalloc(&dm, 24); cudaMalloc(&dc, 142 * 1024 * 8);
    cudaMemcpy(da, h.data(), words * 4, cudaMemcpyHostToDevice); cudaMemcpy(dst, st.data(), st.size() * 4, cudaMemcpyHostToDevice); cudaMemcpy(dm, m, 24, cudaMemcpyHostToDevice);
    std::vector<unsigned long long> c(142 * 1024); double best_rate = 0; std::vector<double> all; std::vector<std::vector<double>> per_sm_all;
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (int rep = 0; rep < 4; ++rep) {   // rep 0 warms L2 (and the clocks); the others are reported
        cudaMemset(dc, 0, 142 * 1024 * 8);
        cudaEventRecord(e0); chase<<<142, 1024>>>(da, dst, dm, C, N, dc, sink); cudaEventRecord(e1);
        cudaError_t e = cudaDeviceSynchronize(); if (e != cudaSuccess) { fprintf(stderr, "%s\n", cudaGetErrorString(e)); return 1; }
        if (rep == 0) continue;
        cudaMemcpy(c.data(), dc, c.size() * 8, cudaMemcpyDeviceToHost);
        for (unsigned b = 0; b < 142; ++b) for (unsigned t = 0; t < 1024; ++t) if (c[b * 1024 + t]) all.push_back((double)c[b * 1024 + t] / N);
    }
    std::sort(all.begin(), all.end());
    const double med = all[all.size() / 2], p5 = all[all.size() / 20], p95 = all[all.size() * 19 / 20];
    // loads per us from the whole GPU: chasers x N loads / time of the slowest chaser (SM clock 2.5 GHz assumed via mean cycles)
    const double chasers = (double)sms.size() * (C >= 32 ? 32 * (C / 32) : 1), cyc_mean = std::accumulate(all.begin(), all.end(), 0.0) / all.size();
    printf("SMs %zu C %u N %u MB %g | cycles/load median %.1f p5 %.1f p95 %.1f | chasers %.0f | loads per cycle (all chasers) %.3f\n",
           sms.size(), C, N, MB, med, p5, p95, chasers, chasers / cyc_mean);
}
