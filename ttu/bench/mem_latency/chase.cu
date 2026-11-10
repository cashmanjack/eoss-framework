// Pointer chase: load to use latency of L1, L2 and DRAM on one SM (plain CUDA, no TTU).
//
//   chase SIZE_KB MODE OUT.json
//     MODE  ca     warm up and time in the same kernel, loads may use L1 (ld.global.ca)
//           cg     same, loads bypass L1 (ld.global.cg)
//           cross  warm up in one launch (ca), time in the next launch without warm up (ca)
//           cold   read 256 MB, then time without warm up (ca)
//
// One thread follows a random cycle over the array, one element per 128 byte line,
// so every load depends on the previous one and prefetching cannot help.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(2); } } while (0)

template <bool CG>
__device__ __forceinline__ unsigned ld(const unsigned *p)
{
    unsigned v;
    if (CG) asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(p));
    else    asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
}

template <bool CG>
__global__ void chase(const unsigned *a, unsigned start, unsigned warm, unsigned n, unsigned long long *cyc, unsigned *sink)
{
    unsigned j = start;
    for (unsigned i = 0; i < warm; ++i) j = ld<CG>(a + j);
    const unsigned long long t0 = clock64();
    for (unsigned i = 0; i < n; ++i) j = ld<CG>(a + j);
    const unsigned long long t1 = clock64();
    cyc[0] = t1 - t0;
    sink[0] = j;   // keeps the chain live; also the next start
}

__global__ void flush_read(const uint4 *p, size_t n, unsigned *sink)
{
    unsigned acc = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        const uint4 v = p[i]; acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x12345678u) sink[0] = acc;
}

int main(int argc, char **argv)
{
    if (argc < 4) { fprintf(stderr, "usage: chase SIZE_KB ca|cg|cross|cold OUT.json\n"); return 2; }
    const size_t kb = strtoull(argv[1], 0, 10); const char *mode = argv[2];
    const size_t words = kb * 1024 / 4, lines = words / 32;
    if (lines < 2 || kb > 4u << 20) { fprintf(stderr, "bad size\n"); return 2; }
    // random cycle over lines; element 0 of each line holds the index of the next line's element 0
    std::vector<unsigned> perm(lines); std::iota(perm.begin(), perm.end(), 0u);
    std::mt19937 rng(7); std::shuffle(perm.begin() + 1, perm.end(), rng);
    std::vector<unsigned> h(words, 0);
    for (size_t i = 0; i < lines; ++i) h[(size_t)perm[i] * 32] = perm[(i + 1) % lines] * 32;
    unsigned *a, *sink; unsigned long long *cyc;
    CK(cudaMalloc(&a, words * 4)); CK(cudaMalloc(&sink, 4)); CK(cudaMalloc(&cyc, 8));
    CK(cudaMemcpy(a, h.data(), words * 4, cudaMemcpyHostToDevice));
    const unsigned n = (unsigned)std::min<size_t>(lines, 20000);   // timed loads per launch
    const unsigned warm = (unsigned)std::min<size_t>(2 * lines, 4000000);
    const size_t fn = (256u << 20) / sizeof(uint4); uint4 *fbuf = nullptr;
    const bool cold = !strcmp(mode, "cold"), cross = !strcmp(mode, "cross"), cg = !strcmp(mode, "cg");
    if (!cold && !cross && !cg && strcmp(mode, "ca")) { fprintf(stderr, "bad mode\n"); return 2; }
    if (cold) { CK(cudaMalloc(&fbuf, fn * sizeof(uint4))); CK(cudaMemset(fbuf, 1, fn * sizeof(uint4))); }
    FILE *jf = fopen(argv[3], "w");
    fprintf(jf, "{\"size_kb\": %zu, \"mode\": \"%s\", \"lines\": %zu, \"timed_loads\": %u, \"cycles_per_load\": [", kb, mode, lines, n);
    for (int rep = 0; rep < 11; ++rep) {
        unsigned start = 0;
        if (cross) { chase<false><<<1, 1>>>(a, 0, 0, warm, cyc, sink); CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); }
        if (cold) { flush_read<<<142 * 8, 256>>>(fbuf, fn, sink); CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); }
        const unsigned w = (cross || cold) ? 0 : warm;
        if (cg) chase<true><<<1, 1>>>(a, start, w, n, cyc, sink);
        else    chase<false><<<1, 1>>>(a, start, w, n, cyc, sink);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        unsigned long long c; CK(cudaMemcpy(&c, cyc, 8, cudaMemcpyDeviceToHost));
        fprintf(jf, "%s%.2f", rep ? ", " : "", (double)c / n);
    }
    fprintf(jf, "]}\n"); fclose(jf);
    return 0;
}
