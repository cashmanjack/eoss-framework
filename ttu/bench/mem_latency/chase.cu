// Pointer chase: load to use latency of L1, L2 and DRAM on one SM (plain CUDA, no TTU).
//
//   chase SIZE_KB MODE OUT.json
//     MODE  ca     warm up and time in the same kernel, loads may use L1 (ld.global.ca)
//           cg     same, loads bypass L1 (ld.global.cg)
//           cross  warm up in one launch (ca), time in the next launch without warm up (ca)
//           cold   read 256 MB, then time without warm up (ca)
//           thrash:K  one warp: lane 0 warms the chase, then every lane does K L1 allocating loads at
//                     buf + 128 lane + 4096 k (the Setup_3 --thrash pattern), then lane 0 times the chase
//           evict:KB  warm the chase, then walk a second random chain over KB kilobytes twice, then time
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

__global__ void chase_thrash(const unsigned *a, unsigned warm, unsigned n, const unsigned char *buf, unsigned k_loads,
                             unsigned long long *cyc, unsigned *sink)
{
    unsigned j = 0, acc = 0;
    if (threadIdx.x == 0) for (unsigned i = 0; i < warm; ++i) j = ld<false>(a + j);
    acc = j;                      // keep the warm up live
    __syncwarp();
    for (unsigned k = 0; k < k_loads; ++k) acc += ld<false>((const unsigned *)(buf + 128 * threadIdx.x + 4096ull * k));
    __syncwarp();
    if (threadIdx.x == 0) {
        j = 0;
        const unsigned long long t0 = clock64();
        for (unsigned i = 0; i < n; ++i) j = ld<false>(a + j);
        const unsigned long long t1 = clock64();
        cyc[0] = t1 - t0; sink[0] = j + acc;
    }
}

__global__ void chase_evict(const unsigned *a, const unsigned *b, unsigned warm, unsigned bwalk, unsigned n,
                            unsigned long long *cyc, unsigned *sink)
{
    unsigned j = 0, k = 0;
    for (unsigned i = 0; i < warm; ++i) j = ld<false>(a + j);
    k = j & 0;                    // warm up result live, k still starts at 0
    for (unsigned i = 0; i < bwalk; ++i) k = ld<false>(b + k);
    sink[1] = j;                  // keep the warm up
    j = 0;
    const unsigned long long t0 = clock64();
    for (unsigned i = 0; i < n; ++i) j = ld<false>(a + j);
    const unsigned long long t1 = clock64();
    cyc[0] = t1 - t0; sink[0] = j + k;
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
    if (argc < 4) { fprintf(stderr, "usage: chase SIZE_KB ca|cg|cross|cold OUT.json [SMEM_BYTES]\n"); return 2; }
    // SMEM_BYTES (ca mode): dynamic shared memory reserved by the launch (unused), which shrinks L1 (Memory_1 carveout)
    const int smem = argc > 4 ? atoi(argv[4]) : 0;
    if (smem) CK(cudaFuncSetAttribute(chase<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    const size_t kb = strtoull(argv[1], 0, 10); const char *mode = argv[2];
    const size_t words = kb * 1024 / 4, lines = words / 32;
    if (lines < 2 || kb > 4u << 20) { fprintf(stderr, "bad size\n"); return 2; }
    // random cycle over lines; element 0 of each line holds the index of the next line's element 0
    std::vector<unsigned> perm(lines); std::iota(perm.begin(), perm.end(), 0u);
    std::mt19937 rng(7); std::shuffle(perm.begin() + 1, perm.end(), rng);
    std::vector<unsigned> h(words, 0);
    for (size_t i = 0; i < lines; ++i) h[(size_t)perm[i] * 32] = perm[(i + 1) % lines] * 32;
    int nsm = 0; { int dv = 0; CK(cudaGetDevice(&dv)); CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dv)); }
    unsigned *a, *sink; unsigned long long *cyc;
    CK(cudaMalloc(&a, words * 4)); CK(cudaMalloc(&sink, 8)); CK(cudaMalloc(&cyc, 8));
    CK(cudaMemcpy(a, h.data(), words * 4, cudaMemcpyHostToDevice));
    const unsigned n = (unsigned)std::min<size_t>(lines, 20000);   // timed loads per launch
    const unsigned warm = (unsigned)std::min<size_t>(2 * lines, 4000000);
    const size_t fn = (256u << 20) / sizeof(uint4); uint4 *fbuf = nullptr;
    const bool cold = !strcmp(mode, "cold"), cross = !strcmp(mode, "cross"), cg = !strcmp(mode, "cg");
    const bool thr = !strncmp(mode, "thrash:", 7); const unsigned kl = thr ? atoi(mode + 7) : 0;
    const bool ev = !strncmp(mode, "evict:", 6); const size_t ekb = ev ? strtoull(mode + 6, 0, 10) : 0;
    if (!cold && !cross && !cg && !thr && !ev && strcmp(mode, "ca")) { fprintf(stderr, "bad mode\n"); return 2; }
    unsigned *eb = nullptr; unsigned ewalk = 0;
    if (ev) {   // second random chain over ekb KB, one element per 128 byte line
        const size_t ew = ekb * 1024 / 4, el = ew / 32;
        std::vector<unsigned> pe(el); std::iota(pe.begin(), pe.end(), 0u); std::shuffle(pe.begin() + 1, pe.end(), rng);
        std::vector<unsigned> he(ew, 0);
        for (size_t i = 0; i < el; ++i) he[(size_t)pe[i] * 32] = pe[(i + 1) % el] * 32;
        CK(cudaMalloc(&eb, ew * 4)); CK(cudaMemcpy(eb, he.data(), ew * 4, cudaMemcpyHostToDevice));
        ewalk = (unsigned)(2 * el);
    }
    unsigned char *tb = nullptr;
    if (thr) { if (kl > 1024) { fprintf(stderr, "K <= 1024\n"); return 2; } CK(cudaMalloc(&tb, 4u << 20)); CK(cudaMemset(tb, 0, 4u << 20)); }
    if (cold) { CK(cudaMalloc(&fbuf, fn * sizeof(uint4))); CK(cudaMemset(fbuf, 1, fn * sizeof(uint4))); }
    FILE *jf = fopen(argv[3], "w");
    fprintf(jf, "{\"smem\": %d, \"size_kb\": %zu, \"mode\": \"%s\", \"lines\": %zu, \"timed_loads\": %u, \"cycles_per_load\": [", smem, kb, mode, lines, n);
    for (int rep = 0; rep < 11; ++rep) {
        unsigned start = 0;
        if (cross) { chase<false><<<1, 1>>>(a, 0, 0, warm, cyc, sink); CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); }
        if (cold) { flush_read<<<nsm * 8, 256>>>(fbuf, fn, sink); CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); }
        const unsigned w = (cross || cold) ? 0 : warm;
        if (thr) chase_thrash<<<1, 32>>>(a, warm, n, tb, kl, cyc, sink);
        else if (ev) chase_evict<<<1, 1>>>(a, eb, warm, ewalk, n, cyc, sink);
        else if (cg) chase<true><<<1, 1>>>(a, start, w, n, cyc, sink);
        else    chase<false><<<1, 1, smem>>>(a, start, w, n, cyc, sink);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        unsigned long long c; CK(cudaMemcpy(&c, cyc, 8, cudaMemcpyDeviceToHost));
        fprintf(jf, "%s%.2f", rep ? ", " : "", (double)c / n);
    }
    fprintf(jf, "]}\n"); fclose(jf);
    return 0;
}
