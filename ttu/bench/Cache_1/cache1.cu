// Cache_1 / Cache_2 (plain CUDA, no TTU): fine grained pointer chase for L1 line, sector, capacity, associativity, replacement and the L2 (near / far, associativity).
// One thread of one block follows a dependent chain of m addresses at a byte stride S (sequential cyclic or random cyclic order); average cycles per load by clock64.
//   cache1 MODE       MODE: l1cap | l1sect | l1assoc | l1repl | l2assoc | l2lat     (text on stdout, wrapper run.py saves it)
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(2); } } while (0)
template <bool CG> __device__ __forceinline__ unsigned ld(const unsigned *p) { unsigned v; if (CG) asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(p)); else asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(v) : "l"(p)); return v; }
__global__ void setup(unsigned *a, const unsigned *pos, const unsigned *nxt, unsigned m) { for (unsigned k = 0; k < m; ++k) a[pos[k]] = nxt[k]; }
template <bool CG> __global__ void chase(const unsigned *a, unsigned start, unsigned warm, unsigned n, unsigned long long *cyc, unsigned *sink)
{
    unsigned j = start;
    for (unsigned i = 0; i < warm; ++i) j = ld<CG>(a + j);
    const unsigned long long t0 = clock64();
    for (unsigned i = 0; i < n; ++i) j = ld<CG>(a + j);
    cyc[0] = clock64() - t0; sink[0] = j;
}
template <bool CG> __global__ void each(const unsigned *a, const unsigned *order, unsigned m, unsigned warm_passes, unsigned *lat, unsigned *sink)
{   // timed single loads, one per element of order; the address depends on the sum of the loaded values (s >> 31 = 0 at run time) so loads do not overlap; the add that uses the loaded value is
    // asm volatile, so the second clock read issues after the data arrived (in order issue)
    unsigned s = 0;
    for (unsigned p = 0; p < warm_passes; ++p) for (unsigned k = 0; k < m; ++k) { const unsigned v = ld<CG>(a + order[k] + (s >> 31)); asm volatile("add.u32 %0, %0, %1;" : "+r"(s) : "r"(v)); }
    for (unsigned k = 0; k < m; ++k) {
        unsigned o = order[k]; asm volatile("add.u32 %0, %0, 0;" : "+r"(o));   // the order entry is fetched (and waited for) before the clock starts: the first run timed its 1 in 8 sector misses
        const unsigned long long t0 = clock64(); const unsigned v = ld<CG>(a + o + (s >> 31)); asm volatile("add.u32 %0, %0, %1;" : "+r"(s) : "r"(v));
        lat[k] = (unsigned)(clock64() - t0);
    }
    sink[0] = s;
}
__global__ void flush(const unsigned *a, size_t words, unsigned *sink) { unsigned s = 0; for (size_t i = 0; i < words; i += 32) s += ld<false>(a + i); sink[1] = s; }
static unsigned *dbuf, *dpos, *dnxt, *dsink; static unsigned long long *dcyc; static size_t BUFB = 320ull << 20;
static double run(unsigned m, unsigned long long S, bool random, bool cg, unsigned passes = 40)
{   // m addresses at byte stride S; returns cycles per access
    std::vector<unsigned> order(m); std::iota(order.begin(), order.end(), 0); if (random) { std::mt19937 g(1234 + m); std::shuffle(order.begin() + 1, order.end(), g); }
    std::vector<unsigned> pos(m), nxt(m); for (unsigned k = 0; k < m; ++k) pos[order[k]] = (unsigned)(order[k] * (S / 4));
    for (unsigned k = 0; k < m; ++k) { pos[k] = (unsigned)(order[k] * (S / 4)); nxt[k] = (unsigned)(order[(k + 1) % m] * (S / 4)); }
    CK(cudaMemcpy(dpos, pos.data(), 4 * m, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dnxt, nxt.data(), 4 * m, cudaMemcpyHostToDevice));
    setup<<<1, 1>>>(dbuf, dpos, dnxt, m); CK(cudaDeviceSynchronize());
    if (!cg) { flush<<<1, 1>>>(dbuf + (BUFB / 8) / 4, (size_t)(4 << 20) / 4, dsink); CK(cudaDeviceSynchronize()); }
    const unsigned n = std::max(20000u, m * passes), warm = std::max(2 * m, 4000u);
    if (cg) chase<true><<<1, 1>>>(dbuf, pos[0], warm, n, dcyc, dsink); else chase<false><<<1, 1>>>(dbuf, pos[0], warm, n, dcyc, dsink);
    CK(cudaDeviceSynchronize()); unsigned long long c; CK(cudaMemcpy(&c, dcyc, 8, cudaMemcpyDeviceToHost)); return (double)c / n;
}
int main(int argc, char **argv)
{
    const std::string mode = argc > 1 ? argv[1] : "l1cap"; CK(cudaSetDevice(0));
    CK(cudaMalloc(&dbuf, BUFB)); CK(cudaMemset(dbuf, 0, BUFB)); CK(cudaMalloc(&dpos, 4 << 20)); CK(cudaMalloc(&dnxt, 4 << 20)); CK(cudaMalloc(&dsink, 64)); CK(cudaMalloc(&dcyc, 64));
    cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0)); printf("device %s SMs %d L2 %d KB\n", pr.name, pr.multiProcessorCount, pr.l2CacheSize / 1024);
    if (mode == "l1cap") {   // capacity and tag line: footprint N at strides 32, 64, 128, 256 (sequential cyclic and random cyclic), .ca loads
        for (int rnd = 0; rnd < 2; ++rnd) for (unsigned S : {32u, 64u, 128u, 256u}) { printf("stride %u order %s\n", S, rnd ? "random" : "seq");
            for (unsigned kb = 8; kb <= 320; kb += (kb < 64 ? 8 : 4)) { const unsigned m = kb * 1024 / S; if (m < 2) continue; printf("  N_KB %u m %u cyc %.2f\n", kb, m, run(m, S, rnd, false)); } }
    } else if (mode == "l1sect") {   // stride sweep over a 1 MB array (3x L1, in L2): fraction of misses = min(1, S / sector)
        for (int rnd = 0; rnd < 2; ++rnd) for (unsigned S : {4u, 8u, 16u, 32u, 64u, 128u, 256u}) { const unsigned m = (1 << 20) / S; printf("stride %u order %s cyc %.2f\n", S, rnd ? "random" : "seq", run(m, S, rnd, false, 8)); }
    } else if (mode == "l1assoc") {   // m addresses at stride S = 128 2^j: the m where latency rises is the ways once S covers the set period
        for (int j = 0; j <= 13; ++j) { const unsigned long long S = 128ull << j; printf("S %llu\n", S);
            for (unsigned m : {1u, 2u, 3u, 4u, 5u, 6u, 8u, 10u, 12u, 14u, 16u, 20u, 24u, 32u, 48u, 64u, 96u, 128u}) { if (S * m > BUFB / 2) break; printf("  m %u cyc %.2f\n", m, run(m, S, false, false)); } }
    } else if (mode == "l1assoc2") {   // lines at a power of two stride up to the capacity (m 64 .. 1400): does the knee sit at the capacity (hashed set index) or at the ways
        for (unsigned long long S : {128ull, 1024ull, 4096ull, 16384ull, 65536ull, 131072ull}) { printf("S %llu\n", S);
            for (unsigned m = 600; m <= 1100; m += 50) { if (S * m > BUFB / 2) break; printf("  m %u cyc %.2f\n", m, run(m, S, false, false)); } }
    } else if (mode == "l1repl") {   // cyclic miss fraction just above the ways, sequential and random order, stride 128 * 2^j for the two largest conflict strides
        for (int j : {9, 11, 13}) { const unsigned long long S = 128ull << j; printf("S %llu\n", S);
            for (int rnd = 0; rnd < 2; ++rnd) for (unsigned m = 2; m <= 20; ++m) { if (S * m > BUFB / 2) break; printf("  order %s m %u cyc %.2f\n", rnd ? "random" : "seq", m, run(m, S, rnd, false, 200)); } }
    } else if (mode == "l2assoc") {   // L2: .cg loads, m addresses at stride S = 4096 .. 2 MB, m up to 160: latency rises at the ways (or at the cache size)
        for (int j = 5; j <= 14; ++j) { const unsigned long long S = 128ull << j; printf("S %llu\n", S);
            for (unsigned m : {1u, 2u, 4u, 8u, 12u, 16u, 20u, 24u, 32u, 48u, 64u, 96u, 128u, 160u}) { if (S * m > BUFB - (8ull << 20)) break; printf("  m %u cyc %.2f\n", m, run(m, S, false, true, 20)); } }
    } else if (mode == "l2lat") {   // per line L2 hit latency over 16 MB (warm, .cg), random order: histogram on the host
        const unsigned lines = 131072; std::vector<unsigned> order(lines); for (unsigned k = 0; k < lines; ++k) order[k] = k * 32; const int seed = argc > 2 ? atoi(argv[2]) : 99; if (seed >= 0) { std::mt19937 g(seed); std::shuffle(order.begin(), order.end(), g); }   // seed -1: ascending address order
        unsigned *dord, *dlat; CK(cudaMalloc(&dord, 4 * lines)); CK(cudaMalloc(&dlat, 4 * lines)); CK(cudaMemcpy(dord, order.data(), 4 * lines, cudaMemcpyHostToDevice));
        each<true><<<1, 1>>>(dbuf, dord, lines, 1, dlat, dsink); CK(cudaDeviceSynchronize());   // pass 1 warms the L2 (timed loop of the same kernel run again below)
        each<true><<<1, 1>>>(dbuf, dord, lines, 1, dlat, dsink); CK(cudaDeviceSynchronize());
        std::vector<unsigned> lat(lines); CK(cudaMemcpy(lat.data(), dlat, 4 * lines, cudaMemcpyDeviceToHost));
        for (unsigned k = 0; k < lines; ++k) printf("lat %u %u\n", order[k] / 32, lat[k]);
    }
    return 0;
}
