// L2 throughput for read only, write only and mixed streams (plain CUDA, no TTU, no OptiX). Predictions: model/predictions_5090_batch7.md part 2.
// build: nvcc -O3 -arch=sm_120 -o l2stream l2stream.cu       run: ./l2stream            (all modes and footprints)   or   ./l2stream MODE MB   (one config, for ncu: kernels stream_<mode> are the timed ones, warm_<mode> the warm up)
// MODE 0 read only (A = F), 1 write only (B = F), 2 copy 1:1 (read A = F/2, write B = F/2), 3 two reads one write (A = 2F/3, B = F/3).
// 16 byte vector accesses with L1 bypass (ld.global.cg / st.global.cg); a warp covers 512 contiguous bytes (whole sectors). Grid: 3 blocks of 512 threads per SM.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "CUDA error %s line %d\n", cudaGetErrorString(e), __LINE__); exit(2); } } while (0)
__device__ __forceinline__ unsigned long long gtimer() { unsigned long long t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }
template <int MODE> __device__ __forceinline__ void body(const uint4* A, uint4* B, size_t n, int passes, unsigned long long* tm, unsigned* sink) {
  unsigned long long c0 = clock64(), t0 = gtimer(); size_t g = (size_t)blockIdx.x * blockDim.x + threadIdx.x, G = (size_t)gridDim.x * blockDim.x; unsigned acc = 0;
  for (int p = 0; p < passes; ++p) {
    #pragma unroll 4
    for (size_t i = g; i < n; i += G) {
      if (MODE == 0) { uint4 v = __ldcg(A + i); acc ^= v.x ^ v.y ^ v.z ^ v.w; }
      else if (MODE == 1) { __stcg(B + i, make_uint4((unsigned)i, p, (unsigned)i + 1, p + 1)); }
      else if (MODE == 2) { uint4 v = __ldcg(A + i); __stcg(B + i, make_uint4(v.x + p, v.y, v.z, v.w)); }
      else { uint4 v = __ldcg(A + i), u = __ldcg(A + n + i); __stcg(B + i, make_uint4(v.x + u.x + p, v.y ^ u.y, v.z, v.w)); }
    }
  }
  if (acc == 0x9e3779b9u) sink[g] = acc;
  __syncthreads();
  if (threadIdx.x == 0) { tm[blockIdx.x * 2] = clock64() - c0; tm[blockIdx.x * 2 + 1] = gtimer() - t0; }
}
template <int MODE> __global__ void stream(const uint4* A, uint4* B, size_t n, int passes, unsigned long long* tm, unsigned* sink) { body<MODE>(A, B, n, passes, tm, sink); }
template <int MODE> __global__ void warm(const uint4* A, uint4* B, size_t n, int passes, unsigned long long* tm, unsigned* sink) { body<MODE>(A, B, n, passes, tm, sink); }
typedef void (*K)(const uint4*, uint4*, size_t, int, unsigned long long*, unsigned*);
int main(int argc, char** argv) {
  int dev = 0; cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, dev)); int nsm = pr.multiProcessorCount; int blocks = nsm * 3, thr = 512;
  K ks[4] = {stream<0>, stream<1>, stream<2>, stream<3>}, kw[4] = {warm<0>, warm<1>, warm<2>, warm<3>};
  const char* nm[4] = {"read", "write", "copy1to1", "read2write1"};
  std::vector<double> mbs = {2, 8, 24, 48, 72, 256}; std::vector<int> modes = {0, 1, 2, 3};
  if (argc == 3) { modes = {atoi(argv[1])}; mbs = {atof(argv[2])}; }
  size_t maxb = (size_t)300 << 20; void *dA, *dB; CK(cudaMalloc(&dA, maxb)); CK(cudaMalloc(&dB, maxb)); CK(cudaMemset(dA, 1, maxb)); CK(cudaMemset(dB, 2, maxb));
  unsigned long long* dtm; unsigned* dsink; CK(cudaMalloc(&dtm, blocks * 16)); CK(cudaMalloc(&dsink, (size_t)blocks * thr * 4));
  printf("l2stream SMs %d blocks %d x %d threads\nmode,footprint_MB,passes,ms_median,ms_min,GHz,read_sectors_per_launch,write_sectors_per_launch,sectors_per_cycle,TB_per_s_requested\n", nsm, blocks, thr);
  for (double mb : mbs) for (int m : modes) {
    size_t F = (size_t)(mb * 1048576.0), n, rb, wb;   // n = number of 16 B elements per stream element group
    if (m == 0) { n = F / 16; rb = F; wb = 0; } else if (m == 1) { n = F / 16; rb = 0; wb = F; } else if (m == 2) { n = F / 32; rb = F / 2; wb = F / 2; } else { n = F / 48; rb = F / 3 * 2; wb = F / 3; }
    int passes = (int)std::max<double>(2, 6e9 / (double)(rb + wb)); passes = std::min(passes, 4000);
    kw[m]<<<blocks, thr>>>((const uint4*)dA, (uint4*)dB, n, 1, dtm, dsink); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    std::vector<float> ms; cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1); double ghz = 0;
    for (int r = 0; r < 6; ++r) {
      cudaEventRecord(e0); ks[m]<<<blocks, thr>>>((const uint4*)dA, (uint4*)dB, n, passes, dtm, dsink); cudaEventRecord(e1); CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());
      float t; cudaEventElapsedTime(&t, e0, e1); if (r > 0) ms.push_back(t);
      std::vector<unsigned long long> tm(blocks * 2); CK(cudaMemcpy(tm.data(), dtm, blocks * 16, cudaMemcpyDeviceToHost)); double c = 0, ns = 0; for (int b = 0; b < blocks; ++b) { c += tm[2 * b]; ns += tm[2 * b + 1]; } ghz = c / ns;
    }
    std::sort(ms.begin(), ms.end()); double med = ms[ms.size() / 2]; double rs = (double)rb / 32 * passes, ws = (double)wb / 32 * passes, cyc = med * 1e-3 * ghz * 1e9;
    printf("%s,%.0f,%d,%.4f,%.4f,%.3f,%.0f,%.0f,%.2f,%.3f\n", nm[m], mb, passes, med, ms[0], ghz, rs, ws, (rs + ws) / cyc, (rs + ws) * 32 / (med * 1e-3) / 1e12);
    fflush(stdout);
  }
  return 0;
}
