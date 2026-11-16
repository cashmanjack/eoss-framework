// SM level instruction and memory costs for the callback term (plain CUDA, no OptiX, no TTU). One block of WARPS warps per SM, all SMs.
// Output: cycles per warp instruction per SM (clock64 window of the slowest warp of the SM), for
//   alu: independent FFMA   mem: independent loads, lane address pattern P (L1 tags per instruction) and working set W bytes.
// usage: cbcost alu | mem STRIDE_BYTES WSET_BYTES   (STRIDE: lane i reads base + i*STRIDE + a pseudo random line of the working set per iteration)
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
__global__ void alu(float* o, long long* cyc, int n) {
  float a = threadIdx.x, b = 1.0001f, c0 = 0.5f, c1 = 0.25f, c2 = 0.125f, c3 = 0.0625f;
  long long t0 = clock64();
  for (int i = 0; i < n; i++) { a = fmaf(a, b, c0); c0 = fmaf(c0, b, c1); c1 = fmaf(c1, b, c2); c2 = fmaf(c2, b, c3); }
  long long t1 = clock64(); o[blockIdx.x * blockDim.x + threadIdx.x] = a + c0 + c1 + c2;
  if ((threadIdx.x & 31) == 0) cyc[blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32] = t1 - t0;
}
__global__ void mem(const char* base, unsigned long long wset, unsigned stride, float* o, long long* cyc, int n) {
  unsigned lane = threadIdx.x & 31, w = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32; unsigned long long nl = wset / 128; float s = 0;
  unsigned long long h = 2654435761ull * (w + 1);
  long long t0 = clock64();
  for (int i = 0; i < n; i += 4) {
    float v[4];
    #pragma unroll
    for (int u = 0; u < 4; u++) {
      h = h * 2654435761ull + 40503ull; h ^= h >> 15;
      unsigned long long line = (h >> 9) & (nl - 1); const char* p = base + line * 128 + ((lane * stride) & 127);
      if (stride >= 128) p = base + (((h >> 9) + (unsigned long long)lane * 2654435761ull) & (nl - 1)) * 128;   // every lane its own line (wset a power of two)
      asm volatile("ld.global.ca.f32 %0, [%1];" : "=f"(v[u]) : "l"(p));
    }
    s += v[0] + v[1] + v[2] + v[3];
  }
  long long t1 = clock64(); o[blockIdx.x * blockDim.x + threadIdx.x] = s;
  if (lane == 0) cyc[w] = t1 - t0;
}
int main(int argc, char** argv) {
  int dev = 0; cudaSetDevice(dev); cudaDeviceProp pr; cudaGetDeviceProperties(&pr, dev); int nsm = pr.multiProcessorCount;
  int warps = argc > 4 ? atoi(argv[4]) : 32, n = 4096; float* o; long long* cyc; cudaMalloc(&o, (size_t)nsm * warps * 32 * 4); cudaMalloc(&cyc, (size_t)nsm * warps * 8);
  char* base = nullptr; unsigned long long wset = 0; unsigned stride = 0;
  bool isalu = argv[1][0] == 'a';
  if (!isalu) { stride = atoi(argv[2]); wset = strtoull(argv[3], 0, 10); cudaMalloc(&base, wset); cudaMemset(base, 1, wset); }
  for (int rep = 0; rep < 3; rep++) {
    cudaDeviceSynchronize();
    if (isalu) alu<<<nsm, warps * 32>>>(o, cyc, n); else mem<<<nsm, warps * 32>>>(base, wset, stride, o, cyc, n);
    cudaDeviceSynchronize();
  }
  long long* h = (long long*)malloc((size_t)nsm * warps * 8); cudaMemcpy(h, cyc, (size_t)nsm * warps * 8, cudaMemcpyDeviceToHost);
  double worst = 0; for (int b = 0; b < nsm; b++) { long long m = 0; for (int w = 0; w < warps; w++) if (h[b * warps + w] > m) m = h[b * warps + w]; worst += m; } worst /= nsm;
  double instr = (double)warps * (isalu ? 4.0 * n : (double)n);   // warp instructions of the measured kind per SM
  printf("%s stride %u wset %llu warps %d: SM cycles per warp instruction %.3f (window %.0f cycles, %.0f instructions)\n", argv[1], stride, wset, warps, worst / instr, worst, instr);
  return 0;
}
