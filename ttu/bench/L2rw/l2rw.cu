// L2 read against write cost, plain CUDA (no TTU code). Predictions: model/predictions_4090_batch7.md part A.
// usage: l2rw FOOTPRINT_MB PATTERN WQ ITERS REPS   PATTERN 0 coal (warp touches 128 contiguous B, 4 B per lane), 1 lane4 (each lane its own 32 B sector, 4 B access), 2 lane32 (each lane its own sector, 32 B access)
// WQ = writes per 4 accesses (0..4). All accesses bypass L1 (ld.global.cg / st.global.cg), warp uniform choice, independent, addresses pseudo random, warp aligned, inside the footprint.
// prints one line: key=value ...   Build: nvcc -O3 -arch=sm_89 (or sm_120) l2rw.cu -o l2rw
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
__device__ __forceinline__ unsigned hsh(unsigned a, unsigned b) { unsigned x = a * 0x9E3779B1u ^ (b + 0x7F4A7C15u) * 0x85EBCA6Bu; x ^= x >> 15; x *= 0x2C1B3C6Du; x ^= x >> 12; x *= 0x297A2D39u; x ^= x >> 15; return x; }
template <int PAT> __global__ void k(char *buf, unsigned nunits, int iters, int wq, float *out) {
  unsigned lane = threadIdx.x & 31, wg = (blockIdx.x * blockDim.x + threadIdx.x) >> 5; float s = 0.f;
#pragma unroll 4
  for (int i = 0; i < iters; i++) {
    unsigned u = hsh(wg, i) % nunits; bool w = (i & 3) < wq;
    if (PAT == 0) { float *p = (float *)(buf + (size_t)u * 128) + lane; if (w) __stcg(p, 1.f + i); else s += __ldcg(p); }
    else if (PAT == 1) { float *p = (float *)(buf + (size_t)u * 1024 + lane * 32); if (w) __stcg(p, 1.f + i); else s += __ldcg(p); }
    else { float4 *p = (float4 *)(buf + (size_t)u * 1024 + lane * 32); if (w) { __stcg(p, make_float4(i, i, i, i)); __stcg(p + 1, make_float4(i, i, i, i)); } else { float4 a = __ldcg(p), b = __ldcg(p + 1); s += a.x + a.y + a.z + a.w + b.x + b.y + b.z + b.w; } }
  }
  if (s == 123456.f) out[0] = s;
}
int main(int c, char **v) {
  if (c < 6) { puts("usage"); return 2; }
  double mb = atof(v[1]); int pat = atoi(v[2]), wq = atoi(v[3]), iters = atoi(v[4]), reps = atoi(v[5]);
  size_t bytes = (size_t)(mb * 1048576); cudaDeviceProp P; cudaGetDeviceProperties(&P, 0);
  char *buf; float *out; cudaMalloc(&buf, bytes); cudaMalloc(&out, 64); cudaMemset(buf, 0, bytes);
  unsigned unit = pat == 0 ? 128 : 1024, nunits = bytes / unit; int blocks = P.multiProcessorCount * 8, thr = 256;
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b); double best[64]; int nr = 0;
  for (int r = -2; r < reps; r++) {
    cudaEventRecord(a);
    if (pat == 0) k<0><<<blocks, thr>>>(buf, nunits, iters, wq, out); else if (pat == 1) k<1><<<blocks, thr>>>(buf, nunits, iters, wq, out); else k<2><<<blocks, thr>>>(buf, nunits, iters, wq, out);
    cudaEventRecord(b); cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms, a, b); if (r >= 0) best[nr++] = ms;
  }
  for (int i = 0; i < nr; i++) for (int j = i + 1; j < nr; j++) if (best[j] < best[i]) { double t = best[i]; best[i] = best[j]; best[j] = t; }
  double ms = best[nr / 2]; double warps = (double)blocks * thr / 32, acc = warps * iters, spa = pat == 0 ? 4 : (pat == 1 ? 32 : 64); /* v2: lane32 issues two 16 B requests per access, 64 sectors per warp access (v1 counted 32, found from the ncu sector counts: ns_per_sector of lane32 in v1 files is 2x too high) */ double sect = acc * spa;
  cudaError_t e = cudaGetLastError();
  printf("v=2 gpu=\"%s\" sm=%d mb=%g pat=%d wq=%d iters=%d reps=%d ms_median=%.5f ms_min=%.5f ms_max=%.5f sectors=%.0f ns_per_sector=%.6f TBps=%.4f err=%d\n", P.name, P.multiProcessorCount, mb, pat, wq, iters, reps, ms, best[0], best[nr - 1], sect, ms * 1e6 / sect, sect * 32 / (ms * 1e-3) / 1e12, (int)e);
  return e != cudaSuccess;
}
