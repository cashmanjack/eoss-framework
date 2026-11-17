// DRAM bandwidth of one GPU with the traffic shape of a ray launch (2026-11-17, tethys GPU 1). Plain CUDA, no TTU code, no OptiX.
// copy: reads 8,388,608 x 32 B records (like the ray input) and writes 8,388,608 x 32 B (like the results), buffers 268 MB each (> 72 MB L2).
// read: reads the same 268 MB and writes one word per block. write: writes 268 MB. L2 is flushed by writing a 512 MB buffer before each timed launch.
// Prints median GB/s over 21 launches (bytes moved / cudaEvent time). Build: nvcc -O2 -arch=sm_89 model/bound_bw_bandwidth.cu -o OUT
#include <cstdio>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
__global__ void k_copy(const uint4* in, uint4* out, size_t n) { size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i < n) { out[2*i] = in[2*i]; out[2*i+1] = in[2*i+1]; } }
__global__ void k_read(const uint4* in, unsigned* out, size_t n) { size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; unsigned a = 0; if (i < n) { uint4 x = in[2*i], y = in[2*i+1]; a = x.x ^ x.y ^ x.z ^ x.w ^ y.x ^ y.y ^ y.z ^ y.w; } if (a == 0x12345678u) out[0] = a; }
__global__ void k_write(uint4* out, size_t n) { size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i < n) { out[2*i] = make_uint4(i, 1, 2, 3); out[2*i+1] = make_uint4(i, 4, 5, 6); } }
__global__ void k_flush(uint4* f, size_t n) { size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i < n) f[i] = make_uint4(i, i, i, i); }
int main() {
  const size_t n = 8388608; const int B = 256; uint4 *in, *out, *fl; unsigned* o2; size_t fn = 512ull << 20 >> 4;
  cudaMalloc(&in, n * 32); cudaMalloc(&out, n * 32); cudaMalloc(&fl, fn * 16); cudaMalloc(&o2, 4); cudaMemset(in, 1, n * 32); cudaMemset(out, 0, n * 32);
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b); int g = (n + B - 1) / B;
  const char* nm[3] = {"copy (268 MB in, 268 MB out)", "read only (268 MB)", "write only (268 MB)"}; double bytes[3] = {2.0 * n * 32, 1.0 * n * 32, 1.0 * n * 32};
  for (int m = 0; m < 3; m++) {
    std::vector<float> ts;
    for (int r = 0; r < 25; r++) {
      k_flush<<<(fn + 255) / 256, 256>>>(fl, fn); cudaDeviceSynchronize(); cudaEventRecord(a);
      if (m == 0) k_copy<<<g, B>>>(in, out, n); else if (m == 1) k_read<<<g, B>>>(in, o2, n); else k_write<<<g, B>>>(out, n);
      cudaEventRecord(b); cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms, a, b); if (r >= 4) ts.push_back(ms);
    }
    std::sort(ts.begin(), ts.end()); float med = ts[ts.size() / 2];
    printf("%-30s median %.4f ms  min %.4f  max %.4f  -> %.1f GB/s (median), %.1f GB/s (best)\n", nm[m], med, ts[0], ts.back(), bytes[m] / (med * 1e-3) / 1e9, bytes[m] / (ts[0] * 1e-3) / 1e9);
  }
  cudaError_t e = cudaGetLastError(); printf("cuda error: %s\n", cudaGetErrorString(e)); return e != cudaSuccess;
}
