// Pointer chase load generator that runs beside an OptiX launch (2026-11-17). Plain CUDA, own code.
// chase_start(blocks, nodes): chain of `nodes` nodes of 128 B (one u32 next index per node), one random cycle (Sattolo), blocks x 128 threads, each thread follows its own start.
// Per block, thread 0 publishes (steps, cycles) every 256 steps into pinned memory; the host snapshots them before and after the OptiX launch.
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <random>
#include <cstdint>
__global__ void chase_k(const unsigned* chain, unsigned nodes, volatile int* stop, volatile unsigned long long* st, volatile unsigned long long* cy) {
  unsigned p = (unsigned)(((unsigned long long)(blockIdx.x * blockDim.x + threadIdx.x) * 104729ull) % nodes);
  unsigned long long steps = 0, t0 = clock64();
  while (true) {
    #pragma unroll 1
    for (int i = 0; i < 256; i++) p = __ldcg(chain + (size_t)p * 32);
    steps += 256;
    if (threadIdx.x == 0) { st[blockIdx.x] = steps; cy[blockIdx.x] = clock64() - t0; }
    if (*stop) break;
  }
  if (p == 0xFFFFFFFFu) st[0] = 1;
}
static unsigned* d_chain = nullptr; static int* h_stop = nullptr; static unsigned long long *h_st = nullptr, *h_cy = nullptr; static int nb = 0; static cudaStream_t cs;
extern "C" void chase_start(int blocks, unsigned nodes) {
  nb = blocks; std::vector<unsigned> perm(nodes); for (unsigned i = 0; i < nodes; i++) perm[i] = i;
  std::mt19937_64 g(12345); for (unsigned i = nodes - 1; i > 0; i--) { unsigned j = g() % i; std::swap(perm[i], perm[j]); }   // Sattolo: one cycle
  std::vector<unsigned> h((size_t)nodes * 32, 0); for (unsigned i = 0; i < nodes; i++) h[(size_t)i * 32] = perm[i];
  cudaMalloc(&d_chain, h.size() * 4); cudaMemcpy(d_chain, h.data(), h.size() * 4, cudaMemcpyHostToDevice);
  cudaHostAlloc((void**)&h_stop, 4, cudaHostAllocMapped); cudaHostAlloc((void**)&h_st, 8 * blocks, cudaHostAllocMapped); cudaHostAlloc((void**)&h_cy, 8 * blocks, cudaHostAllocMapped);
  *h_stop = 0; for (int i = 0; i < blocks; i++) { h_st[i] = 0; h_cy[i] = 0; }
  int *d_stop; unsigned long long *d_st, *d_cy; cudaHostGetDevicePointer((void**)&d_stop, h_stop, 0); cudaHostGetDevicePointer((void**)&d_st, h_st, 0); cudaHostGetDevicePointer((void**)&d_cy, h_cy, 0);
  cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking);
  chase_k<<<blocks, 128, 0, cs>>>(d_chain, nodes, d_stop, d_st, d_cy);
}
// cumulative snapshot over blocks: total steps and total cycles of thread 0 of every block (cycles per step = cy/st)
extern "C" void chase_snap(double* steps, double* cycles) { double s = 0, c = 0; for (int i = 0; i < nb; i++) { s += (double)h_st[i]; c += (double)h_cy[i]; } *steps = s; *cycles = c; }
extern "C" void chase_stop() { *h_stop = 1; cudaStreamSynchronize(cs); }
