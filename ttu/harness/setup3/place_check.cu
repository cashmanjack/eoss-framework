// q32 plain CUDA placement check (no TTU, no OptiX). One plain launch of NSM blocks x 128 threads; the block that lands on SM `target` stays resident (role), the others exit.
// Then a stand-in for the TTU kernel (a grid of NSM blocks x 32 threads, default stream, as the smfilter builds: the block on the wanted SM is the one that works) is launched; the last block releases the role block. 20 trials per target.
// Question: does the stand-in land on SM 0 when the role block is on SM 0 (own SM test) or on SM 1 (sibling test)? Every wait has a device side timeout (500 ms): nothing can hang.
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <cuda_runtime.h>
__device__ unsigned smid_() { unsigned s; asm volatile("mov.u32 %0, %%smid;" : "=r"(s)); return s; }
__device__ unsigned long long gt_() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__global__ void role(unsigned target, volatile unsigned *f, volatile unsigned *info)
{
    __shared__ int go;
    if (threadIdx.x == 0) { go = 0; if (smid_() == target && atomicAdd((unsigned *)&f[64], 1u) == 0) go = 1; }
    __syncthreads(); if (!go) return;
    if (threadIdx.x == 0) { info[0] = smid_(); info[1] = 1; const unsigned long long t = gt_(); while (!f[0] && gt_() - t < 500000000ull) { } info[2] = f[0] ? 1 : 2; }
    __syncthreads();
}
__global__ void __launch_bounds__(32) standin(volatile unsigned *f, volatile unsigned *sm, unsigned n) { if (threadIdx.x == 0) { sm[blockIdx.x] = smid_(); __threadfence(); if (atomicAdd((unsigned *)&f[65], 1u) == n - 1) f[0] = 1; } }
int main()
{
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0); const int nsm = p.multiProcessorCount; printf("device %s, %d SMs\n", p.name, nsm);
    unsigned *f, *info, *sms; cudaMallocManaged((void **)&f, 4096); cudaMallocHost((void **)&info, 64); cudaMallocHost((void **)&sms, 4 * nsm); cudaStream_t s; cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    for (unsigned target : {0u, 1u, 2u}) {
        int same = 0, ok = 0, ntr = 20, on0 = 0; unsigned seen[4] = {0, 0, 0, 0};
        for (int t = 0; t < ntr; ++t) {
            cudaMemset(f, 0, 4096); cudaDeviceSynchronize(); for (int k = 0; k < 16; ++k) info[k] = 0;
            role<<<nsm, 128, 0, s>>>(target, f, info);
            for (int w = 0; w < 20000 && !info[1]; ++w) usleep(50);
            if (!info[1]) { printf("target %u trial %d: role block did not land\n", target, t); cudaStreamSynchronize(s); continue; }
            for (int k = 0; k < nsm; ++k) sms[k] = 9999; usleep(2000); standin<<<nsm, 32>>>(f, sms, nsm); cudaError_t e = cudaDeviceSynchronize(); cudaStreamSynchronize(s);
            if (e != cudaSuccess) { printf("error %s\n", cudaGetErrorString(e)); return 1; }
            ++ok; int on = 0; for (int k = 0; k < nsm; ++k) if (sms[k] == 0) { ++on0; break; } for (int k = 0; k < nsm; ++k) if (sms[k] == info[0]) ++on; if (on >= 1) ++same; seen[on < 3 ? on : 3]++;
        }
        printf("role on SM %u: a stand-in block landed beside it in %d of %d trials (blocks on that SM: 0:%u 1:%u 2:%u 3+:%u); a stand-in block on SM 0: %d of %d\n", target, same, ok, seen[0], seen[1], seen[2], seen[3], on0, ok);
    }
    return 0;
}
