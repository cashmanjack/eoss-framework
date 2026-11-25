// Clock warmup for the baselines: keeps every SM busy for MS milliseconds (FMA spin), then exits. Own process, run to completion BEFORE a timed run.
// usage: warm MS
#include <cstdio>
#include <cstdlib>
__global__ void spin(float* o, long long cycles) {
  float a = threadIdx.x * 1e-3f + 1.f; long long t0 = clock64();
  while (clock64() - t0 < cycles) { for (int i = 0; i < 64; i++) a = fmaf(a, 1.0000001f, 1e-7f); }
  if (a == 12345.f) o[0] = a;
}
int main(int argc, char** argv) {
  int ms = argc > 1 ? atoi(argv[1]) : 2000; int dev = 0, sms = 0, clk = 0; cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev); cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, dev);
  float* d; cudaMalloc(&d, 4); spin<<<sms * 4, 256>>>(d, (long long)clk * ms); cudaError_t e = cudaDeviceSynchronize();
  printf("warm %d ms on %d SMs: %s\n", ms, sms, cudaGetErrorString(e)); return e != cudaSuccess;
}
