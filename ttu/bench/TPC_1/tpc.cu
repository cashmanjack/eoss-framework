// TPC_1 (OptiX only, no TTU code): OptiX random ray traces on whatever SMs are left, while a persistent plain CUDA compute kernel owns SMs by parity.
//   tpc PTX SOUP_N RAYS.bin LAUNCHES PLACE TYPE [WINDOW_MS]
//     PLACE: none (no compute kernel) | all (compute blocks on every SM, 2 per SM) | odd (compute only on odd numbered SMs, 4 blocks each, filling their registers so OptiX cannot land there)
//     TYPE : alu (FMA chain, no memory) | l1 (reads a private 32 KB region over and over: L1 pressure) | mem (streams a 1 GB buffer: L2 / DRAM pressure)
//     LAUNCHES: optix launches in the timed window (0 = no RT; the window is then WINDOW_MS long, default 100)
// Prints one JSON line: per launch ms of the OptiX launches, compute iterations per second (device clock based window = host wall time), blocks per SM that stayed (compute), SM of every
// resident compute block counted by parity.
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include "tpc_params.h"
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(2); } } while (0)
#define OX(x) do { OptixResult r_ = (x); if (r_ != OPTIX_SUCCESS) { fprintf(stderr, "%s:%d %s: %d\n", __FILE__, __LINE__, #x, (int)r_); exit(2); } } while (0)
struct Tri { float3 a, b, c; };
static std::string read_file(const char *p) { std::ifstream f(p); if (!f) { fprintf(stderr, "cannot read %s\n", p); exit(2); } std::stringstream s; s << f.rdbuf(); return s.str(); }

__device__ static unsigned smid() { unsigned s; asm volatile("mov.u32 %0, %%smid;" : "=r"(s)); return s; }
// persistent compute kernel. parity: -1 = any SM, 0/1 = only SMs with that parity (blocks elsewhere exit at once).
template <int TYPE>
__device__ void compute(int parity, const volatile int *stop, unsigned long long *iters, unsigned long long *resident, const uint4 *buf, size_t nbuf)
{
    __shared__ int ok; __shared__ unsigned my_sm;
    if (threadIdx.x == 0) { const unsigned s = smid(); my_sm = s; ok = parity < 0 || (int)(s & 1) == parity; if (ok) atomicAdd(&resident[s], 1ull); }
    __syncthreads(); if (!ok) return;
    float x = 1.f + threadIdx.x * 1e-3f, y = 0.5f; unsigned long long it = 0; uint4 acc = make_uint4(0, 0, 0, 0);
    const size_t base = ((size_t)blockIdx.x * 2048) % (nbuf > 4096 ? nbuf - 2048 : 1);   // l1: 2048 uint4 = 32 KB per block
    size_t pos = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 7919 % (nbuf ? nbuf : 1);
    while (!*stop) {
        if (TYPE == 0) { for (int k = 0; k < 1024; ++k) { x = fmaf(x, 1.0000001f, y); y = fmaf(y, 0.9999999f, x); } }
        else if (TYPE == 1) { for (int k = 0; k < 32; ++k) { const uint4 v = buf[base + ((threadIdx.x + 256 * k) & 2047)]; acc.x += v.x; acc.y ^= v.y; acc.z += v.z; acc.w ^= v.w; } }
        else { for (int k = 0; k < 32; ++k) { const uint4 v = buf[pos]; pos += 1024 * 256 + 1; if (pos >= nbuf) pos -= nbuf; acc.x += v.x; acc.y ^= v.y; acc.z += v.z; acc.w ^= v.w; } }
        ++it; if (threadIdx.x == 0) atomicAdd(&iters[my_sm], 1ull);   // one block iteration, counted per SM slot (sampled by the host during the window)
    }
    if (x == 12345.f || acc.x == 0xdeadbeef) iters[1023] += 1;   // keep the work alive
}
// 384 thread blocks: 4 per SM use all 1536 resident threads (nothing else fits: OptiX cannot land there), 2 per SM (all) leave half
template <int TYPE> __global__ void __launch_bounds__(384) compute4(int p, const volatile int *s, unsigned long long *i, unsigned long long *r, const uint4 *b, size_t n) { compute<TYPE>(p, s, i, r, b, n); }
// force 64 registers with a big unused array? Instead the host pads with dynamic shared memory (24 KB x 4 = 96 KB) so only 4 blocks fit.

int main(int argc, char **argv)
{
    if (argc < 7) { fprintf(stderr, "usage\n"); return 2; }
    const int n_tri = atoi(argv[2]), launches = atoi(argv[4]); const std::string place = argv[5], type = argv[6]; const int window_ms = argc > 7 ? atoi(argv[7]) : 100;
    std::vector<Ray8> rays; { FILE *f = fopen(argv[3], "rb"); Ray8 y; while (f && fread(&y, sizeof y, 1, f) == 1) rays.push_back(y); if (f) fclose(f); }
    const unsigned NR = (unsigned)rays.size();
    CK(cudaFree(0)); OX(optixInit()); OptixDeviceContextOptions co{}; OptixDeviceContext ctx; OX(optixDeviceContextCreate(0, &co, &ctx));
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0)); const int NSM = prop.multiProcessorCount;
    const float e = 3.f / std::cbrt((float)n_tri); std::mt19937 g(1234); std::uniform_real_distribution<float> U(-1.f, 1.f);
    std::vector<Tri> v; for (int i = 0; i < n_tri; ++i) { const float3 c{U(g), U(g), U(g)}; auto pt = [&]() { return float3{c.x + 0.5f * e * U(g), c.y + 0.5f * e * U(g), c.z + 0.5f * e * U(g)}; }; v.push_back({pt(), pt(), pt()}); }
    CUdeviceptr dv; CK(cudaMalloc((void **)&dv, v.size() * sizeof(Tri))); CK(cudaMemcpy((void *)dv, v.data(), v.size() * sizeof(Tri), cudaMemcpyHostToDevice));
    unsigned gflags = OPTIX_GEOMETRY_FLAG_NONE; OptixBuildInput bi{}; bi.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    bi.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3; bi.triangleArray.numVertices = (unsigned)v.size() * 3; bi.triangleArray.vertexBuffers = &dv; bi.triangleArray.flags = &gflags; bi.triangleArray.numSbtRecords = 1;
    OptixAccelBuildOptions ao{}; ao.buildFlags = OPTIX_BUILD_FLAG_NONE; ao.operation = OPTIX_BUILD_OPERATION_BUILD;
    OptixAccelBufferSizes bs{}; OX(optixAccelComputeMemoryUsage(ctx, &ao, &bi, 1, &bs));
    CUdeviceptr tmp, gas; CK(cudaMalloc((void **)&tmp, bs.tempSizeInBytes)); CK(cudaMalloc((void **)&gas, bs.outputSizeInBytes));
    OptixTraversableHandle h; OX(optixAccelBuild(ctx, 0, &ao, &bi, 1, tmp, bs.tempSizeInBytes, gas, bs.outputSizeInBytes, &h, nullptr, 0)); CK(cudaDeviceSynchronize()); CK(cudaFree((void *)tmp));
    OptixModuleCompileOptions mo{}; OptixPipelineCompileOptions po{}; po.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS; po.numPayloadValues = 7; po.numAttributeValues = 2;
    po.pipelineLaunchParamsVariableName = "params"; po.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
    std::string ptx = read_file(argv[1]); char log[4096]; size_t ls = sizeof log; OptixModule m; OX(optixModuleCreate(ctx, &mo, &po, ptx.c_str(), ptx.size(), log, &ls, &m));
    OptixProgramGroupOptions go{}; OptixProgramGroupDesc d[3] = {};
    d[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN; d[0].raygen.module = m; d[0].raygen.entryFunctionName = "__raygen__plain";
    d[1].kind = OPTIX_PROGRAM_GROUP_KIND_MISS; d[1].miss.module = m; d[1].miss.entryFunctionName = "__miss__ms";
    d[2].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP; d[2].hitgroup.moduleCH = m; d[2].hitgroup.entryFunctionNameCH = "__closesthit__ch";
    OptixProgramGroup pg[3]; ls = sizeof log; OX(optixProgramGroupCreate(ctx, d, 3, &go, log, &ls, pg));
    OptixPipelineLinkOptions lo{}; lo.maxTraceDepth = 1; OptixPipeline pipe; ls = sizeof log; OX(optixPipelineCreate(ctx, &po, &lo, pg, 3, log, &ls, &pipe));
    struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) Rec { char h[OPTIX_SBT_RECORD_HEADER_SIZE]; };
    Rec rec[3]; for (int k = 0; k < 3; ++k) OX(optixSbtRecordPackHeader(pg[k], &rec[k]));
    CUdeviceptr drec; CK(cudaMalloc((void **)&drec, sizeof rec)); CK(cudaMemcpy((void *)drec, rec, sizeof rec, cudaMemcpyHostToDevice));
    OptixShaderBindingTable sbt{}; sbt.raygenRecord = drec; sbt.missRecordBase = drec + sizeof(Rec); sbt.missRecordStrideInBytes = sizeof(Rec); sbt.missRecordCount = 1;
    sbt.hitgroupRecordBase = drec + 2 * sizeof(Rec); sbt.hitgroupRecordStrideInBytes = sizeof(Rec); sbt.hitgroupRecordCount = 1;
    Params prm{}; prm.handle = h; prm.n = NR; prm.bits = 0;
    if (NR) { CK(cudaMalloc((void **)&prm.rays, NR * sizeof(Ray8))); CK(cudaMemcpy((void *)prm.rays, rays.data(), NR * sizeof(Ray8), cudaMemcpyHostToDevice)); CK(cudaMalloc((void **)&prm.out, NR * sizeof(RefOut))); }
    CUdeviceptr dprm; CK(cudaMalloc((void **)&dprm, sizeof prm)); CK(cudaMemcpy((void *)dprm, &prm, sizeof prm, cudaMemcpyHostToDevice));
    cudaStream_t sr, sc, ss; CK(cudaStreamCreateWithFlags(&sr, cudaStreamNonBlocking)); CK(cudaStreamCreateWithFlags(&sc, cudaStreamNonBlocking)); CK(cudaStreamCreateWithFlags(&ss, cudaStreamNonBlocking));
    if (launches > 0) { OX(optixLaunch(pipe, sr, dprm, sizeof prm, &sbt, NR, 1, 1)); CK(cudaStreamSynchronize(sr)); OX(optixLaunch(pipe, sr, dprm, sizeof prm, &sbt, NR, 1, 1)); CK(cudaStreamSynchronize(sr)); }   // warm up
    // compute kernel
    int *dstop; unsigned long long *diters, *dres; uint4 *dbuf = nullptr; const size_t nbuf = (size_t)1 << 26;   // 1 GB of uint4
    CK(cudaMalloc((void **)&dstop, 4)); CK(cudaMemset(dstop, 0, 4)); CK(cudaMalloc((void **)&diters, 1024 * 8)); CK(cudaMemset(diters, 0, 1024 * 8)); CK(cudaMalloc((void **)&dres, 1024 * 8)); CK(cudaMemset(dres, 0, 1024 * 8));
    CK(cudaMalloc((void **)&dbuf, nbuf * 16)); CK(cudaMemset(dbuf, 1, nbuf * 16));
    const bool comp = place != "none"; const int parity = place == "odd" ? 1 : -1; const int tp = type == "alu" ? 0 : type == "l1" ? 1 : 2;
    const int per_sm = place == "odd" ? 4 : 2;
    auto launch_compute = [&]() {
        // dynamic shared memory pads the footprint so that exactly per_sm blocks fit one SM (4: 24 KB each = 96 KB of 100; 2: 48 KB each)
        if (getenv("TPC_CARVE")) { const int cv = atoi(getenv("TPC_CARVE")); CK(cudaFuncSetAttribute(compute4<0>, cudaFuncAttributePreferredSharedMemoryCarveout, cv)); CK(cudaFuncSetAttribute(compute4<1>, cudaFuncAttributePreferredSharedMemoryCarveout, cv)); CK(cudaFuncSetAttribute(compute4<2>, cudaFuncAttributePreferredSharedMemoryCarveout, cv)); }
        const size_t dyn = 0; const int grid = place == "odd" ? NSM * 4 * 40 : NSM * 2;
        if (tp == 0) { CK(cudaFuncSetAttribute(compute4<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dyn)); compute4<0><<<grid, 384, 0, sc>>>(parity, dstop, diters, dres, dbuf, nbuf); }
        else if (tp == 1) { CK(cudaFuncSetAttribute(compute4<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dyn)); compute4<1><<<grid, 384, 0, sc>>>(parity, dstop, diters, dres, dbuf, nbuf); }
        else { CK(cudaFuncSetAttribute(compute4<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dyn)); compute4<2><<<grid, 384, 0, sc>>>(parity, dstop, diters, dres, dbuf, nbuf); }
        CK(cudaGetLastError());
    };
    std::thread wd([&]() { std::this_thread::sleep_for(std::chrono::milliseconds(getenv("TPC_WD_MS") ? atoi(getenv("TPC_WD_MS")) : 20000)); int o = 1; cudaMemcpyAsync(dstop, &o, 4, cudaMemcpyHostToDevice, ss); });   // watchdog: stops the compute kernel even if an OptiX launch is stuck behind it
    if (comp) { launch_compute(); std::this_thread::sleep_for(std::chrono::milliseconds(300)); }   // let the blocks settle on their SMs
    auto sample = [&]() { unsigned long long a[1024]; CK(cudaMemcpyAsync(a, diters, 1024 * 8, cudaMemcpyDeviceToHost, ss)); CK(cudaStreamSynchronize(ss)); unsigned long long t = 0; for (int q = 0; q < NSM; ++q) t += a[q]; return t; };
    unsigned long long it0 = sample();
    auto t0 = std::chrono::steady_clock::now(); std::vector<float> ms; cudaEvent_t a0, a1; CK(cudaEventCreate(&a0)); CK(cudaEventCreate(&a1));
    if (launches > 0) for (int k = 0; k < launches; ++k) { CK(cudaEventRecord(a0, sr)); OX(optixLaunch(pipe, sr, dprm, sizeof prm, &sbt, NR, 1, 1)); CK(cudaEventRecord(a1, sr)); CK(cudaEventSynchronize(a1)); float x; CK(cudaEventElapsedTime(&x, a0, a1)); ms.push_back(x); }
    else std::this_thread::sleep_for(std::chrono::milliseconds(window_ms));
    unsigned long long itw = sample(); auto t1 = std::chrono::steady_clock::now();
    int one = 1; CK(cudaMemcpyAsync(dstop, &one, 4, cudaMemcpyHostToDevice, ss)); CK(cudaStreamSynchronize(ss)); CK(cudaStreamSynchronize(sc));
    auto t2 = std::chrono::steady_clock::now();
    unsigned long long res[1024]; CK(cudaMemcpy(res, dres, 1024 * 8, cudaMemcpyDeviceToHost));
    const double win = std::chrono::duration<double>(t1 - t0).count(); (void)t2;
    unsigned long long blocks_even = 0, blocks_odd = 0; int sms_used = 0; for (int s = 0; s < NSM; ++s) { (s & 1 ? blocks_odd : blocks_even) += res[s]; sms_used += res[s] > 0; }
    wd.join();
    printf("{\"place\":\"%s\",\"type\":\"%s\",\"rays\":%u,\"ms\":[", place.c_str(), type.c_str(), NR); for (size_t k = 0; k < ms.size(); ++k) printf("%s%.4f", k ? "," : "", ms[k]);
    printf("],\"window_s\":%.4f,\"compute_iters\":%llu,\"blocks_even\":%llu,\"blocks_odd\":%llu,\"sms_with_blocks\":%d}\n", win, itw - it0, blocks_even, blocks_odd, sms_used);
    return 0;
}
