// NNsphere B host (OptiX only, no TTU code). usage: nn sph|cus PTX OUT.json data=FILE.bin n=N K=K r=R [qn=Q chunk=262144 reps=5 dump=PREFIX]
// Timed span (events): GAS build (cus: plus the AABB kernel) + result buffer memset + all query launches. Allocation, point upload and the final copies are outside the span.
// The count/dump run after the timed reps: same launches with count=1, lists, list sizes and candidate counts written to new files PREFIX_{ids,n,cand}.bin.
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include "nn_params.h"
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(2); } } while (0)
#define OX(x) do { OptixResult r_ = (x); if (r_ != OPTIX_SUCCESS) { fprintf(stderr, "%s:%d %s: %d\n", __FILE__, __LINE__, #x, (int)r_); exit(2); } } while (0)
__global__ void mkaabb(const float3 *p, OptixAabb *b, unsigned n, float h) { const unsigned i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) b[i] = {p[i].x - h, p[i].y - h, p[i].z - h, p[i].x + h, p[i].y + h, p[i].z + h}; }
__device__ __forceinline__ unsigned spread10(unsigned v) { v &= 0x3ff; v = (v | (v << 16)) & 0x030000FF; v = (v | (v << 8)) & 0x0300F00F; v = (v | (v << 4)) & 0x030C30C3; v = (v | (v << 2)) & 0x09249249; return v; }
__global__ void mkmorton(const float3 *p, unsigned *key, unsigned *idx, unsigned n) { const unsigned i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) { auto q = [](float x) { return (unsigned)fminf(fmaxf(x * 1024.f, 0.f), 1023.f); }; key[i] = spread10(q(p[i].x)) | (spread10(q(p[i].y)) << 1) | (spread10(q(p[i].z)) << 2); idx[i] = i; } }
__global__ void gather(const float3 *p, const unsigned *idx, float3 *o, unsigned n) { const unsigned i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) o[i] = p[idx[i]]; }
int main(int argc, char **argv)
{
    if (argc < 4) return 2; const std::string arm = argv[1]; const bool sph = arm == "sph"; if (!sph && arm != "cus") return 2;
    std::map<std::string, std::string> kv; for (int a = 4; a < argc; ++a) { std::string s = argv[a]; const size_t e = s.find('='); kv[s.substr(0, e)] = s.substr(e + 1); }
    auto I = [&](const char *k, long d) { return kv.count(k) ? atol(kv[k].c_str()) : d; };
    const bool dosort = I("sort", 0) != 0; const long N = I("n", 0), Kk = I("K", 0), Q = I("qn", N), CH = I("chunk", 262144); const int reps = (int)I("reps", 5); const double r = atof(kv["r"].c_str());
    std::vector<float> h((size_t)N * 3); { FILE *f = fopen(kv["data"].c_str(), "rb"); if (!f || fread(h.data(), 4, h.size(), f) != h.size()) { fprintf(stderr, "bad data\n"); return 2; } fclose(f); }
    CK(cudaFree(0)); OX(optixInit()); OptixDeviceContextOptions co{}; OptixDeviceContext ctx; OX(optixDeviceContextCreate(0, &co, &ctx));
    float3 *dp0; CK(cudaMalloc((void **)&dp0, N * 12)); CK(cudaMemcpy(dp0, h.data(), N * 12, cudaMemcpyHostToDevice));
    float3 *dps = nullptr; unsigned *dkey = nullptr, *dperm = nullptr; if (dosort) { CK(cudaMalloc((void **)&dps, N * 12)); CK(cudaMalloc((void **)&dkey, N * 4)); CK(cudaMalloc((void **)&dperm, N * 4)); }
    float3 *dp = dosort ? dps : dp0;   // points (and queries) the search uses: the Morton sorted copy when sort=1 (ids in the lists are then positions in the sorted order; PREFIX_perm.bin maps them back)
    auto sortpts = [&]() { mkmorton<<<(N + 255) / 256, 256>>>(dp0, dkey, dperm, (unsigned)N); thrust::sort_by_key(thrust::device_pointer_cast(dkey), thrust::device_pointer_cast(dkey) + N, thrust::device_pointer_cast(dperm)); gather<<<(N + 255) / 256, 256>>>(dp0, dperm, dps, (unsigned)N); };
    if (dosort) sortpts();   // once before the size query and the pipeline set up
    const float rb = (float)(r * (1 + 5e-4)), L = (float)(2 * r * (1 + 1e-3)), rr = (float)(1e-3 * r), pad = (float)(r * (1 + 1e-4));
    float *drad; CK(cudaMalloc((void **)&drad, 4)); CK(cudaMemcpy(drad, &rb, 4, cudaMemcpyHostToDevice));
    OptixAabb *dbb = nullptr; if (!sph) CK(cudaMalloc((void **)&dbb, N * sizeof(OptixAabb)));
    unsigned gflag = sph ? OPTIX_GEOMETRY_FLAG_NONE : OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT; CUdeviceptr dA = (CUdeviceptr)dp, dB = (CUdeviceptr)drad, dAb = (CUdeviceptr)dbb; OptixBuildInput bi{};
    if (sph) { bi.type = OPTIX_BUILD_INPUT_TYPE_SPHERES; bi.sphereArray.vertexBuffers = &dA; bi.sphereArray.numVertices = (unsigned)N; bi.sphereArray.radiusBuffers = &dB; bi.sphereArray.singleRadius = 1; bi.sphereArray.flags = &gflag; bi.sphereArray.numSbtRecords = 1; }
    else { bi.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES; bi.customPrimitiveArray.aabbBuffers = &dAb; bi.customPrimitiveArray.numPrimitives = (unsigned)N; bi.customPrimitiveArray.flags = &gflag; bi.customPrimitiveArray.numSbtRecords = 1; }
    OptixAccelBuildOptions ao{}; ao.buildFlags = OPTIX_BUILD_FLAG_NONE; ao.operation = OPTIX_BUILD_OPERATION_BUILD; OptixAccelBufferSizes bs{};
    if (!sph) mkaabb<<<(N + 255) / 256, 256>>>(dp, dbb, (unsigned)N, pad);   // sizes only need the pointer; the real kernel runs again inside the span
    OX(optixAccelComputeMemoryUsage(ctx, &ao, &bi, 1, &bs)); CUdeviceptr tmp, gas; CK(cudaMalloc((void **)&tmp, bs.tempSizeInBytes)); CK(cudaMalloc((void **)&gas, bs.outputSizeInBytes));
    OptixModuleCompileOptions mo{}; OptixPipelineCompileOptions po{}; po.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS; po.numPayloadValues = 3; po.numAttributeValues = 2; po.pipelineLaunchParamsVariableName = "params";
    po.usesPrimitiveTypeFlags = sph ? OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE : OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;
    std::ifstream pf(argv[2]); std::stringstream ss; ss << pf.rdbuf(); std::string ptx = ss.str(); char log[4096]; size_t ls = sizeof log; OptixModule m; OX(optixModuleCreate(ctx, &mo, &po, ptx.c_str(), ptx.size(), log, &ls, &m));
    OptixModule bm = nullptr; if (sph) { OptixBuiltinISOptions bio{}; bio.builtinISModuleType = OPTIX_PRIMITIVE_TYPE_SPHERE; bio.buildFlags = ao.buildFlags; OX(optixBuiltinISModuleGet(ctx, &mo, &po, &bio, &bm)); }
    OptixProgramGroupOptions go{}; OptixProgramGroupDesc d[3] = {}; d[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN; d[0].raygen.module = m; d[0].raygen.entryFunctionName = sph ? "__raygen__sph" : "__raygen__cus";
    d[1].kind = OPTIX_PROGRAM_GROUP_KIND_MISS; d[1].miss.module = m; d[1].miss.entryFunctionName = "__miss__ms";
    d[2].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP; if (sph) { d[2].hitgroup.moduleAH = m; d[2].hitgroup.entryFunctionNameAH = "__anyhit__sph"; d[2].hitgroup.moduleIS = bm; } else { d[2].hitgroup.moduleIS = m; d[2].hitgroup.entryFunctionNameIS = "__intersection__cus"; }
    OptixProgramGroup pg[3]; ls = sizeof log; OX(optixProgramGroupCreate(ctx, d, 3, &go, log, &ls, pg));
    OptixPipelineLinkOptions lo{}; lo.maxTraceDepth = 1; OptixPipeline pipe; ls = sizeof log; OX(optixPipelineCreate(ctx, &po, &lo, pg, 3, log, &ls, &pipe));
    struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) Rec { char hd[OPTIX_SBT_RECORD_HEADER_SIZE]; }; Rec rec[3]; for (int k = 0; k < 3; ++k) OX(optixSbtRecordPackHeader(pg[k], &rec[k]));
    CUdeviceptr drec; CK(cudaMalloc((void **)&drec, sizeof rec)); CK(cudaMemcpy((void *)drec, rec, sizeof rec, cudaMemcpyHostToDevice));
    OptixShaderBindingTable sbt{}; sbt.raygenRecord = drec; sbt.missRecordBase = drec + sizeof(Rec); sbt.missRecordStrideInBytes = sizeof(Rec); sbt.missRecordCount = 1; sbt.hitgroupRecordBase = drec + 2 * sizeof(Rec); sbt.hitgroupRecordStrideInBytes = sizeof(Rec); sbt.hitgroupRecordCount = 1;
    float *dld; int *dli; unsigned *dn, *dcand; CK(cudaMalloc((void **)&dld, (size_t)Q * Kk * 4)); CK(cudaMalloc((void **)&dli, (size_t)Q * Kk * 4)); CK(cudaMalloc((void **)&dn, (size_t)N * 4)); CK(cudaMalloc((void **)&dcand, (size_t)N * 4));
    Params prm{}; prm.pts = dp; prm.ld = dld; prm.li = dli; prm.nout = dn; prm.cand = dcand; prm.r2 = (float)(r * r); prm.L = L; prm.rr = rr; prm.dir = make_float3(1.f, 0.f, 0.f);
    CUdeviceptr dprm; CK(cudaMalloc((void **)&dprm, sizeof prm)); OptixTraversableHandle root = 0;
    FILE *jf = fopen(argv[3], "wx"); if (!jf) { fprintf(stderr, "refuse to overwrite %s\n", argv[3]); return 2; }
    cudaEvent_t e0, e1, e2, e3, e4; double sort_ms_last = 0; CK(cudaEventCreate(&e4)); CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1)); CK(cudaEventCreate(&e2)); CK(cudaEventCreate(&e3));
    auto span = [&](unsigned count, double *build_ms, double *total_ms, std::vector<float> *chunk_ms) {
        CK(cudaEventRecord(e0)); if (dosort) sortpts(); CK(cudaEventRecord(e4)); if (!sph) mkaabb<<<(N + 255) / 256, 256>>>(dp, dbb, (unsigned)N, pad);
        OX(optixAccelBuild(ctx, 0, &ao, &bi, 1, tmp, bs.tempSizeInBytes, gas, bs.outputSizeInBytes, &root, nullptr, 0)); CK(cudaEventRecord(e1));
        CK(cudaMemset(dli, 0xFF, (size_t)Q * Kk * 4)); CK(cudaMemset(dld, 0x7F, (size_t)Q * Kk * 4)); prm.handle = root; prm.count = count;
        for (long s = 0; s < Q; s += CH) { prm.qoff = (unsigned)s; prm.qn = (unsigned)std::min<long>(CH, Q - s); CK(cudaMemcpy((void *)dprm, &prm, sizeof prm, cudaMemcpyHostToDevice));
            if (chunk_ms) CK(cudaEventRecord(e2)); OX(optixLaunch(pipe, 0, dprm, sizeof prm, &sbt, prm.qn, 1, 1));
            if (chunk_ms) { CK(cudaEventRecord(e3)); CK(cudaEventSynchronize(e3)); float t; CK(cudaEventElapsedTime(&t, e2, e3)); chunk_ms->push_back(t); } }
        CK(cudaEventRecord(e3)); CK(cudaEventSynchronize(e3)); float a, b; CK(cudaEventElapsedTime(&a, e0, e1)); CK(cudaEventElapsedTime(&b, e0, e3)); *build_ms = a; *total_ms = b; { float sm; CK(cudaEventElapsedTime(&sm, e0, e4)); sort_ms_last = sm; }
    };
    double bm0, tm0; span(0, &bm0, &tm0, nullptr);   // warm up
    fprintf(jf, "{\"arm\": \"%s\", \"n\": %ld, \"q\": %ld, \"K\": %ld, \"r\": %.9g, \"chunk\": %ld, \"gas_bytes\": %zu, \"warm_total_ms\": %.4f, \"total_ms\": [", arm.c_str(), N, Q, Kk, r, CH, (size_t)bs.outputSizeInBytes, tm0);
    std::vector<double> bms, wms, sms; for (int k = 0; k < reps; ++k) { double bb, tt; CK(cudaDeviceSynchronize()); const auto w0 = std::chrono::steady_clock::now(); span(0, &bb, &tt, nullptr); CK(cudaDeviceSynchronize()); wms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - w0).count()); bms.push_back(bb); sms.push_back(sort_ms_last); fprintf(jf, "%s%.4f", k ? "," : "", tt); fflush(jf); }   // wall_ms: host clock around the same span (cross check of the events)
    fprintf(jf, "], \"sort\": %d, \"sort_ms\": [", (int)dosort); for (size_t k = 0; k < sms.size(); ++k) fprintf(jf, "%s%.4f", k ? "," : "", sms[k]); fprintf(jf, "], \"wall_ms\": ["); for (size_t k = 0; k < wms.size(); ++k) fprintf(jf, "%s%.4f", k ? "," : "", wms[k]); fprintf(jf, "], \"build_ms\": ["); for (size_t k = 0; k < bms.size(); ++k) fprintf(jf, "%s%.4f", k ? "," : "", bms[k]);
    std::vector<float> cm; double bb, tt; span(1, &bb, &tt, &cm); fprintf(jf, "], \"count_run_total_ms\": %.4f, \"count_run_chunks_ms\": [", tt); for (size_t k = 0; k < cm.size(); ++k) fprintf(jf, "%s%.4f", k ? "," : "", cm[k]);
    std::vector<unsigned> hc(Q), hn(Q); CK(cudaMemcpy(hc.data(), dcand, (size_t)Q * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hn.data(), dn, (size_t)Q * 4, cudaMemcpyDeviceToHost));
    double sc = 0, sn = 0; unsigned mx = 0; for (long k = 0; k < Q; ++k) { sc += hc[k]; sn += hn[k]; mx = std::max(mx, hc[k]); }
    fprintf(jf, "], \"mean_cand\": %.6f, \"max_cand\": %u, \"mean_list\": %.6f", sc / Q, mx, sn / Q);
    if (kv.count("dump")) {
        std::vector<int> hi((size_t)Q * Kk); CK(cudaMemcpy(hi.data(), dli, hi.size() * 4, cudaMemcpyDeviceToHost)); const std::string P = kv["dump"];
        FILE *a = fopen((P + "_ids.bin").c_str(), "wxb"), *b = fopen((P + "_n.bin").c_str(), "wxb"), *c = fopen((P + "_cand.bin").c_str(), "wxb"); if (!a || !b || !c) { fprintf(stderr, "dump files exist\n"); return 2; }
        if (dosort) { std::vector<unsigned> hp(N); CK(cudaMemcpy(hp.data(), dperm, N * 4, cudaMemcpyDeviceToHost)); FILE *pf2 = fopen((P + "_perm.bin").c_str(), "wxb"); if (pf2) { fwrite(hp.data(), 4, N, pf2); fclose(pf2); } }
        fwrite(hi.data(), 4, hi.size(), a); fwrite(hn.data(), 4, Q, b); fwrite(hc.data(), 4, Q, c); fclose(a); fclose(b); fclose(c); fprintf(jf, ", \"dump\": \"%s\"", P.c_str());
    }
    fprintf(jf, "}\n"); fclose(jf); printf("done %s\n", argv[3]); return 0;
}
