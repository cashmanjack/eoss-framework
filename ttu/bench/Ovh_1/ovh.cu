// Ovh_1 host (OptiX only, no TTU code). Predictions: model/predictions_ovh.md.
//   ovh PTX OUT.json scene=tri1|soup nrep=0,1,3 sizes=32,4096,... reps=20 ncal=20 [ncu=1: one size, one launch, no events]
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include "ovh_params.h"
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(2); } } while (0)
#define OX(x) do { OptixResult r_ = (x); if (r_ != OPTIX_SUCCESS) { fprintf(stderr, "%s:%d %s: %d\n", __FILE__, __LINE__, #x, (int)r_); exit(2); } } while (0)
struct Tri { float3 a, b, c; };
static std::string read_file(const char *p) { std::ifstream f(p); if (!f) { fprintf(stderr, "cannot read %s\n", p); exit(2); } std::stringstream s; s << f.rdbuf(); return s.str(); }
static std::vector<long long> parse_list(const std::string &s) { std::vector<long long> v; std::stringstream ss(s); std::string t; while (std::getline(ss, t, ',')) v.push_back(atoll(t.c_str())); return v; }

int main(int argc, char **argv)
{
    if (argc < 3) { fprintf(stderr, "usage\n"); return 2; }
    std::map<std::string, std::string> kv; for (int a = 3; a < argc; ++a) { std::string s = argv[a]; const size_t e = s.find('='); if (e == std::string::npos) { fprintf(stderr, "bad arg %s\n", argv[a]); return 2; } kv[s.substr(0, e)] = s.substr(e + 1); }
    const std::string scene = kv.count("scene") ? kv["scene"] : "tri1";
    const std::vector<long long> nreps = parse_list(kv.count("nrep") ? kv["nrep"] : "1");
    const std::vector<long long> sizes = parse_list(kv.count("sizes") ? kv["sizes"] : "32,4096,16384,65536,262144,1048576,4194304,8388608");
    const int reps = kv.count("reps") ? atoi(kv["reps"].c_str()) : 20, ncal = kv.count("ncal") ? atoi(kv["ncal"].c_str()) : 20, ncu = kv.count("ncu") ? atoi(kv["ncu"].c_str()) : 0;
    CK(cudaFree(0)); OX(optixInit()); OptixDeviceContextOptions co{}; OptixDeviceContext ctx; OX(optixDeviceContextCreate(0, &co, &ctx));
    std::mt19937 g(1234); std::uniform_real_distribution<float> U(-1.f, 1.f);
    std::vector<Tri> v; unsigned sc = 0;
    if (scene == "tri1") v.push_back(Tri{{-4.f, -4.f, 0.f}, {4.f, -4.f, 0.f}, {0.f, 4.f, 0.f}});
    else if (scene == "soup") {   // the soup of setup3 / BVH_4 / Flags_1 (same generator), 100,000 triangles
        sc = 1; const long soupn = 100000; const float e = 3.f / std::cbrt((float)soupn);
        for (long i = 0; i < soupn; ++i) { const float3 c{U(g), U(g), U(g)}; auto pt = [&]() { return float3{c.x + 0.5f * e * U(g), c.y + 0.5f * e * U(g), c.z + 0.5f * e * U(g)}; }; v.push_back({pt(), pt(), pt()}); }
    } else { fprintf(stderr, "scene?\n"); return 2; }
    CUdeviceptr dv; CK(cudaMalloc((void **)&dv, v.size() * sizeof(Tri))); CK(cudaMemcpy((void *)dv, v.data(), v.size() * sizeof(Tri), cudaMemcpyHostToDevice));
    unsigned gflag = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
    OptixBuildInput bi{}; bi.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES; bi.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3; bi.triangleArray.numVertices = (unsigned)v.size() * 3;
    bi.triangleArray.vertexBuffers = &dv; bi.triangleArray.flags = &gflag; bi.triangleArray.numSbtRecords = 1;
    OptixAccelBuildOptions ao{}; ao.buildFlags = OPTIX_BUILD_FLAG_NONE; ao.operation = OPTIX_BUILD_OPERATION_BUILD;
    OptixAccelBufferSizes bs{}; OX(optixAccelComputeMemoryUsage(ctx, &ao, &bi, 1, &bs));
    CUdeviceptr tmp, gas; CK(cudaMalloc((void **)&tmp, bs.tempSizeInBytes)); CK(cudaMalloc((void **)&gas, bs.outputSizeInBytes));
    OptixTraversableHandle root; OX(optixAccelBuild(ctx, 0, &ao, &bi, 1, tmp, bs.tempSizeInBytes, gas, bs.outputSizeInBytes, &root, nullptr, 0)); CK(cudaDeviceSynchronize());
    OptixModuleCompileOptions mo{}; OptixPipelineCompileOptions po{};
    po.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS; po.numPayloadValues = 2; po.numAttributeValues = 2; po.pipelineLaunchParamsVariableName = "params"; po.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
    std::string ptx = read_file(argv[1]); char log[4096]; size_t ls = sizeof log; OptixModule m; OX(optixModuleCreate(ctx, &mo, &po, ptx.c_str(), ptx.size(), log, &ls, &m));
    OptixProgramGroupOptions go{}; OptixProgramGroupDesc d[3] = {};
    d[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN; d[0].raygen.module = m; d[0].raygen.entryFunctionName = "__raygen__rg";
    d[1].kind = OPTIX_PROGRAM_GROUP_KIND_MISS; d[1].miss.module = m; d[1].miss.entryFunctionName = "__miss__ms";
    d[2].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP; d[2].hitgroup.moduleCH = m; d[2].hitgroup.entryFunctionNameCH = "__closesthit__ch";
    OptixProgramGroup pg[3]; ls = sizeof log; OX(optixProgramGroupCreate(ctx, d, 3, &go, log, &ls, pg));
    OptixPipelineLinkOptions lo{}; lo.maxTraceDepth = 1; OptixPipeline pipe; ls = sizeof log; OX(optixPipelineCreate(ctx, &po, &lo, pg, 3, log, &ls, &pipe));
    struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) Rec { char h[OPTIX_SBT_RECORD_HEADER_SIZE]; };
    Rec rec[3]; for (int k = 0; k < 3; ++k) OX(optixSbtRecordPackHeader(pg[k], &rec[k]));
    CUdeviceptr drec; CK(cudaMalloc((void **)&drec, sizeof rec)); CK(cudaMemcpy((void *)drec, rec, sizeof rec, cudaMemcpyHostToDevice));
    OptixShaderBindingTable sbt{}; sbt.raygenRecord = drec; sbt.missRecordBase = drec + sizeof(Rec); sbt.missRecordStrideInBytes = sizeof(Rec); sbt.missRecordCount = 1;
    sbt.hitgroupRecordBase = drec + 2 * sizeof(Rec); sbt.hitgroupRecordStrideInBytes = sizeof(Rec); sbt.hitgroupRecordCount = 1;
    FILE *jf = fopen(argv[2], "wx"); if (!jf) { fprintf(stderr, "refuse to overwrite or cannot write %s\n", argv[2]); return 2; }
    fprintf(jf, "{\"scene\": \"%s\", \"ptx\": \"%s\", \"reps\": %d, \"ncal\": %d, \"runs\": [\n", scene.c_str(), argv[1], reps, ncal);
    unsigned long long *dcal; unsigned *dsink; const long long maxn = *std::max_element(sizes.begin(), sizes.end());
    CK(cudaMalloc((void **)&dcal, (size_t)(maxn / 32 + 1) * 32)); CK(cudaMalloc((void **)&dsink, (size_t)(maxn / 32 + 1) * 4)); CK(cudaMemset(dsink, 0, (size_t)(maxn / 32 + 1) * 4));
    bool first = true;
    for (long long nr : nreps) for (long long N : sizes) {
        Params prm{}; prm.handle = root; prm.n = (unsigned)N; prm.nrep = (unsigned)nr; prm.scene = sc; prm.sink = dsink; prm.cal = nullptr;
        CUdeviceptr dprm; CK(cudaMalloc((void **)&dprm, sizeof prm));
        auto launch = [&]() { CK(cudaMemcpy((void *)dprm, &prm, sizeof prm, cudaMemcpyHostToDevice)); OX(optixLaunch(pipe, 0, dprm, sizeof prm, &sbt, (unsigned)N, 1, 1)); };
        launch(); CK(cudaDeviceSynchronize());   // warm up (code, BVH in L2)
        std::vector<float> ms; cudaEvent_t a0, a1; CK(cudaEventCreate(&a0)); CK(cudaEventCreate(&a1));
        for (int k = 0; k < (ncu ? 1 : reps); ++k) { CK(cudaMemcpy((void *)dprm, &prm, sizeof prm, cudaMemcpyHostToDevice)); CK(cudaEventRecord(a0)); OX(optixLaunch(pipe, 0, dprm, sizeof prm, &sbt, (unsigned)N, 1, 1)); CK(cudaEventRecord(a1)); CK(cudaEventSynchronize(a1)); float t; CK(cudaEventElapsedTime(&t, a0, a1)); ms.push_back(t); }
        // calibration launches: per warp clock64 and globaltimer at start and end of the raygen
        const size_t nw = (size_t)((N + 31) / 32); std::vector<unsigned long long> cal; prm.cal = dcal;
        for (int k = 0; k < (ncu ? 0 : ncal); ++k) { launch(); CK(cudaDeviceSynchronize()); std::vector<unsigned long long> c(nw * 4); CK(cudaMemcpy(c.data(), dcal, nw * 32, cudaMemcpyDeviceToHost)); cal.insert(cal.end(), c.begin(), c.end()); }
        fprintf(jf, "%s {\"nrep\": %lld, \"n\": %lld, \"ms\": [", first ? "" : ",\n", nr, N); first = false;
        for (size_t k = 0; k < ms.size(); ++k) fprintf(jf, "%s%.6f", k ? "," : "", ms[k]);
        fprintf(jf, "], \"ncalw\": %zu, \"cal\": [", nw);
        for (size_t k = 0; k < cal.size(); ++k) fprintf(jf, "%s%llu", k ? "," : "", cal[k]);
        fprintf(jf, "]}");
        CK(cudaFree((void *)dprm));
    }
    fprintf(jf, "\n]}\n"); fclose(jf); printf("done %s\n", argv[2]);
    return 0;
}
