// Struct_2 host. s2 PTX OUT.json mode=build|inst|part key=value ...
//  build: N=1000000 flags=none|trace|build|update|compact reps=15      (time of optixAccelBuild; update also times a refit; compact also times optixAccelCompact)
//  inst:  M=4096 total=1000000 big=2097152 reps=20 (M=0: flat GAS of total triangles)   part: active=32 spread=0|1 big=2097152 N=1000000
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
#include "s2_params.h"
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(2); } } while (0)
#define OX(x) do { OptixResult r_ = (x); if (r_ != OPTIX_SUCCESS) { fprintf(stderr, "%s:%d %s: %d\n", __FILE__, __LINE__, #x, (int)r_); exit(2); } } while (0)
struct Tri { float3 a, b, c; };
static OptixDeviceContext ctx;
static std::string read_file(const char *p) { std::ifstream f(p); if (!f) { fprintf(stderr, "cannot read %s\n", p); exit(2); } std::stringstream s; s << f.rdbuf(); return s.str(); }
static std::vector<Tri> soup(long n, unsigned seed, float scale, float3 off)
{ std::mt19937 g(seed); std::uniform_real_distribution<float> U(-1.f, 1.f); std::vector<Tri> v; const float e = 3.f / std::cbrt((float)n) * scale;
  for (long i = 0; i < n; ++i) { const float3 c{U(g) * scale, U(g) * scale, U(g) * scale}; auto pt = [&]() { return float3{off.x + c.x + 0.5f * e * U(g), off.y + c.y + 0.5f * e * U(g), off.z + c.z + 0.5f * e * U(g)}; }; v.push_back({pt(), pt(), pt()}); } return v; }
struct Gas { CUdeviceptr buf; OptixTraversableHandle h; size_t bytes; double ms; };
static Gas build_gas(const std::vector<Tri> &v, unsigned flags, bool compact, double *refit_ms = nullptr, double *compact_ms = nullptr)
{
    CUdeviceptr dv; CK(cudaMalloc((void **)&dv, v.size() * sizeof(Tri))); CK(cudaMemcpy((void *)dv, v.data(), v.size() * sizeof(Tri), cudaMemcpyHostToDevice));
    unsigned gf = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT; OptixBuildInput bi{}; bi.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES; bi.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3; bi.triangleArray.numVertices = (unsigned)v.size() * 3; bi.triangleArray.vertexBuffers = &dv; bi.triangleArray.flags = &gf; bi.triangleArray.numSbtRecords = 1;
    OptixAccelBuildOptions ao{}; ao.buildFlags = flags | (compact ? OPTIX_BUILD_FLAG_ALLOW_COMPACTION : 0); ao.operation = OPTIX_BUILD_OPERATION_BUILD;
    OptixAccelBufferSizes bs{}; OX(optixAccelComputeMemoryUsage(ctx, &ao, &bi, 1, &bs)); CUdeviceptr tmp, out, csz; CK(cudaMalloc((void **)&tmp, bs.tempSizeInBytes)); CK(cudaMalloc((void **)&out, bs.outputSizeInBytes)); CK(cudaMalloc((void **)&csz, 8));
    OptixAccelEmitDesc em{}; em.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE; em.result = csz; cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    Gas g{}; g.bytes = bs.outputSizeInBytes; CK(cudaEventRecord(a)); OX(optixAccelBuild(ctx, 0, &ao, &bi, 1, tmp, bs.tempSizeInBytes, out, bs.outputSizeInBytes, &g.h, compact ? &em : nullptr, compact ? 1 : 0)); CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    float t; cudaEventElapsedTime(&t, a, b); g.ms = t; g.buf = out;
    if (refit_ms) { ao.operation = OPTIX_BUILD_OPERATION_UPDATE; CK(cudaEventRecord(a)); OX(optixAccelBuild(ctx, 0, &ao, &bi, 1, tmp, bs.tempSizeInBytes, out, bs.outputSizeInBytes, &g.h, nullptr, 0)); CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b)); cudaEventElapsedTime(&t, a, b); *refit_ms = t; }
    if (compact) { unsigned long long cs; CK(cudaMemcpy(&cs, (void *)csz, 8, cudaMemcpyDeviceToHost)); CUdeviceptr g2; CK(cudaMalloc((void **)&g2, cs)); CK(cudaEventRecord(a)); OX(optixAccelCompact(ctx, 0, g.h, g2, cs, &g.h)); CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b)); cudaEventElapsedTime(&t, a, b); if (compact_ms) *compact_ms = t; g.bytes = cs; g.buf = g2; }
    CK(cudaFree((void *)tmp)); CK(cudaFree((void *)dv)); return g;
}
int main(int argc, char **argv)
{
    if (argc < 3) return 2; std::map<std::string, std::string> kv; for (int a = 3; a < argc; ++a) { std::string s = argv[a]; const size_t e = s.find('='); kv[s.substr(0, e)] = s.substr(e + 1); }
    auto geti = [&](const char *k, long long d) { return kv.count(k) ? atoll(kv[k].c_str()) : d; }; auto gets = [&](const char *k, const char *d) { return kv.count(k) ? kv[k] : std::string(d); };
    const std::string mode = gets("mode", "inst"); CK(cudaFree(0)); OX(optixInit()); OptixDeviceContextOptions co{}; OX(optixDeviceContextCreate(0, &co, &ctx));
    FILE *jf = fopen(argv[2], "wx"); if (!jf) { fprintf(stderr, "cannot create %s\n", argv[2]); return 2; }
    if (mode == "build") {
        const long N = geti("N", 1000000); const int reps = (int)geti("reps", 15); const std::string fl = gets("flags", "none"); auto v = soup(N, 1234, 1.f, float3{0, 0, 0});
        unsigned f = OPTIX_BUILD_FLAG_NONE; bool comp = false, upd = false; if (fl == "trace") f = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE; else if (fl == "build") f = OPTIX_BUILD_FLAG_PREFER_FAST_BUILD; else if (fl == "update") { f = OPTIX_BUILD_FLAG_ALLOW_UPDATE; upd = true; } else if (fl == "compact") comp = true;
        std::vector<double> bt, rt, ct; size_t bytes = 0;
        for (int r = 0; r < reps + 2; ++r) { double rf = 0, cf = 0; Gas g = build_gas(v, f, comp, upd ? &rf : nullptr, comp ? &cf : nullptr); if (r >= 2) { bt.push_back(g.ms); rt.push_back(rf); ct.push_back(cf); } bytes = g.bytes; CK(cudaFree((void *)g.buf)); }
        fprintf(jf, "{\"mode\": \"build\", \"N\": %ld, \"flags\": \"%s\", \"bytes\": %zu, \"build_ms\": [", N, fl.c_str(), bytes); for (size_t i = 0; i < bt.size(); ++i) fprintf(jf, "%s%.5f", i ? "," : "", bt[i]); fprintf(jf, "], \"refit_ms\": ["); for (size_t i = 0; i < rt.size(); ++i) fprintf(jf, "%s%.5f", i ? "," : "", rt[i]); fprintf(jf, "], \"compact_ms\": ["); for (size_t i = 0; i < ct.size(); ++i) fprintf(jf, "%s%.5f", i ? "," : "", ct[i]); fprintf(jf, "]}\n"); fclose(jf); printf("done\n"); return 0;
    }
    // trace modes: build a scene
    const unsigned NB = (unsigned)geti("big", 2097152); const int reps = (int)geti("reps", 20); OptixTraversableHandle root; size_t bytes = 0; double ias_ms = 0, gas_ms = 0; long M = 0, T = 0; unsigned A = (unsigned)geti("active", 32);
    if (mode == "inst") {
        M = geti("M", 4096); const long total = geti("total", 1000000);
        if (M == 0) { auto v = soup(total, 1234, 1.f, float3{0, 0, 0}); Gas g = build_gas(v, 0, false); root = g.h; bytes = g.bytes; gas_ms = g.ms; }
        else {
            T = total / M; const float sc = 1.f / std::cbrt((float)M);   // each instance is a small cube of triangles, translated; same total triangle count
            auto v = soup(T, 99, sc, float3{0, 0, 0}); Gas g = build_gas(v, 0, false); bytes = g.bytes; gas_ms = g.ms;
            std::mt19937 gr(7); std::uniform_real_distribution<float> U(-1.f, 1.f); std::vector<OptixInstance> inst(M);
            for (long i = 0; i < M; ++i) { OptixInstance &I = inst[i]; memset(&I, 0, sizeof I); const float tr = 1.f - sc; const float m[12] = {1, 0, 0, tr * U(gr), 0, 1, 0, tr * U(gr), 0, 0, 1, tr * U(gr)}; memcpy(I.transform, m, 48); I.instanceId = (unsigned)i; I.visibilityMask = 255; I.traversableHandle = g.h; }
            CUdeviceptr di; CK(cudaMalloc((void **)&di, M * sizeof(OptixInstance))); CK(cudaMemcpy((void *)di, inst.data(), M * sizeof(OptixInstance), cudaMemcpyHostToDevice));
            OptixBuildInput ib{}; ib.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES; ib.instanceArray.instances = di; ib.instanceArray.numInstances = (unsigned)M; OptixAccelBuildOptions ao{}; ao.buildFlags = OPTIX_BUILD_FLAG_NONE; ao.operation = OPTIX_BUILD_OPERATION_BUILD;
            OptixAccelBufferSizes is{}; OX(optixAccelComputeMemoryUsage(ctx, &ao, &ib, 1, &is)); CUdeviceptr tmp, ias; CK(cudaMalloc((void **)&tmp, is.tempSizeInBytes)); CK(cudaMalloc((void **)&ias, is.outputSizeInBytes));
            cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b); CK(cudaEventRecord(a)); OX(optixAccelBuild(ctx, 0, &ao, &ib, 1, tmp, is.tempSizeInBytes, ias, is.outputSizeInBytes, &root, nullptr, 0)); CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b)); float t; cudaEventElapsedTime(&t, a, b); ias_ms = t;
            bytes = g.bytes + is.outputSizeInBytes; (void)bytes; bytes = g.bytes + is.outputSizeInBytes;
        }
    } else { M = 0; auto v = soup(geti("N", 1000000), 1234, 1.f, float3{0, 0, 0}); Gas g = build_gas(v, 0, false); root = g.h; bytes = g.bytes; gas_ms = g.ms; }
    std::mt19937 gr(2026); std::uniform_real_distribution<float> U(-1.f, 1.f); std::normal_distribution<float> Nn(0.f, 1.f); std::vector<Ray8> rays(NB);
    for (auto &r : rays) { float3 d{Nn(gr), Nn(gr), Nn(gr)}; const float l = std::sqrt(d.x * d.x + d.y * d.y + d.z * d.z); r = Ray8{{U(gr), U(gr), U(gr)}, 0.f, {d.x / l, d.y / l, d.z / l}, 1e16f}; }
    OptixModuleCompileOptions mo{}; OptixPipelineCompileOptions po{}; po.traversableGraphFlags = (mode == "inst" && M > 0) ? OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING : OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    po.numPayloadValues = 4; po.numAttributeValues = 2; po.pipelineLaunchParamsVariableName = "params"; po.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
    std::string ptx = read_file(argv[1]); char log[4096]; size_t ls = sizeof log; OptixModule m; OX(optixModuleCreate(ctx, &mo, &po, ptx.c_str(), ptx.size(), log, &ls, &m));
    OptixProgramGroupOptions go{}; OptixProgramGroupDesc d[3] = {}; d[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN; d[0].raygen.module = m; d[0].raygen.entryFunctionName = "__raygen__rg"; d[1].kind = OPTIX_PROGRAM_GROUP_KIND_MISS; d[1].miss.module = m; d[1].miss.entryFunctionName = "__miss__ms";
    d[2].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP; d[2].hitgroup.moduleCH = m; d[2].hitgroup.entryFunctionNameCH = "__closesthit__ch"; OptixProgramGroup pg[3]; ls = sizeof log; OX(optixProgramGroupCreate(ctx, d, 3, &go, log, &ls, pg));
    OptixPipelineLinkOptions lo{}; lo.maxTraceDepth = 1; OptixPipeline pipe; ls = sizeof log; OX(optixPipelineCreate(ctx, &po, &lo, pg, 3, log, &ls, &pipe)); OX(optixPipelineSetStackSize(pipe, 0, 0, 2048, (mode == "inst" && M > 0) ? 2 : 1));
    struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) Rec { char h[OPTIX_SBT_RECORD_HEADER_SIZE]; }; Rec rec[3]; for (int k = 0; k < 3; ++k) OX(optixSbtRecordPackHeader(pg[k], &rec[k]));
    CUdeviceptr drec; CK(cudaMalloc((void **)&drec, sizeof rec)); CK(cudaMemcpy((void *)drec, rec, sizeof rec, cudaMemcpyHostToDevice));
    OptixShaderBindingTable sbt{}; sbt.raygenRecord = drec; sbt.missRecordBase = drec + sizeof(Rec); sbt.missRecordStrideInBytes = sizeof(Rec); sbt.missRecordCount = 1; sbt.hitgroupRecordBase = drec + 2 * sizeof(Rec); sbt.hitgroupRecordStrideInBytes = sizeof(Rec); sbt.hitgroupRecordCount = 1;
    Ray8 *drays; Out *dout; CK(cudaMalloc((void **)&drays, (size_t)NB * sizeof(Ray8))); CK(cudaMalloc((void **)&dout, (size_t)NB * sizeof(Out))); CK(cudaMemcpy(drays, rays.data(), (size_t)NB * sizeof(Ray8), cudaMemcpyHostToDevice));
    Params prm{}; prm.rays = drays; prm.out = dout; prm.handle = root; prm.n = NB; prm.active = A; prm.spread = (unsigned)geti("spread", 0); CUdeviceptr dprm; CK(cudaMalloc((void **)&dprm, sizeof prm));
    const unsigned nthreads = (unsigned)(((unsigned long long)NB * 32 / A + 31) / 32 * 32);
    auto launch = [&]() { CK(cudaMemcpy((void *)dprm, &prm, sizeof prm, cudaMemcpyHostToDevice)); OX(optixLaunch(pipe, 0, dprm, sizeof prm, &sbt, nthreads, 1, 1)); };
    launch(); CK(cudaDeviceSynchronize()); cudaEvent_t a0, a1; CK(cudaEventCreate(&a0)); CK(cudaEventCreate(&a1)); std::vector<float> ms;
    for (int k = 0; k < reps; ++k) { CK(cudaMemcpy((void *)dprm, &prm, sizeof prm, cudaMemcpyHostToDevice)); CK(cudaEventRecord(a0)); OX(optixLaunch(pipe, 0, dprm, sizeof prm, &sbt, nthreads, 1, 1)); CK(cudaEventRecord(a1)); CK(cudaEventSynchronize(a1)); float t; cudaEventElapsedTime(&t, a0, a1); ms.push_back(t); }
    std::vector<Out> h(NB); CK(cudaMemcpy(h.data(), dout, (size_t)NB * sizeof(Out), cudaMemcpyDeviceToHost)); unsigned long long hits = 0, hs = 0; for (auto &o : h) { hits += o.hit; hs += o.prim + o.tbits; }
    fprintf(jf, "{\"mode\": \"%s\", \"M\": %ld, \"T\": %ld, \"big\": %u, \"active\": %u, \"spread\": %u, \"threads\": %u, \"bytes\": %zu, \"gas_build_ms\": %.5f, \"ias_build_ms\": %.5f, \"hit_frac\": %.5f, \"checksum\": %llu, \"ms\": [", mode.c_str(), M, T, NB, A, prm.spread, nthreads, bytes, gas_ms, ias_ms, (double)hits / NB, hs);
    for (size_t k = 0; k < ms.size(); ++k) fprintf(jf, "%s%.5f", k ? "," : "", ms[k]); fprintf(jf, "]}\n"); fclose(jf); printf("done\n"); return 0;
}
