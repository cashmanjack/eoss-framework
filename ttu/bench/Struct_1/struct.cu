// Struct_1 host (OptiX only, no TTU code). Predictions: model/predictions_struct_*.md.
//   struct PTX OUT.json key=value ...
//   scene=soup:N | cust:K:N:R   (K u uniform cube, c 200 clusters sigma 0.03, t thin rods half widths (30R,R,R) clipped, o uniform shifted +1000; R half width)
//   bf=none|trace|build upd=0|1 comp=0|1      build flags of the GAS (compaction = ALLOW_COMPACTION then optixAccelCompact)
//   inst=none|id1|r90_1|r37_1|id2|r90_2  any=1 (pipeline graph flag ALLOW_ANY with the one level instance)
//   rays=rand|pts tmax=R big=NB nbig=N probe=N bigonly=1 dump=PREFIX nolaunch=1 dumpres=FILE
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
#include "struct_params.h"
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(2); } } while (0)
#define OX(x) do { OptixResult r_ = (x); if (r_ != OPTIX_SUCCESS) { fprintf(stderr, "%s:%d %s: %d\n", __FILE__, __LINE__, #x, (int)r_); exit(2); } } while (0)
struct Tri { float3 a, b, c; };
static std::string read_file(const char *p) { std::ifstream f(p); if (!f) { fprintf(stderr, "cannot read %s\n", p); exit(2); } std::stringstream s; s << f.rdbuf(); return s.str(); }
static OptixDeviceContext ctx;
static OptixTraversableHandle build_ias(OptixTraversableHandle child, const float *m, unsigned id, CUdeviceptr *out)
{
    OptixInstance I{}; memcpy(I.transform, m, 48); I.instanceId = id; I.visibilityMask = 255; I.flags = OPTIX_INSTANCE_FLAG_NONE; I.traversableHandle = child;
    CUdeviceptr di; CK(cudaMalloc((void **)&di, sizeof I)); CK(cudaMemcpy((void *)di, &I, sizeof I, cudaMemcpyHostToDevice));
    OptixBuildInput ib{}; ib.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES; ib.instanceArray.instances = di; ib.instanceArray.numInstances = 1;
    OptixAccelBuildOptions ao{}; ao.buildFlags = OPTIX_BUILD_FLAG_NONE; ao.operation = OPTIX_BUILD_OPERATION_BUILD;
    OptixAccelBufferSizes is{}; OX(optixAccelComputeMemoryUsage(ctx, &ao, &ib, 1, &is));
    CUdeviceptr tmp, ias; CK(cudaMalloc((void **)&tmp, is.tempSizeInBytes)); CK(cudaMalloc((void **)&ias, is.outputSizeInBytes));
    OptixTraversableHandle h; OX(optixAccelBuild(ctx, 0, &ao, &ib, 1, tmp, is.tempSizeInBytes, ias, is.outputSizeInBytes, &h, nullptr, 0)); CK(cudaDeviceSynchronize()); *out = ias; return h;
}
int main(int argc, char **argv)
{
    if (argc < 3) { fprintf(stderr, "usage\n"); return 2; }
    std::map<std::string, std::string> kv; for (int a = 3; a < argc; ++a) { std::string s = argv[a]; const size_t e = s.find('='); if (e == std::string::npos) { fprintf(stderr, "bad arg %s\n", argv[a]); return 2; } kv[s.substr(0, e)] = s.substr(e + 1); }
    auto geti = [&](const char *k, long long d) { return kv.count(k) ? atoll(kv[k].c_str()) : d; };
    auto gets = [&](const char *k, const char *d) { return kv.count(k) ? kv[k] : std::string(d); };
    const std::string scene = gets("scene", "soup:100000"), bf = gets("bf", "none"), inst = gets("inst", "none"), rk = gets("rays", "rand");
    const int upd = (int)geti("upd", 0), comp = (int)geti("comp", 0), any = (int)geti("any", 0), nbigl = (int)geti("nbig", 20), nprobe = (int)geti("probe", 0), bigonly = (int)geti("bigonly", 0);
    const unsigned NB = (unsigned)geti("big", 262144); const float tmaxv = kv.count("tmax") ? (float)atof(kv["tmax"].c_str()) : (rk == "pts" ? 1e-16f : 1e16f);
    CK(cudaFree(0)); OX(optixInit()); OptixDeviceContextOptions co{}; OX(optixDeviceContextCreate(0, &co, &ctx));
    std::mt19937 g(1234); std::uniform_real_distribution<float> U(-1.f, 1.f); std::normal_distribution<float> Nn(0.f, 1.f);
    std::vector<Tri> v; std::vector<float> bx; bool cust = false; long N = 0; char kc = 'u'; float R = 0;
    long soupn = 0;
    if (sscanf(scene.c_str(), "soup:%ld", &soupn) == 1) {
        const float e = 3.f / std::cbrt((float)soupn);
        for (long i = 0; i < soupn; ++i) { const float3 c{U(g), U(g), U(g)}; auto pt = [&]() { return float3{c.x + 0.5f * e * U(g), c.y + 0.5f * e * U(g), c.z + 0.5f * e * U(g)}; }; v.push_back({pt(), pt(), pt()}); }
        N = soupn;
    } else if (sscanf(scene.c_str(), "cust:%c:%ld:%f", &kc, &N, &R) == 3) {
        cust = true; std::vector<float3> cl(200); for (auto &c : cl) c = {U(g), U(g), U(g)};
        for (long i = 0; i < N; ++i) {
            float3 c{U(g), U(g), U(g)}; float h[3] = {R, R, R};
            if (kc == 'c') { const float3 q = cl[i % 200]; c = {q.x + 0.03f * Nn(g), q.y + 0.03f * Nn(g), q.z + 0.03f * Nn(g)}; }
            if (kc == 't') { const int ax = (int)(g() % 3); h[ax] = 30.f * R; }
            if (kc == 'o') { c.x += 1000.f; c.y += 1000.f; c.z += 1000.f; }
            bx.push_back(c.x - h[0]); bx.push_back(c.y - h[1]); bx.push_back(c.z - h[2]); bx.push_back(c.x + h[0]); bx.push_back(c.y + h[1]); bx.push_back(c.z + h[2]);
        }
    } else { fprintf(stderr, "scene?\n"); return 2; }
    unsigned bflags = OPTIX_BUILD_FLAG_NONE; if (bf == "trace") bflags |= OPTIX_BUILD_FLAG_PREFER_FAST_TRACE; else if (bf == "build") bflags |= OPTIX_BUILD_FLAG_PREFER_FAST_BUILD; else if (bf != "none") { fprintf(stderr, "bf?\n"); return 2; }
    if (upd) bflags |= OPTIX_BUILD_FLAG_ALLOW_UPDATE; if (comp) bflags |= OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    unsigned gflag = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT; CUdeviceptr dv = 0, dbx = 0; OptixBuildInput bi{};
    if (cust) {
        CK(cudaMalloc((void **)&dbx, bx.size() * 4)); CK(cudaMemcpy((void *)dbx, bx.data(), bx.size() * 4, cudaMemcpyHostToDevice));
        bi.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES; bi.customPrimitiveArray.aabbBuffers = &dbx; bi.customPrimitiveArray.numPrimitives = (unsigned)N; bi.customPrimitiveArray.flags = &gflag; bi.customPrimitiveArray.numSbtRecords = 1;
    } else {
        CK(cudaMalloc((void **)&dv, v.size() * sizeof(Tri))); CK(cudaMemcpy((void *)dv, v.data(), v.size() * sizeof(Tri), cudaMemcpyHostToDevice));
        bi.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES; bi.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3; bi.triangleArray.numVertices = (unsigned)v.size() * 3; bi.triangleArray.vertexBuffers = &dv; bi.triangleArray.flags = &gflag; bi.triangleArray.numSbtRecords = 1;
    }
    OptixAccelBuildOptions ao{}; ao.buildFlags = bflags; ao.operation = OPTIX_BUILD_OPERATION_BUILD;
    OptixAccelBufferSizes bs{}; OX(optixAccelComputeMemoryUsage(ctx, &ao, &bi, 1, &bs));
    CUdeviceptr tmp, gas, csz; CK(cudaMalloc((void **)&tmp, bs.tempSizeInBytes)); CK(cudaMalloc((void **)&gas, bs.outputSizeInBytes)); CK(cudaMalloc((void **)&csz, 8));
    OptixAccelEmitDesc em{}; em.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE; em.result = csz;
    OptixTraversableHandle hg; OX(optixAccelBuild(ctx, 0, &ao, &bi, 1, tmp, bs.tempSizeInBytes, gas, bs.outputSizeInBytes, &hg, comp ? &em : nullptr, comp ? 1 : 0)); CK(cudaDeviceSynchronize());
    size_t gas_final = bs.outputSizeInBytes; unsigned long long compsz = 0;
    if (comp) {
        CK(cudaMemcpy(&compsz, (void *)csz, 8, cudaMemcpyDeviceToHost)); CUdeviceptr g2; CK(cudaMalloc((void **)&g2, compsz));
        OX(optixAccelCompact(ctx, 0, hg, g2, compsz, &hg)); CK(cudaDeviceSynchronize()); gas = g2; gas_final = compsz;
    }
    OptixTraversableHandle root = hg; CUdeviceptr ias1 = 0, ias2 = 0;
    const float I3[12] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
    const float RZ90[12] = {0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0};      // rotation about z by 90 degrees (maps the cube onto itself)
    const float RX90[12] = {1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0};      // rotation about x by 90 degrees
    const float c37 = std::cos(0.6458f), s37 = std::sin(0.6458f); const float RZ37[12] = {c37, -s37, 0, 0.05f, s37, c37, 0, -0.03f, 0, 0, 1, 0.02f};
    if (inst == "id1") root = build_ias(hg, I3, 7, &ias1);
    else if (inst == "r90_1") root = build_ias(hg, RZ90, 7, &ias1);
    else if (inst == "r37_1") root = build_ias(hg, RZ37, 7, &ias1);
    else if (inst == "id2") { const OptixTraversableHandle in = build_ias(hg, I3, 5, &ias1); root = build_ias(in, I3, 7, &ias2); }
    else if (inst == "r90_2") { const OptixTraversableHandle in = build_ias(hg, RX90, 5, &ias1); root = build_ias(in, RZ90, 7, &ias2); }
    else if (inst != "none") { fprintf(stderr, "inst?\n"); return 2; }
    const int levels = inst == "none" ? 0 : (inst.back() == '2' ? 2 : 1);
    // rays (host generated, fixed seed)
    std::vector<Ray8> big; { std::mt19937 gr(2026); const float off = (cust && kc == 'o') ? 1000.f : 0.f;
        for (unsigned i = 0; i < NB; ++i) { float3 d{Nn(gr), Nn(gr), Nn(gr)}; const float l = std::sqrt(d.x * d.x + d.y * d.y + d.z * d.z); const float3 o{U(gr) + off, U(gr) + off, U(gr) + off};
            if (rk == "pts") d = {0.f, 0.f, 1.f}; else d = {d.x / l, d.y / l, d.z / l}; big.push_back(Ray8{{o.x, o.y, o.z}, 0.f, {d.x, d.y, d.z}, tmaxv}); } }
    std::vector<char> hb(std::min<size_t>(gas_final, 0x100)); CK(cudaMemcpy(hb.data(), (void *)gas, hb.size(), cudaMemcpyDeviceToHost));
    FILE *jf = fopen(argv[2], "wx"); if (!jf) { fprintf(stderr, "refuse to overwrite or cannot write %s\n", argv[2]); return 2; }
    fprintf(jf, "{\"scene\": \"%s\", \"bf\": \"%s\", \"upd\": %d, \"comp\": %d, \"inst\": \"%s\", \"any\": %d, \"rays\": \"%s\", \"tmax\": %g, \"levels\": %d, \"buffer_bytes\": %zu, \"temp_bytes\": %zu, \"final_bytes\": %zu, \"compacted_bytes\": %llu, \"hdr\": [", scene.c_str(), bf.c_str(), upd, comp, inst.c_str(), any, rk.c_str(), tmaxv, levels, (size_t)bs.outputSizeInBytes, (size_t)bs.tempSizeInBytes, gas_final, compsz);
    for (size_t k = 0; k + 8 <= hb.size(); k += 8) { unsigned long long w; memcpy(&w, &hb[k], 8); fprintf(jf, "%s%llu", k ? "," : "", w); } fprintf(jf, "]");
    if (kv.count("dump")) {
        const std::string pre = kv["dump"]; auto w = [&](const std::string &e, const void *p, size_t n) { FILE *f = fopen((pre + e).c_str(), "wbx"); if (!f) { fprintf(stderr, "refuse to overwrite %s\n", (pre + e).c_str()); exit(2); } fwrite(p, 1, n, f); fclose(f); };
        std::vector<char> b(gas_final); CK(cudaMemcpy(b.data(), (void *)gas, b.size(), cudaMemcpyDeviceToHost)); w(".gas", b.data(), b.size());
        if (cust) w(".aabb", bx.data(), bx.size() * 4); else w(".tris", v.data(), v.size() * sizeof(Tri)); w(".rays", big.data(), big.size() * sizeof(Ray8));
    }
    if (geti("nolaunch", 0)) { fprintf(jf, "}\n"); fclose(jf); printf("done (no launch) %s\n", argv[2]); return 0; }
    OptixModuleCompileOptions mo{}; OptixPipelineCompileOptions po{};
    po.traversableGraphFlags = levels == 0 ? OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS : (levels == 1 && !any) ? OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING : OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
    po.numPayloadValues = 4; po.numAttributeValues = 2; po.pipelineLaunchParamsVariableName = "params"; po.usesPrimitiveTypeFlags = cust ? OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM : OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
    std::string ptx = read_file(argv[1]); char log[4096]; size_t ls = sizeof log; OptixModule m; OX(optixModuleCreate(ctx, &mo, &po, ptx.c_str(), ptx.size(), log, &ls, &m));
    OptixProgramGroupOptions go{}; OptixProgramGroupDesc d[3] = {};
    d[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN; d[0].raygen.module = m; d[0].raygen.entryFunctionName = "__raygen__rg";
    d[1].kind = OPTIX_PROGRAM_GROUP_KIND_MISS; d[1].miss.module = m; d[1].miss.entryFunctionName = "__miss__ms";
    d[2].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP; d[2].hitgroup.moduleCH = m; d[2].hitgroup.entryFunctionNameCH = "__closesthit__ch";
    if (cust) { d[2].hitgroup.moduleIS = m; d[2].hitgroup.entryFunctionNameIS = "__intersection__cnt"; }
    OptixProgramGroup pg[3]; ls = sizeof log; OX(optixProgramGroupCreate(ctx, d, 3, &go, log, &ls, pg));
    OptixPipelineLinkOptions lo{}; lo.maxTraceDepth = 1; OptixPipeline pipe; ls = sizeof log; OX(optixPipelineCreate(ctx, &po, &lo, pg, 3, log, &ls, &pipe));
    OX(optixPipelineSetStackSize(pipe, 0, 0, 2048, levels + 1));
    struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) Rec { char h[OPTIX_SBT_RECORD_HEADER_SIZE]; };
    Rec rec[3]; for (int k = 0; k < 3; ++k) OX(optixSbtRecordPackHeader(pg[k], &rec[k]));
    CUdeviceptr drec; CK(cudaMalloc((void **)&drec, sizeof rec)); CK(cudaMemcpy((void *)drec, rec, sizeof rec, cudaMemcpyHostToDevice));
    OptixShaderBindingTable sbt{}; sbt.raygenRecord = drec; sbt.missRecordBase = drec + sizeof(Rec); sbt.missRecordStrideInBytes = sizeof(Rec); sbt.missRecordCount = 1;
    sbt.hitgroupRecordBase = drec + 2 * sizeof(Rec); sbt.hitgroupRecordStrideInBytes = sizeof(Rec); sbt.hitgroupRecordCount = 1;
    Ray8 *drays; Out *dout; unsigned long long *dclk; CK(cudaMalloc((void **)&drays, (size_t)NB * sizeof(Ray8))); CK(cudaMalloc((void **)&dout, (size_t)NB * sizeof(Out))); CK(cudaMalloc((void **)&dclk, 32 * 8 * 8));
    CK(cudaMemcpy(drays, big.data(), (size_t)NB * sizeof(Ray8), cudaMemcpyHostToDevice));
    Params prm{}; prm.rays = drays; prm.out = dout; prm.handle = root; prm.n = NB; prm.nrep = 1; prm.clk = nullptr;
    CUdeviceptr dprm; CK(cudaMalloc((void **)&dprm, sizeof prm));
    auto launch = [&](unsigned n) { CK(cudaMemcpy((void *)dprm, &prm, sizeof prm, cudaMemcpyHostToDevice)); OX(optixLaunch(pipe, 0, dprm, sizeof prm, &sbt, n, 1, 1)); };
    if (nprobe > 0 && !bigonly) {
        fprintf(jf, ", \"probe_launches\": %d, \"probe_clk\": [", nprobe); prm.clk = dclk; prm.nrep = 3; bool first = true;
        for (int k = 0; k < nprobe; ++k) { launch(32); CK(cudaDeviceSynchronize()); std::vector<unsigned long long> c(32 * 8); CK(cudaMemcpy(c.data(), dclk, 32 * 8 * 8, cudaMemcpyDeviceToHost)); for (auto x : c) { fprintf(jf, "%s%llu", first ? "" : ",", x); first = false; } }
        fprintf(jf, "]"); prm.clk = nullptr; prm.nrep = 1;
    }
    launch(NB); CK(cudaDeviceSynchronize());   // warm up
    std::vector<float> ms; cudaEvent_t a0, a1; CK(cudaEventCreate(&a0)); CK(cudaEventCreate(&a1));
    for (int k = 0; k < (bigonly ? 1 : nbigl); ++k) { CK(cudaMemcpy((void *)dprm, &prm, sizeof prm, cudaMemcpyHostToDevice)); CK(cudaEventRecord(a0)); OX(optixLaunch(pipe, 0, dprm, sizeof prm, &sbt, NB, 1, 1)); CK(cudaEventRecord(a1)); CK(cudaEventSynchronize(a1)); float t; CK(cudaEventElapsedTime(&t, a0, a1)); ms.push_back(t); }
    fprintf(jf, ", \"big\": %u, \"ms\": [", NB); for (size_t k = 0; k < ms.size(); ++k) fprintf(jf, "%s%.6f", k ? "," : "", ms[k]); fprintf(jf, "]");
    if (kv.count("dumpres")) { std::vector<Out> h(NB); CK(cudaMemcpy(h.data(), dout, (size_t)NB * sizeof(Out), cudaMemcpyDeviceToHost)); FILE *f = fopen(kv["dumpres"].c_str(), "wbx"); if (!f) { fprintf(stderr, "refuse to overwrite %s\n", kv["dumpres"].c_str()); return 2; } fwrite(h.data(), sizeof(Out), NB, f); fclose(f); }
    fprintf(jf, "}\n"); fclose(jf); printf("done %s\n", argv[2]);
    return 0;
}
