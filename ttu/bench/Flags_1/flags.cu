// Flags_1 host (OptiX only, no TTU code). Predictions: model/predictions_flags.md.
//   flags PTX OUT.json key=value ...
//   scene=layers:L[:alt] | ias:K:A|C|D|V|L<n>|R<n>|E|Q (raymiss=1: rays in the boxes, outside the triangle) | soup:N      rflags=<OptixRayFlags bits> rmask=<ray visibility mask> ah=0|1 (any hit program in the hit group) gopaque=0|1 (build time DISABLE_ANYHIT)
//   nrep=1..3 (traces per raygen in the probe launches) probe=N (probe launches of 32 rays, default 100) big=NB (rays of the big launches, default 262144) nbig=N (timed big launches, default 20)
//   tmax=R (soup: ray tmax) bigonly=1 (skip the probe launches, 2 big launches only: for ncu)
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
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include "flags_params.h"
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(2); } } while (0)
#define OX(x) do { OptixResult r_ = (x); if (r_ != OPTIX_SUCCESS) { fprintf(stderr, "%s:%d %s: %d\n", __FILE__, __LINE__, #x, (int)r_); exit(2); } } while (0)
struct Tri { float3 a, b, c; };
static std::string read_file(const char *p) { std::ifstream f(p); if (!f) { fprintf(stderr, "cannot read %s\n", p); exit(2); } std::stringstream s; s << f.rdbuf(); return s.str(); }

static OptixTraversableHandle build_gas(OptixDeviceContext ctx, const std::vector<Tri> &v, unsigned gflag, CUdeviceptr *gas_out)
{
    CUdeviceptr dv; CK(cudaMalloc((void **)&dv, v.size() * sizeof(Tri))); CK(cudaMemcpy((void *)dv, v.data(), v.size() * sizeof(Tri), cudaMemcpyHostToDevice));
    OptixBuildInput bi{}; bi.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    bi.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3; bi.triangleArray.numVertices = (unsigned)v.size() * 3; bi.triangleArray.vertexBuffers = &dv;
    bi.triangleArray.flags = &gflag; bi.triangleArray.numSbtRecords = 1;
    OptixAccelBuildOptions ao{}; ao.buildFlags = OPTIX_BUILD_FLAG_NONE; ao.operation = OPTIX_BUILD_OPERATION_BUILD;
    OptixAccelBufferSizes bs{}; OX(optixAccelComputeMemoryUsage(ctx, &ao, &bi, 1, &bs));
    CUdeviceptr tmp, gas; CK(cudaMalloc((void **)&tmp, bs.tempSizeInBytes)); CK(cudaMalloc((void **)&gas, bs.outputSizeInBytes));
    OptixTraversableHandle h; OX(optixAccelBuild(ctx, 0, &ao, &bi, 1, tmp, bs.tempSizeInBytes, gas, bs.outputSizeInBytes, &h, nullptr, 0)); CK(cudaDeviceSynchronize()); CK(cudaFree((void *)tmp));
    *gas_out = gas; return h;
}

int main(int argc, char **argv)
{
    if (argc < 3) { fprintf(stderr, "usage\n"); return 2; }
    std::map<std::string, std::string> kv; for (int a = 3; a < argc; ++a) { std::string s = argv[a]; const size_t e = s.find('='); if (e == std::string::npos) { fprintf(stderr, "bad arg %s\n", argv[a]); return 2; } kv[s.substr(0, e)] = s.substr(e + 1); }
    auto geti = [&](const char *k, long long d) { return kv.count(k) ? atoll(kv[k].c_str()) : d; };
    auto getf = [&](const char *k, double d) { return kv.count(k) ? atof(kv[k].c_str()) : d; };
    const std::string scene = kv.count("scene") ? kv["scene"] : "layers:64";
    const unsigned rflags = (unsigned)geti("rflags", 0), rmask = (unsigned)geti("rmask", 255), nrep = (unsigned)geti("nrep", 1);
    const int ah = (int)geti("ah", 0), gopaque = (int)geti("gopaque", 0), nprobe = (int)geti("probe", 100), nbigl = (int)geti("nbig", 20), bigonly = (int)geti("bigonly", 0);
    const unsigned NB = (unsigned)geti("big", 262144); const float tmaxv = (float)getf("tmax", 1e16);
    CK(cudaFree(0)); OX(optixInit()); OptixDeviceContextOptions co{}; OptixDeviceContext ctx; OX(optixDeviceContextCreate(0, &co, &ctx));
    const unsigned gflag = gopaque ? OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT : OPTIX_GEOMETRY_FLAG_NONE;
    std::mt19937 g(1234); std::uniform_real_distribution<float> U(-1.f, 1.f);
    std::vector<Tri> v; OptixTraversableHandle root = 0; CUdeviceptr gas = 0; bool instancing = false;
    int nearest_prim = -1; std::vector<Ray8> warp, big;
    unsigned NW = 32;
    char kind[16] = ""; int L = 0, K = 0; long soupn = 0; char mode[8] = "";
    if (sscanf(scene.c_str(), "layers:%d", &L) == 1) {   // L large triangles in parallel planes z = 0.1 rank, storage order random; ':alt' flips the winding of odd ranks
        const bool alt = scene.find(":alt") != std::string::npos; strcpy(kind, "layers");
        std::vector<int> perm(L); std::iota(perm.begin(), perm.end(), 0); std::shuffle(perm.begin(), perm.end(), g);
        for (int k = 0; k < L; ++k) { const float z = 0.1f * perm[k]; Tri t{{-4.f, -4.f, z}, {4.f, -4.f, z}, {0.f, 4.f, z}}; if (alt && (perm[k] & 1)) std::swap(t.b, t.c); v.push_back(t); if (perm[k] == 0) nearest_prim = k; }
        root = build_gas(ctx, v, gflag, &gas);
        for (unsigned i = 0; i < NB; ++i) { Ray8 r{{0.5f * U(g), 0.5f * U(g), -1.f}, 0.f, {0.f, 0.f, 1.f}, tmaxv}; big.push_back(r); }
        warp.assign(NW, Ray8{{0.1f, 0.05f, -1.f}, 0.f, {0.f, 0.f, 1.f}, tmaxv});
    } else if (sscanf(scene.c_str(), "ias:%d:%7s", &K, mode) == 2) {   // K instances of one big triangle at z = 2k; A: masks 1 (k < K-1), 2 (last); D: all masks 2; C: the last instance only (mask 2)
        strcpy(kind, "ias"); instancing = true;
        v.push_back(Tri{{-4.f, -4.f, 0.f}, {4.f, -4.f, 0.f}, {0.f, 4.f, 0.f}});
        const OptixTraversableHandle hg = build_gas(ctx, v, gflag, &gas);
        std::vector<OptixInstance> inst; const int k0 = (!strcmp(mode, "C")) ? K - 1 : 0;
        const int modeN = atoi(mode + 1);   // V, L<n>, R<n>, E, Q: lone masked / removed instances (masklone predictions): every n-th (k % n == n/2), E: odd masked, Q: even only
        const char mc = mode[0];
        for (int k = k0; k < K; ++k) {
            if (mc == 'R' && k % modeN == modeN / 2) continue;
            if (mc == 'Q' && (k & 1)) continue;
            OptixInstance I{}; const float m[12] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 2.f * k}; memcpy(I.transform, m, 48);
            I.instanceId = 100 + k; I.sbtOffset = 0; I.flags = OPTIX_INSTANCE_FLAG_NONE; I.traversableHandle = hg;
            if (mc == 'A') I.visibilityMask = (k < K - 1) ? 1 : 2;
            else if (mc == 'V' || mc == 'R' || mc == 'Q') I.visibilityMask = 1;
            else if (mc == 'L') I.visibilityMask = (k % modeN == modeN / 2) ? 2 : 1;
            else if (mc == 'E') I.visibilityMask = (k & 1) ? 2 : 1;
            else I.visibilityMask = 2;   // C, D
            inst.push_back(I);
        }
        CUdeviceptr di; CK(cudaMalloc((void **)&di, inst.size() * sizeof(OptixInstance))); CK(cudaMemcpy((void *)di, inst.data(), inst.size() * sizeof(OptixInstance), cudaMemcpyHostToDevice));
        OptixBuildInput ib{}; ib.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES; ib.instanceArray.instances = di; ib.instanceArray.numInstances = (unsigned)inst.size();
        OptixAccelBuildOptions ao{}; ao.buildFlags = OPTIX_BUILD_FLAG_NONE; ao.operation = OPTIX_BUILD_OPERATION_BUILD;
        OptixAccelBufferSizes is{}; OX(optixAccelComputeMemoryUsage(ctx, &ao, &ib, 1, &is));
        CUdeviceptr itmp, ias; CK(cudaMalloc((void **)&itmp, is.tempSizeInBytes)); CK(cudaMalloc((void **)&ias, is.outputSizeInBytes));
        OX(optixAccelBuild(ctx, 0, &ao, &ib, 1, itmp, is.tempSizeInBytes, ias, is.outputSizeInBytes, &root, nullptr, 0)); CK(cudaDeviceSynchronize());
        const bool rmiss = geti("raymiss", 0) != 0;   // origin near (3.5, 3.5): inside every instance's box, outside the triangle
        for (unsigned i = 0; i < NB; ++i) { Ray8 r{{rmiss ? 3.5f + 0.3f * U(g) : 0.5f * U(g), rmiss ? 3.5f + 0.3f * U(g) : 0.5f * U(g), -1.f}, 0.f, {0.f, 0.f, 1.f}, tmaxv}; big.push_back(r); }
        warp.assign(NW, Ray8{{rmiss ? 3.5f : 0.1f, rmiss ? 3.5f : 0.05f, -1.f}, 0.f, {0.f, 0.f, 1.f}, tmaxv});
    } else if (sscanf(scene.c_str(), "soup:%ld", &soupn) == 1) {   // the soup of setup3 / BVH_4 (same generator); random rays, tmax = tmaxv
        strcpy(kind, "soup");
        const float e = 3.f / std::cbrt((float)soupn);
        for (long i = 0; i < soupn; ++i) { const float3 c{U(g), U(g), U(g)}; auto pt = [&]() { return float3{c.x + 0.5f * e * U(g), c.y + 0.5f * e * U(g), c.z + 0.5f * e * U(g)}; }; v.push_back({pt(), pt(), pt()}); }
        root = build_gas(ctx, v, gflag, &gas);
        std::mt19937 gr(2026); std::normal_distribution<float> N(0.f, 1.f);
        for (unsigned i = 0; i < NB; ++i) { float3 d{N(gr), N(gr), N(gr)}; const float l = std::sqrt(d.x * d.x + d.y * d.y + d.z * d.z); big.push_back(Ray8{{U(gr), U(gr), U(gr)}, 0.f, {d.x / l, d.y / l, d.z / l}, tmaxv}); }
        warp.assign(big.begin(), big.begin() + NW);
    } else { fprintf(stderr, "scene?\n"); return 2; }
    OptixModuleCompileOptions mo{}; OptixPipelineCompileOptions po{};
    po.traversableGraphFlags = instancing ? OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING : OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    po.numPayloadValues = 4; po.numAttributeValues = 2; po.pipelineLaunchParamsVariableName = "params"; po.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
    std::string ptx = read_file(argv[1]); char log[4096]; size_t ls = sizeof log; OptixModule m; OX(optixModuleCreate(ctx, &mo, &po, ptx.c_str(), ptx.size(), log, &ls, &m));
    OptixProgramGroupOptions go{}; OptixProgramGroupDesc d[3] = {};
    d[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN; d[0].raygen.module = m; d[0].raygen.entryFunctionName = "__raygen__rg";
    d[1].kind = OPTIX_PROGRAM_GROUP_KIND_MISS; d[1].miss.module = m; d[1].miss.entryFunctionName = "__miss__ms";
    d[2].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP; d[2].hitgroup.moduleCH = m; d[2].hitgroup.entryFunctionNameCH = "__closesthit__ch";
    if (ah) { d[2].hitgroup.moduleAH = m; d[2].hitgroup.entryFunctionNameAH = "__anyhit__ah"; }
    OptixProgramGroup pg[3]; ls = sizeof log; OX(optixProgramGroupCreate(ctx, d, 3, &go, log, &ls, pg));
    OptixPipelineLinkOptions lo{}; lo.maxTraceDepth = 1; OptixPipeline pipe; ls = sizeof log; OX(optixPipelineCreate(ctx, &po, &lo, pg, 3, log, &ls, &pipe));
    struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) Rec { char h[OPTIX_SBT_RECORD_HEADER_SIZE]; };
    Rec rec[3]; for (int k = 0; k < 3; ++k) OX(optixSbtRecordPackHeader(pg[k], &rec[k]));
    CUdeviceptr drec; CK(cudaMalloc((void **)&drec, sizeof rec)); CK(cudaMemcpy((void *)drec, rec, sizeof rec, cudaMemcpyHostToDevice));
    OptixShaderBindingTable sbt{}; sbt.raygenRecord = drec; sbt.missRecordBase = drec + sizeof(Rec); sbt.missRecordStrideInBytes = sizeof(Rec); sbt.missRecordCount = 1;
    sbt.hitgroupRecordBase = drec + 2 * sizeof(Rec); sbt.hitgroupRecordStrideInBytes = sizeof(Rec); sbt.hitgroupRecordCount = 1;
    FILE *jf = fopen(argv[2], "wx"); if (!jf) { fprintf(stderr, "refuse to overwrite or cannot write %s\n", argv[2]); return 2; }
    fprintf(jf, "{\"scene\": \"%s\", \"rflags\": %u, \"rmask\": %u, \"ah\": %d, \"gopaque\": %d, \"nrep\": %u, \"tmax\": %g, \"nearest_prim\": %d, \"ntris\": %zu,\n", scene.c_str(), rflags, rmask, ah, gopaque, nrep, tmaxv, nearest_prim, v.size());
    auto launch_set = [&](const std::vector<Ray8> &rays, bool probe, unsigned nr_, std::vector<Out> &res, std::vector<unsigned long long> *clk, std::vector<float> *ms, int reps) {
        Params prm{}; prm.handle = root; prm.n = (unsigned)rays.size(); prm.flags = rflags; prm.mask = rmask; prm.nrep = nr_;
        CK(cudaMalloc((void **)&prm.rays, rays.size() * sizeof(Ray8))); CK(cudaMemcpy((void *)prm.rays, rays.data(), rays.size() * sizeof(Ray8), cudaMemcpyHostToDevice));
        CK(cudaMalloc((void **)&prm.out, rays.size() * sizeof(Out)));
        if (probe) CK(cudaMalloc((void **)&prm.clk, rays.size() * 8 * 8));
        CUdeviceptr dprm; CK(cudaMalloc((void **)&dprm, sizeof prm)); CK(cudaMemcpy((void *)dprm, &prm, sizeof prm, cudaMemcpyHostToDevice));
        OX(optixLaunch(pipe, 0, dprm, sizeof prm, &sbt, (unsigned)rays.size(), 1, 1)); CK(cudaDeviceSynchronize());   // reference / warm up launch
        res.resize(rays.size()); CK(cudaMemcpy(res.data(), prm.out, rays.size() * sizeof(Out), cudaMemcpyDeviceToHost));
        cudaEvent_t a0, a1; CK(cudaEventCreate(&a0)); CK(cudaEventCreate(&a1));
        for (int k = 0; k < reps; ++k) {
            CK(cudaEventRecord(a0)); OX(optixLaunch(pipe, 0, dprm, sizeof prm, &sbt, (unsigned)rays.size(), 1, 1)); CK(cudaEventRecord(a1)); CK(cudaEventSynchronize(a1));
            float t; CK(cudaEventElapsedTime(&t, a0, a1)); if (ms) ms->push_back(t);
            if (probe) { std::vector<unsigned long long> c(rays.size() * 8); CK(cudaMemcpy(c.data(), prm.clk, c.size() * 8, cudaMemcpyDeviceToHost)); clk->insert(clk->end(), c.begin(), c.end()); }
        }
        CK(cudaFree((void *)prm.rays)); CK(cudaFree((void *)prm.out)); if (probe) CK(cudaFree((void *)prm.clk)); CK(cudaFree((void *)dprm));
    };
    auto dump_out = [&](const char *name, const std::vector<Out> &o) {
        fprintf(jf, " \"%s\": [", name);
        for (size_t i = 0; i < o.size(); ++i) fprintf(jf, "%s[%u,%u,%u,%u,%u,%u]", i ? "," : "", o[i].hit, o[i].tbits, o[i].prim, o[i].ah, o[i].front, o[i].inst);
        fprintf(jf, "],\n");
    };
    if (!bigonly) {
        std::vector<Out> res; std::vector<unsigned long long> clk; std::vector<float> ms;
        launch_set(warp, true, nrep, res, &clk, &ms, nprobe);
        dump_out("warp_out", res);
        fprintf(jf, " \"probe_launches\": %d, \"probe_clk\": [", nprobe);
        for (int r = 0; r < nprobe; ++r) { fprintf(jf, "%s[", r ? ",\n  " : ""); for (size_t k = 0; k < (size_t)NW * 8; ++k) fprintf(jf, "%s%llu", k ? "," : "", clk[(size_t)r * NW * 8 + k]); fprintf(jf, "]"); }
        fprintf(jf, "],\n");
    }
    {
        std::vector<Out> res; std::vector<float> ms;
        launch_set(big, false, 1, res, nullptr, &ms, bigonly ? 1 : nbigl);
        double hit = 0, ah_ = 0, nearest = 0; for (const Out &o : res) { hit += o.hit; ah_ += o.ah; nearest += (o.hit && (int)o.prim == nearest_prim); }
        fprintf(jf, " \"big_n\": %zu, \"big_hit_frac\": %.6f, \"big_ah_mean\": %.4f, \"big_nearest_frac\": %.6f, \"big_ms\": [", res.size(), hit / res.size(), ah_ / res.size(), nearest / res.size());
        for (size_t k = 0; k < ms.size(); ++k) fprintf(jf, "%s%.6f", k ? "," : "", ms[k]);
        fprintf(jf, "]}\n");
    }
    fclose(jf); printf("done %s\n", argv[2]);
    return 0;
}
