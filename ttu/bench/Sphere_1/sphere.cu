// Sphere_1 host (OptiX only, no TTU code). Predictions: model/predictions_spheres.md.
//   sphere PTX OUT.json kind=sph|custom|tri0|tri1 N=20000 r=0.04 [verify=BIN nver=1048576] [sat=N reps=10 ncal=5] [warp=ids.txt wreps=12] [ncu=1]
// kind: sph = built in spheres, custom = custom primitives (box + our intersection program), tri0 / tri1 = inscribed icosphere (20 / 80 triangles per sphere).
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
#include "sphere_params.h"
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(2); } } while (0)
#define OX(x) do { OptixResult r_ = (x); if (r_ != OPTIX_SUCCESS) { fprintf(stderr, "%s:%d %s: %d\n", __FILE__, __LINE__, #x, (int)r_); exit(2); } } while (0)
struct V3 { float x, y, z; };
static V3 nrm(V3 a, float r) { const float l = r / std::sqrt(a.x * a.x + a.y * a.y + a.z * a.z); return {a.x * l, a.y * l, a.z * l}; }
static V3 mid(V3 a, V3 b) { return {(a.x + b.x) / 2, (a.y + b.y) / 2, (a.z + b.z) / 2}; }
static std::string read_file(const char *p) { std::ifstream f(p); if (!f) { fprintf(stderr, "cannot read %s\n", p); exit(2); } std::stringstream s; s << f.rdbuf(); return s.str(); }
struct Tri { V3 a, b, c; };
static std::vector<Tri> icosphere(int sub)   // unit sphere (vertices on it), 20 * 4^sub triangles
{
    const float t = (1.f + std::sqrt(5.f)) / 2.f;
    std::vector<V3> v = {{-1, t, 0}, {1, t, 0}, {-1, -t, 0}, {1, -t, 0}, {0, -1, t}, {0, 1, t}, {0, -1, -t}, {0, 1, -t}, {t, 0, -1}, {t, 0, 1}, {-t, 0, -1}, {-t, 0, 1}};
    for (auto &x : v) x = nrm(x, 1.f);
    const int f[20][3] = {{0, 11, 5}, {0, 5, 1}, {0, 1, 7}, {0, 7, 10}, {0, 10, 11}, {1, 5, 9}, {5, 11, 4}, {11, 10, 2}, {10, 7, 6}, {7, 1, 8}, {3, 9, 4}, {3, 4, 2}, {3, 2, 6}, {3, 6, 8}, {3, 8, 9}, {4, 9, 5}, {2, 4, 11}, {6, 2, 10}, {8, 6, 7}, {9, 8, 1}};
    std::vector<Tri> T; for (auto &q : f) T.push_back({v[q[0]], v[q[1]], v[q[2]]});
    for (int s = 0; s < sub; ++s) {
        std::vector<Tri> U;
        for (auto &k : T) { V3 ab = nrm(mid(k.a, k.b), 1.f), bc = nrm(mid(k.b, k.c), 1.f), ca = nrm(mid(k.c, k.a), 1.f); U.push_back({k.a, ab, ca}); U.push_back({k.b, bc, ab}); U.push_back({k.c, ca, bc}); U.push_back({ab, bc, ca}); }
        T = U;
    }
    return T;
}

int main(int argc, char **argv)
{
    if (argc < 3) { fprintf(stderr, "usage\n"); return 2; }
    std::map<std::string, std::string> kv; for (int a = 3; a < argc; ++a) { std::string s = argv[a]; const size_t e = s.find('='); if (e == std::string::npos) { fprintf(stderr, "bad arg %s\n", argv[a]); return 2; } kv[s.substr(0, e)] = s.substr(e + 1); }
    auto I = [&](const char *k, long d) { return kv.count(k) ? atol(kv[k].c_str()) : d; };
    const std::string kind = kv.count("kind") ? kv["kind"] : "sph"; const long N = I("N", 20000); const float R = kv.count("r") ? (float)atof(kv["r"].c_str()) : 0.04f;
    const bool is_sph = kind == "sph", is_cus = kind == "custom", is_tri = kind == "tri0" || kind == "tri1";
    if (!is_sph && !is_cus && !is_tri) { fprintf(stderr, "kind?\n"); return 2; }
    CK(cudaFree(0)); OX(optixInit()); OptixDeviceContextOptions co{}; OptixDeviceContext ctx; OX(optixDeviceContextCreate(0, &co, &ctx));
    std::mt19937 g(4321); std::uniform_real_distribution<float> U(-1.f, 1.f);
    const float off = kv.count("off") ? (float)atof(kv["off"].c_str()) : 0.9f, jitf = kv.count("jit") ? (float)atof(kv["jit"].c_str()) : 0.05f;   // line scene: centre offset and ray jitter, both in units of r (D1a: 0.9 and 0.05; D1b: 0.72 and 0.005)
    const int scene = kv.count("scene") && kv["scene"] == "line" ? 1 : 0;   // line: N spheres along z, each centre 0.9 r off the axis in x and y (signs cycle), the axis rays cross every box and hit no sphere
    std::vector<V3> cen(N); for (long k = 0; k < N; ++k) { if (scene == 0) cen[k] = {U(g), U(g), U(g)}; else cen[k] = {off * R * ((k & 1) ? -1.f : 1.f), off * R * ((k & 2) ? -1.f : 1.f), -0.9f + 1.8f * (k + 0.5f) / N}; }
    V3 *dcen; CK(cudaMalloc((void **)&dcen, N * sizeof(V3))); CK(cudaMemcpy(dcen, cen.data(), N * sizeof(V3), cudaMemcpyHostToDevice));
    unsigned tps = 1; unsigned gflag = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT; OptixBuildInput bi{}; CUdeviceptr dA = 0, dB = 0;
    if (is_sph) {
        float *rad; CK(cudaMalloc((void **)&rad, 4)); CK(cudaMemcpy(rad, &R, 4, cudaMemcpyHostToDevice)); dA = (CUdeviceptr)dcen; dB = (CUdeviceptr)rad;
        bi.type = OPTIX_BUILD_INPUT_TYPE_SPHERES; bi.sphereArray.vertexBuffers = &dA; bi.sphereArray.numVertices = (unsigned)N; bi.sphereArray.radiusBuffers = &dB; bi.sphereArray.singleRadius = 1;
        bi.sphereArray.flags = &gflag; bi.sphereArray.numSbtRecords = 1;
    } else if (is_cus) {
        std::vector<OptixAabb> bb(N); for (long k = 0; k < N; ++k) bb[k] = {cen[k].x - R, cen[k].y - R, cen[k].z - R, cen[k].x + R, cen[k].y + R, cen[k].z + R};
        CK(cudaMalloc((void **)&dA, N * sizeof(OptixAabb))); CK(cudaMemcpy((void *)dA, bb.data(), N * sizeof(OptixAabb), cudaMemcpyHostToDevice));
        bi.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES; bi.customPrimitiveArray.aabbBuffers = &dA; bi.customPrimitiveArray.numPrimitives = (unsigned)N; bi.customPrimitiveArray.flags = &gflag; bi.customPrimitiveArray.numSbtRecords = 1;
    } else {
        const std::vector<Tri> ico = icosphere(kind == "tri1" ? 1 : 0); tps = (unsigned)ico.size(); std::vector<V3> vv; vv.reserve((size_t)N * tps * 3);
        for (long k = 0; k < N; ++k) for (auto &t : ico) for (const V3 *p : {&t.a, &t.b, &t.c}) vv.push_back({cen[k].x + p->x * R, cen[k].y + p->y * R, cen[k].z + p->z * R});
        CK(cudaMalloc((void **)&dA, vv.size() * sizeof(V3))); CK(cudaMemcpy((void *)dA, vv.data(), vv.size() * sizeof(V3), cudaMemcpyHostToDevice));
        bi.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES; bi.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3; bi.triangleArray.numVertices = (unsigned)vv.size(); bi.triangleArray.vertexBuffers = &dA; bi.triangleArray.flags = &gflag; bi.triangleArray.numSbtRecords = 1;
    }
    OptixAccelBuildOptions ao{}; ao.buildFlags = OPTIX_BUILD_FLAG_NONE; ao.operation = OPTIX_BUILD_OPERATION_BUILD;
    OptixAccelBufferSizes bs{}; OX(optixAccelComputeMemoryUsage(ctx, &ao, &bi, 1, &bs));
    CUdeviceptr tmp, gas, cs; CK(cudaMalloc((void **)&tmp, bs.tempSizeInBytes)); CK(cudaMalloc((void **)&gas, bs.outputSizeInBytes)); CK(cudaMalloc((void **)&cs, 8));
    OptixTraversableHandle root; OX(optixAccelBuild(ctx, 0, &ao, &bi, 1, tmp, bs.tempSizeInBytes, gas, bs.outputSizeInBytes, &root, nullptr, 0)); CK(cudaDeviceSynchronize());
    const unsigned long long csz = 0;   // no compaction (the flag would change the build); gas_bytes is the buffer OptiX asked for (an upper bound of the tree)
    OptixModuleCompileOptions mo{}; OptixPipelineCompileOptions po{};
    po.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS; po.numPayloadValues = 4; po.numAttributeValues = 2; po.pipelineLaunchParamsVariableName = "params";
    po.usesPrimitiveTypeFlags = is_sph ? OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE : is_cus ? OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM : OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
    std::string ptx = read_file(argv[1]); char log[4096]; size_t ls = sizeof log; OptixModule m; OX(optixModuleCreate(ctx, &mo, &po, ptx.c_str(), ptx.size(), log, &ls, &m));
    OptixModule bm = nullptr; if (is_sph) { OptixBuiltinISOptions bio{}; bio.builtinISModuleType = OPTIX_PRIMITIVE_TYPE_SPHERE; OX(optixBuiltinISModuleGet(ctx, &mo, &po, &bio, &bm)); }
    OptixProgramGroupOptions go{}; OptixProgramGroupDesc d[3] = {};
    d[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN; d[0].raygen.module = m; d[0].raygen.entryFunctionName = "__raygen__rg";
    d[1].kind = OPTIX_PROGRAM_GROUP_KIND_MISS; d[1].miss.module = m; d[1].miss.entryFunctionName = "__miss__ms";
    d[2].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP; d[2].hitgroup.moduleCH = m; d[2].hitgroup.entryFunctionNameCH = "__closesthit__ch";
    if (is_cus) { d[2].hitgroup.moduleIS = m; d[2].hitgroup.entryFunctionNameIS = "__intersection__sph"; }
    if (is_sph) { d[2].hitgroup.moduleIS = bm; d[2].hitgroup.entryFunctionNameIS = nullptr; }
    OptixProgramGroup pg[3]; ls = sizeof log; OX(optixProgramGroupCreate(ctx, d, 3, &go, log, &ls, pg));
    OptixPipelineLinkOptions lo{}; lo.maxTraceDepth = 1; OptixPipeline pipe; ls = sizeof log; OX(optixPipelineCreate(ctx, &po, &lo, pg, 3, log, &ls, &pipe));
    struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) Rec { char h[OPTIX_SBT_RECORD_HEADER_SIZE]; };
    Rec rec[3]; for (int k = 0; k < 3; ++k) OX(optixSbtRecordPackHeader(pg[k], &rec[k]));
    CUdeviceptr drec; CK(cudaMalloc((void **)&drec, sizeof rec)); CK(cudaMemcpy((void *)drec, rec, sizeof rec, cudaMemcpyHostToDevice));
    OptixShaderBindingTable sbt{}; sbt.raygenRecord = drec; sbt.missRecordBase = drec + sizeof(Rec); sbt.missRecordStrideInBytes = sizeof(Rec); sbt.missRecordCount = 1;
    sbt.hitgroupRecordBase = drec + 2 * sizeof(Rec); sbt.hitgroupRecordStrideInBytes = sizeof(Rec); sbt.hitgroupRecordCount = 1;
    FILE *jf = fopen(argv[2], "wx"); if (!jf) { fprintf(stderr, "refuse to overwrite or cannot write %s\n", argv[2]); return 2; }
    fprintf(jf, "{\"kind\": \"%s\", \"N\": %ld, \"r\": %.5f, \"scene\": %d, \"off\": %.3f, \"jit\": %.4f, \"tps\": %u, \"gas_bytes\": %zu, \"gas_compacted_bytes\": %llu, \"ptx\": \"%s\"", kind.c_str(), N, R, scene, off, jitf, tps, (size_t)bs.outputSizeInBytes, csz, argv[1]);
    const long maxn = std::max({I("sat", 0), I("nver", 1 << 20), 32L});
    unsigned long long *dcal, *dcyc; unsigned *dsink; uint4 *dout;
    CK(cudaMalloc((void **)&dcal, (size_t)(maxn / 32 + 1) * 32)); CK(cudaMalloc((void **)&dsink, (size_t)(maxn / 32 + 1) * 4)); CK(cudaMemset(dsink, 0, (size_t)(maxn / 32 + 1) * 4));
    CK(cudaMalloc((void **)&dcyc, 32 * 8)); CK(cudaMalloc((void **)&dout, (size_t)maxn * 16));
    Params prm{}; prm.handle = root; prm.centers = (const float3 *)dcen; prm.radius = R; prm.tps = tps; prm.scene = scene; prm.jit = jitf * R; prm.sink = dsink; prm.cyc = dcyc; prm.cal = nullptr; prm.out = nullptr;
    CUdeviceptr dprm; CK(cudaMalloc((void **)&dprm, sizeof prm));
    auto launch = [&](unsigned n) { CK(cudaMemcpy((void *)dprm, &prm, sizeof prm, cudaMemcpyHostToDevice)); OX(optixLaunch(pipe, 0, dprm, sizeof prm, &sbt, n, 1, 1)); };
    if (kv.count("verify")) {   // per ray results of the first nver rays: (hit, t bits, sphere index, intersection program calls) as uint32 x 4, written to a new file
        const unsigned nv = (unsigned)I("nver", 1 << 20); prm.mode = 0; prm.n = nv; prm.out = dout; launch(nv); CK(cudaDeviceSynchronize());
        std::vector<uint4> h(nv); CK(cudaMemcpy(h.data(), dout, (size_t)nv * 16, cudaMemcpyDeviceToHost));
        FILE *bf = fopen(kv["verify"].c_str(), "wxb"); if (!bf) { fprintf(stderr, "refuse to overwrite %s\n", kv["verify"].c_str()); return 2; } fwrite(h.data(), 16, nv, bf); fclose(bf);
        fprintf(jf, ", \"verify\": \"%s\", \"nver\": %u", kv["verify"].c_str(), nv); prm.out = nullptr;
    }
    if (kv.count("sat")) {   // saturation: N rays from the hash, events, then calibration launches (clock64 and globaltimer per warp)
        const unsigned n = (unsigned)I("sat", 0); const int reps = (int)I("reps", 10), ncal = (int)I("ncal", 5), ncu = (int)I("ncu", 0); prm.mode = 0; prm.n = n; prm.cal = nullptr;
        launch(n); CK(cudaDeviceSynchronize());   // warm up
        std::vector<float> ms; cudaEvent_t a0, a1; CK(cudaEventCreate(&a0)); CK(cudaEventCreate(&a1));
        for (int k = 0; k < (ncu ? 1 : reps); ++k) { CK(cudaMemcpy((void *)dprm, &prm, sizeof prm, cudaMemcpyHostToDevice)); CK(cudaEventRecord(a0)); OX(optixLaunch(pipe, 0, dprm, sizeof prm, &sbt, n, 1, 1)); CK(cudaEventRecord(a1)); CK(cudaEventSynchronize(a1)); float t; CK(cudaEventElapsedTime(&t, a0, a1)); ms.push_back(t); }
        const size_t nw = (n + 31) / 32; std::vector<unsigned long long> cal; prm.cal = dcal;
        for (int k = 0; k < (ncu ? 0 : ncal); ++k) { launch(n); CK(cudaDeviceSynchronize()); std::vector<unsigned long long> c(nw * 4); CK(cudaMemcpy(c.data(), dcal, nw * 32, cudaMemcpyDeviceToHost)); cal.insert(cal.end(), c.begin(), c.end()); }
        fprintf(jf, ", \"sat\": {\"n\": %u, \"ms\": [", n); for (size_t k = 0; k < ms.size(); ++k) fprintf(jf, "%s%.6f", k ? "," : "", ms[k]);
        fprintf(jf, "], \"ncalw\": %zu, \"cal\": [", nw); for (size_t k = 0; k < cal.size(); ++k) fprintf(jf, "%s%llu", k ? "," : "", cal[k]); fprintf(jf, "]}"); prm.cal = nullptr;
    }
    if (kv.count("warp")) {   // one warp, every lane the same ray (ids from a file), per ray wreps launches, the first dropped by the analysis
        std::ifstream f(kv["warp"]); std::vector<unsigned> ids; unsigned x; while (f >> x) ids.push_back(x); const int wr = (int)I("wreps", 12);
        fprintf(jf, ", \"warp\": ["); prm.mode = 1; prm.n = 32; prm.out = nullptr;
        for (size_t q = 0; q < ids.size(); ++q) {
            prm.ray_id = ids[q]; fprintf(jf, "%s{\"id\": %u, \"cyc\": [", q ? "," : "", ids[q]);
            for (int k = 0; k < wr; ++k) { launch(32); CK(cudaDeviceSynchronize()); unsigned long long c[32]; CK(cudaMemcpy(c, dcyc, sizeof c, cudaMemcpyDeviceToHost)); std::sort(c, c + 32); fprintf(jf, "%s%llu", k ? "," : "", c[16]); }
            fprintf(jf, "]}");
        }
        fprintf(jf, "]");
    }
    fprintf(jf, "}\n"); fclose(jf); printf("done %s\n", argv[2]);
    return 0;
}
