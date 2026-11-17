// Omm_1 host (OptiX only, no TTU code). Predictions: model/predictions_phase10b.md section 2.
//   omm PTX OUT.json variant=A0|A|D0|D|B|C level=L [N=100000] [big=2097152] [reps=20] [count=1] [dumpres=FILE]
//   A0/A: any hit alpha tested, pipeline allowOpacityMicromaps 0/1, no micromaps.  D0/D: any hit disabled (opaque), flag 0/1.
//   B: 4 state micromaps (unknown = unknown transparent), any hit program resolves unknown states.  C: 2 state micromaps (unknown = opaque), any hit disabled.
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
void omm_bary(unsigned t, unsigned L, float *b);
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
#include "omm_params.h"
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(2); } } while (0)
#define OX(x) do { OptixResult r_ = (x); if (r_ != OPTIX_SUCCESS) { fprintf(stderr, "%s:%d %s: %d\n", __FILE__, __LINE__, #x, (int)r_); exit(2); } } while (0)
struct Tri { float3 a, b, c; };
static std::string read_file(const char *p) { std::ifstream f(p); if (!f) { fprintf(stderr, "cannot read %s\n", p); exit(2); } std::stringstream s; s << f.rdbuf(); return s.str(); }
static double dseg(double px, double py, double ax, double ay, double bx, double by)
{ const double dx = bx - ax, dy = by - ay; double t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy); t = std::max(0.0, std::min(1.0, t)); const double qx = ax + t * dx - px, qy = ay + t * dy - py; return std::sqrt(qx * qx + qy * qy); }
// 0 transparent (whole micro triangle inside the hole), 1 opaque (whole outside), 2 unknown
static int classify(double cx, double cy, double r, const float2 *v)
{
    double mx = 0; for (int k = 0; k < 3; ++k) mx = std::max(mx, std::hypot(v[k].x - cx, v[k].y - cy));
    double mn;
    auto sgn = [&](double ax, double ay, double bx, double by) { return (cx - bx) * (ay - by) - (ax - bx) * (cy - by); };
    const double d1 = sgn(v[0].x, v[0].y, v[1].x, v[1].y), d2 = sgn(v[1].x, v[1].y, v[2].x, v[2].y), d3 = sgn(v[2].x, v[2].y, v[0].x, v[0].y);
    const bool neg = d1 < 0 || d2 < 0 || d3 < 0, pos = d1 > 0 || d2 > 0 || d3 > 0;
    if (!(neg && pos)) mn = 0; else mn = std::min({dseg(cx, cy, v[0].x, v[0].y, v[1].x, v[1].y), dseg(cx, cy, v[1].x, v[1].y, v[2].x, v[2].y), dseg(cx, cy, v[2].x, v[2].y, v[0].x, v[0].y)});
    const double eps = 1e-6;   // keep boundary cases unknown
    if (mx < r - eps) return 0; if (mn > r + eps) return 1; return 2;
}
int main(int argc, char **argv)
{
    if (argc < 3) { fprintf(stderr, "usage\n"); return 2; }
    std::map<std::string, std::string> kv; for (int a = 3; a < argc; ++a) { std::string s = argv[a]; const size_t e = s.find('='); if (e == std::string::npos) { fprintf(stderr, "bad arg %s\n", argv[a]); return 2; } kv[s.substr(0, e)] = s.substr(e + 1); }
    auto geti = [&](const char *k, long long d) { return kv.count(k) ? atoll(kv[k].c_str()) : d; };
    const std::string var = kv.count("variant") ? kv["variant"] : "A"; const int L = (int)geti("level", 4); const long N = geti("N", 100000); const unsigned NB = (unsigned)geti("big", 2097152); const int reps = (int)geti("reps", 20), docount = (int)geti("count", 1);
    const bool useomm = var == "B" || var == "C", noany = var == "D0" || var == "D" || var == "C", flag = var != "A0" && var != "D0";
    if (!(var == "A0" || var == "A" || var == "D0" || var == "D" || var == "B" || var == "C")) { fprintf(stderr, "variant?\n"); return 2; }
    CK(cudaFree(0)); OX(optixInit()); OptixDeviceContext ctx; OptixDeviceContextOptions co{}; OX(optixDeviceContextCreate(0, &co, &ctx));
    std::mt19937 g(1234); std::uniform_real_distribution<float> U(-1.f, 1.f); std::normal_distribution<float> Nn(0.f, 1.f);
    std::vector<Tri> v; const float e = 3.f / std::cbrt((float)N);
    for (long i = 0; i < N; ++i) { const float3 c{U(g), U(g), U(g)}; auto pt = [&]() { return float3{c.x + 0.5f * e * U(g), c.y + 0.5f * e * U(g), c.z + 0.5f * e * U(g)}; }; v.push_back({pt(), pt(), pt()}); }
    const float radii[4] = {0.15f, 0.25f, 0.35f, 0.45f}; double tfrac[4];
    for (int p = 0; p < 4; ++p) { int in = 0, tot = 0; const int M = 2000; for (int a = 0; a < M; ++a) for (int b = 0; b < M - a; ++b) { const double u = (a + 0.5) / M, w = (b + 0.5) / M; ++tot; double du = u - 1.0 / 3, dv = w - 1.0 / 3; if (du * du + dv * dv <= (double)radii[p] * radii[p]) ++in; } tfrac[p] = (double)in / tot; }
    CUdeviceptr dv, omm = 0, didx = 0; CK(cudaMalloc((void **)&dv, v.size() * sizeof(Tri))); CK(cudaMemcpy((void *)dv, v.data(), v.size() * sizeof(Tri), cudaMemcpyHostToDevice));
    size_t omm_bytes = 0; double unk_frac[4] = {0, 0, 0, 0};
    OptixOpacityMicromapUsageCount uc[4];
    if (useomm) {
        const bool four = var == "B"; const unsigned nmicro = 1u << (2 * L); const size_t bits = four ? 2 : 1; const size_t raw = std::max<size_t>(1, (nmicro * bits + 7) / 8), stride = (raw + 127) / 128 * 128;
        std::vector<unsigned char> data(4 * stride, 0); std::vector<OptixOpacityMicromapDesc> desc(4);
        for (int p = 0; p < 4; ++p) {
            unsigned unk = 0;
            for (unsigned t = 0; t < nmicro; ++t) {
                float bb[6]; omm_bary(t, (unsigned)L, bb); const float2 vv[3] = {{bb[0], bb[1]}, {bb[2], bb[3]}, {bb[4], bb[5]}};
                const int c = classify(1.0 / 3, 1.0 / 3, radii[p], vv); unk += c == 2; unsigned st;
                if (four) st = c == 0 ? OPTIX_OPACITY_MICROMAP_STATE_TRANSPARENT : c == 1 ? OPTIX_OPACITY_MICROMAP_STATE_OPAQUE : OPTIX_OPACITY_MICROMAP_STATE_UNKNOWN_TRANSPARENT;
                else st = c == 0 ? 0 : 1;   // unknown = opaque
                const size_t bit = (size_t)t * bits; data[p * stride + bit / 8] |= (unsigned char)(st << (bit % 8));
            }
            unk_frac[p] = (double)unk / nmicro; desc[p].byteOffset = (unsigned)(p * stride); desc[p].subdivisionLevel = (unsigned short)L; desc[p].format = (unsigned short)(four ? OPTIX_OPACITY_MICROMAP_FORMAT_4_STATE : OPTIX_OPACITY_MICROMAP_FORMAT_2_STATE);
        }
        CUdeviceptr din, ddesc, dtmp; CK(cudaMalloc((void **)&din, data.size())); CK(cudaMemcpy((void *)din, data.data(), data.size(), cudaMemcpyHostToDevice));
        CK(cudaMalloc((void **)&ddesc, 4 * sizeof(OptixOpacityMicromapDesc))); CK(cudaMemcpy((void *)ddesc, desc.data(), 4 * sizeof(OptixOpacityMicromapDesc), cudaMemcpyHostToDevice));
        OptixOpacityMicromapHistogramEntry he{4, (unsigned)L, four ? OPTIX_OPACITY_MICROMAP_FORMAT_4_STATE : OPTIX_OPACITY_MICROMAP_FORMAT_2_STATE};
        OptixOpacityMicromapArrayBuildInput mi{}; mi.flags = OPTIX_OPACITY_MICROMAP_FLAG_PREFER_FAST_TRACE; mi.inputBuffer = din; mi.perMicromapDescBuffer = ddesc; mi.perMicromapDescStrideInBytes = 0; mi.numMicromapHistogramEntries = 1; mi.micromapHistogramEntries = &he;
        OptixMicromapBufferSizes ms{}; OX(optixOpacityMicromapArrayComputeMemoryUsage(ctx, &mi, &ms));
        CK(cudaMalloc((void **)&omm, ms.outputSizeInBytes)); CK(cudaMalloc((void **)&dtmp, std::max<size_t>(ms.tempSizeInBytes, 16)));
        OptixMicromapBuffers mb{}; mb.output = omm; mb.outputSizeInBytes = ms.outputSizeInBytes; mb.temp = dtmp; mb.tempSizeInBytes = ms.tempSizeInBytes;
        OX(optixOpacityMicromapArrayBuild(ctx, 0, &mi, &mb)); CK(cudaDeviceSynchronize()); omm_bytes = ms.outputSizeInBytes;
        std::vector<int> idx(N); for (long i = 0; i < N; ++i) idx[i] = (int)(i & 3); CK(cudaMalloc((void **)&didx, N * 4)); CK(cudaMemcpy((void *)didx, idx.data(), N * 4, cudaMemcpyHostToDevice));
        for (int p = 0; p < 4; ++p) { uc[p].count = (unsigned)((N - p + 3) / 4); uc[p].subdivisionLevel = (unsigned)L; uc[p].format = four ? OPTIX_OPACITY_MICROMAP_FORMAT_4_STATE : OPTIX_OPACITY_MICROMAP_FORMAT_2_STATE; }
        // usage counts are per distinct (format, level): one entry with the total count
        uc[0].count = (unsigned)N;
    }
    unsigned gflag = noany ? OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT : OPTIX_GEOMETRY_FLAG_NONE;
    OptixBuildInput bi{}; bi.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES; bi.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3; bi.triangleArray.numVertices = (unsigned)v.size() * 3; bi.triangleArray.vertexBuffers = &dv; bi.triangleArray.flags = &gflag; bi.triangleArray.numSbtRecords = 1;
    if (useomm) { OptixBuildInputOpacityMicromap &om = bi.triangleArray.opacityMicromap; om.indexingMode = OPTIX_OPACITY_MICROMAP_ARRAY_INDEXING_MODE_INDEXED; om.opacityMicromapArray = omm; om.indexBuffer = didx; om.indexSizeInBytes = 4; om.indexStrideInBytes = 0; om.indexOffset = 0; om.numMicromapUsageCounts = 1; om.micromapUsageCounts = &uc[0]; }
    OptixAccelBuildOptions ao{}; ao.buildFlags = OPTIX_BUILD_FLAG_NONE; ao.operation = OPTIX_BUILD_OPERATION_BUILD;
    OptixAccelBufferSizes bs{}; OX(optixAccelComputeMemoryUsage(ctx, &ao, &bi, 1, &bs));
    CUdeviceptr tmp, gas; CK(cudaMalloc((void **)&tmp, bs.tempSizeInBytes)); CK(cudaMalloc((void **)&gas, bs.outputSizeInBytes));
    OptixTraversableHandle hg; OX(optixAccelBuild(ctx, 0, &ao, &bi, 1, tmp, bs.tempSizeInBytes, gas, bs.outputSizeInBytes, &hg, nullptr, 0)); CK(cudaDeviceSynchronize());
    std::vector<Ray8> big; { std::mt19937 gr(2026); for (unsigned i = 0; i < NB; ++i) { float3 d{Nn(gr), Nn(gr), Nn(gr)}; const float l = std::sqrt(d.x * d.x + d.y * d.y + d.z * d.z); big.push_back(Ray8{{U(gr), U(gr), U(gr)}, 0.f, {d.x / l, d.y / l, d.z / l}, 1e16f}); } }
    OptixModuleCompileOptions mo{}; OptixPipelineCompileOptions po{};
    po.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS; po.numPayloadValues = 3; po.numAttributeValues = 2; po.pipelineLaunchParamsVariableName = "params"; po.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE; po.allowOpacityMicromaps = flag ? 1 : 0;
    std::string ptx = read_file(argv[1]); char log[4096]; size_t ls = sizeof log; OptixModule m; OX(optixModuleCreate(ctx, &mo, &po, ptx.c_str(), ptx.size(), log, &ls, &m));
    OptixProgramGroupOptions go{}; OptixProgramGroupDesc d[3] = {};
    d[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN; d[0].raygen.module = m; d[0].raygen.entryFunctionName = "__raygen__rg";
    d[1].kind = OPTIX_PROGRAM_GROUP_KIND_MISS; d[1].miss.module = m; d[1].miss.entryFunctionName = "__miss__ms";
    d[2].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP; d[2].hitgroup.moduleCH = m; d[2].hitgroup.entryFunctionNameCH = "__closesthit__ch"; d[2].hitgroup.moduleAH = m; d[2].hitgroup.entryFunctionNameAH = "__anyhit__ah";
    OptixProgramGroup pg[3]; ls = sizeof log; OX(optixProgramGroupCreate(ctx, d, 3, &go, log, &ls, pg));
    OptixPipelineLinkOptions lo{}; lo.maxTraceDepth = 1; OptixPipeline pipe; ls = sizeof log; OX(optixPipelineCreate(ctx, &po, &lo, pg, 3, log, &ls, &pipe));
    OX(optixPipelineSetStackSize(pipe, 0, 0, 2048, 1));
    struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) Rec { char h[OPTIX_SBT_RECORD_HEADER_SIZE]; };
    Rec rec[3]; for (int k = 0; k < 3; ++k) OX(optixSbtRecordPackHeader(pg[k], &rec[k]));
    CUdeviceptr drec; CK(cudaMalloc((void **)&drec, sizeof rec)); CK(cudaMemcpy((void *)drec, rec, sizeof rec, cudaMemcpyHostToDevice));
    OptixShaderBindingTable sbt{}; sbt.raygenRecord = drec; sbt.missRecordBase = drec + sizeof(Rec); sbt.missRecordStrideInBytes = sizeof(Rec); sbt.missRecordCount = 1;
    sbt.hitgroupRecordBase = drec + 2 * sizeof(Rec); sbt.hitgroupRecordStrideInBytes = sizeof(Rec); sbt.hitgroupRecordCount = 1;
    Ray8 *drays; Out *dout; CK(cudaMalloc((void **)&drays, (size_t)NB * sizeof(Ray8))); CK(cudaMalloc((void **)&dout, (size_t)NB * sizeof(Out))); CK(cudaMemset(dout, 0, (size_t)NB * sizeof(Out)));
    CK(cudaMemcpy(drays, big.data(), (size_t)NB * sizeof(Ray8), cudaMemcpyHostToDevice));
    Params prm{}; prm.rays = drays; prm.out = dout; prm.handle = hg; prm.n = NB; prm.cnt = 0; for (int p = 0; p < 4; ++p) prm.r2[p] = radii[p] * radii[p];
    CUdeviceptr dprm; CK(cudaMalloc((void **)&dprm, sizeof prm));
    auto launch = [&]() { CK(cudaMemcpy((void *)dprm, &prm, sizeof prm, cudaMemcpyHostToDevice)); OX(optixLaunch(pipe, 0, dprm, sizeof prm, &sbt, NB, 1, 1)); };
    FILE *jf = fopen(argv[2], "wx"); if (!jf) { fprintf(stderr, "refuse to overwrite or cannot write %s\n", argv[2]); return 2; }
    fprintf(jf, "{\"variant\": \"%s\", \"level\": %d, \"N\": %ld, \"big\": %u, \"gas_bytes\": %zu, \"omm_bytes\": %zu, \"hole_frac\": [%.5f,%.5f,%.5f,%.5f], \"unknown_frac\": [%.5f,%.5f,%.5f,%.5f]", var.c_str(), L, N, NB, (size_t)bs.outputSizeInBytes, omm_bytes, tfrac[0], tfrac[1], tfrac[2], tfrac[3], unk_frac[0], unk_frac[1], unk_frac[2], unk_frac[3]);
    double calls_mean = 0, calls_max = 0; 
    launch(); CK(cudaDeviceSynchronize());   // warm up
    if (docount) {
        prm.cnt = 1; CK(cudaMemset(dout, 0, (size_t)NB * sizeof(Out))); launch(); CK(cudaDeviceSynchronize()); prm.cnt = 0;
        std::vector<Out> h(NB); CK(cudaMemcpy(h.data(), dout, (size_t)NB * sizeof(Out), cudaMemcpyDeviceToHost)); double s = 0; unsigned mx = 0, hits = 0; for (auto &o : h) { s += o.calls; mx = std::max(mx, o.calls); hits += o.hit; }
        calls_mean = s / NB; calls_max = mx; fprintf(jf, ", \"calls_mean\": %.6f, \"calls_max\": %u, \"hit_frac\": %.6f", calls_mean, mx, (double)hits / NB);
    }
    std::vector<float> ms; cudaEvent_t a0, a1; CK(cudaEventCreate(&a0)); CK(cudaEventCreate(&a1));
    for (int k = 0; k < reps; ++k) { CK(cudaMemcpy((void *)dprm, &prm, sizeof prm, cudaMemcpyHostToDevice)); CK(cudaEventRecord(a0)); OX(optixLaunch(pipe, 0, dprm, sizeof prm, &sbt, NB, 1, 1)); CK(cudaEventRecord(a1)); CK(cudaEventSynchronize(a1)); float t; CK(cudaEventElapsedTime(&t, a0, a1)); ms.push_back(t); }
    fprintf(jf, ", \"ms\": ["); for (size_t k = 0; k < ms.size(); ++k) fprintf(jf, "%s%.6f", k ? "," : "", ms[k]); fprintf(jf, "]");
    if (kv.count("dumpres")) { std::vector<Out> h(NB); CK(cudaMemcpy(h.data(), dout, (size_t)NB * sizeof(Out), cudaMemcpyDeviceToHost)); FILE *f = fopen(kv["dumpres"].c_str(), "wbx"); if (!f) { fprintf(stderr, "refuse to overwrite %s\n", kv["dumpres"].c_str()); return 2; } fwrite(h.data(), sizeof(Out), NB, f); fclose(f); }
    fprintf(jf, "}\n"); fclose(jf); printf("done %s\n", argv[2]);
    return 0;
}
