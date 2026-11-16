// Load_1 host (OptiX only, no TTU code of our own): loaded point query test.
//   load2 PTX NQ REPEAT NSTEPS     squeeze sequence, trees R (refit chain), U (fresh, update flag), F (fresh); prints one JSON line per step and tree.
// Lattice 50 x 50 x 40 = 100,000 disjoint boxes (half width 0.4, spacing 1), custom primitives; query i = centre of particle order[i].
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(2); } } while (0)
#define OX(x) do { OptixResult r_ = (x); if (r_ != OPTIX_SUCCESS) { fprintf(stderr, "%s:%d %s: %d\n", __FILE__, __LINE__, #x, (int)r_); exit(2); } } while (0)
struct Rec { unsigned long long c0, c1, g0, g1; unsigned sm, calls; };
struct Params { const float *boxes; const float *pts; const unsigned *order; Rec *rec; unsigned *hits; OptixTraversableHandle handle; unsigned n; };
static std::string read_file(const char *p) { std::ifstream f(p); if (!f) { fprintf(stderr, "cannot read %s\n", p); exit(2); } std::stringstream s; s << f.rdbuf(); return s.str(); }

static int g_mode = 0;
static void positions(float f, std::vector<float> &pts, std::vector<float> &bx) {
    const int NX = 50, NY = 50, NZ = 40; auto jit = [](unsigned p, unsigned a) { unsigned h = (p * 2654435761u) ^ (a * 40503u + 12345u); h ^= h >> 15; h *= 2246822519u; h ^= h >> 13; return ((h & 0xffff) / 65535.f - 0.5f) * 0.1f; };
    for (int k = 0; k < NZ; ++k) for (int j = 0; j < NY; ++j) for (int i = 0; i < NX; ++i) { unsigned p = (k * NY + j) * NX + i; float c[3] = {(float)i + jit(p, 0), (float)j + jit(p, 1), f * (float)k + jit(p, 2)};
        if (g_mode) { c[0] = (float)i + jit(p, 0); c[1] = (float)j + jit(p, 1); c[2] = (float)k + jit(p, 2); for (int a = 0; a < 3; ++a) c[a] += 20.f * f * jit(p * 7u + 3u, a + 5); }   // scramble: random displacement of up to +-1 f cells per axis (jit is +-0.05, so 20 f jit is +-f)
        for (int a = 0; a < 3; ++a) { pts[3 * p + a] = c[a]; bx[6 * p + a] = c[a] - 0.4f; bx[6 * p + 3 + a] = c[a] + 0.4f; } }
}
int main(int argc, char **argv)
{
    if (argc < 5) { fprintf(stderr, "usage: load2 PTX NQ REPEAT NSTEPS\n"); return 2; }
    const unsigned NQ = (unsigned)atoi(argv[2]); const int repeat = atoi(argv[3]), nsteps = atoi(argv[4]); const float fs[7] = {1.f, 0.85f, 0.7f, 0.55f, 0.4f, 0.3f, 0.2f}; if (argc > 5) g_mode = atoi(argv[5]); const float fm[7] = {0.f, 0.5f, 1.f, 2.f, 4.f, 8.f, 16.f};
    if (NQ % 32 || NQ > 100000) { fprintf(stderr, "NQ multiple of 32 and at most 100000\n"); return 2; }
    const int NP = 100000; std::vector<float> pts(3 * NP), bx(6 * NP); std::vector<unsigned> order(NQ); for (unsigned i = 0; i < NQ; ++i) order[i] = i;
    CK(cudaFree(0)); OX(optixInit()); OptixDeviceContextOptions co{}; OptixDeviceContext ctx; OX(optixDeviceContextCreate(0, &co, &ctx));
    CUdeviceptr db, dpt, dord; CK(cudaMalloc((void **)&db, bx.size() * 4)); CK(cudaMalloc((void **)&dpt, pts.size() * 4)); CK(cudaMalloc((void **)&dord, NQ * 4)); CK(cudaMemcpy((void *)dord, order.data(), NQ * 4, cudaMemcpyHostToDevice));
    unsigned fl = OPTIX_GEOMETRY_FLAG_NONE; OptixBuildInput bi{}; bi.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES; bi.customPrimitiveArray.aabbBuffers = &db; bi.customPrimitiveArray.numPrimitives = NP; bi.customPrimitiveArray.strideInBytes = 24; bi.customPrimitiveArray.flags = &fl; bi.customPrimitiveArray.numSbtRecords = 1;
    OptixAccelBuildOptions aoU{}; aoU.buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE; aoU.operation = OPTIX_BUILD_OPERATION_BUILD; OptixAccelBuildOptions aoF{}; aoF.buildFlags = OPTIX_BUILD_FLAG_NONE; aoF.operation = OPTIX_BUILD_OPERATION_BUILD;
    OptixAccelBufferSizes bsU{}, bsF{}; OX(optixAccelComputeMemoryUsage(ctx, &aoU, &bi, 1, &bsU)); OX(optixAccelComputeMemoryUsage(ctx, &aoF, &bi, 1, &bsF));
    size_t tmpsz = std::max({bsU.tempSizeInBytes, bsF.tempSizeInBytes, bsU.tempUpdateSizeInBytes}); CUdeviceptr tmp, gasR, gasU, gasF; CK(cudaMalloc((void **)&tmp, tmpsz)); CK(cudaMalloc((void **)&gasR, bsU.outputSizeInBytes)); CK(cudaMalloc((void **)&gasU, bsU.outputSizeInBytes)); CK(cudaMalloc((void **)&gasF, bsF.outputSizeInBytes));
    OptixModuleCompileOptions mo{}; OptixPipelineCompileOptions po{}; po.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS; po.numPayloadValues = 2; po.numAttributeValues = 0;
    po.pipelineLaunchParamsVariableName = "params"; po.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;
    std::string ptx = read_file(argv[1]); char log[4096]; size_t ls = sizeof log; OptixModule m; { OptixResult rr_ = optixModuleCreate(ctx, &mo, &po, ptx.c_str(), ptx.size(), log, &ls, &m); if (rr_ != OPTIX_SUCCESS) { fprintf(stderr, "optixModuleCreate %d: %s\n", (int)rr_, log); return 2; } }
    OptixProgramGroupOptions go{}; OptixProgramGroupDesc d[3] = {}; d[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN; d[0].raygen.module = m; d[0].raygen.entryFunctionName = "__raygen__rg";
    d[1].kind = OPTIX_PROGRAM_GROUP_KIND_MISS; d[1].miss.module = m; d[1].miss.entryFunctionName = "__miss__ms";
    d[2].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP; d[2].hitgroup.moduleCH = m; d[2].hitgroup.entryFunctionNameCH = "__closesthit__ch"; d[2].hitgroup.moduleIS = m; d[2].hitgroup.entryFunctionNameIS = "__intersection__is";
    OptixProgramGroup pg[3]; ls = sizeof log; OX(optixProgramGroupCreate(ctx, d, 3, &go, log, &ls, pg));
    OptixPipelineLinkOptions lo{}; lo.maxTraceDepth = 1; OptixPipeline pipe; ls = sizeof log; OX(optixPipelineCreate(ctx, &po, &lo, pg, 3, log, &ls, &pipe));
    struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) SR { char h[OPTIX_SBT_RECORD_HEADER_SIZE]; }; SR rec[3]; for (int k = 0; k < 3; ++k) OX(optixSbtRecordPackHeader(pg[k], &rec[k]));
    CUdeviceptr drec; CK(cudaMalloc((void **)&drec, sizeof rec)); CK(cudaMemcpy((void *)drec, rec, sizeof rec, cudaMemcpyHostToDevice));
    Params prm{}; prm.n = NQ; prm.boxes = (const float *)db; prm.pts = (const float *)dpt; prm.order = (const unsigned *)dord;
    const unsigned NW = NQ / 32; CK(cudaMalloc((void **)&prm.rec, NW * sizeof(Rec))); CK(cudaMalloc((void **)&prm.hits, NQ * 4));
    CUdeviceptr dprm; CK(cudaMalloc((void **)&dprm, sizeof prm));
    OptixShaderBindingTable sbt{}; sbt.raygenRecord = drec; sbt.missRecordBase = drec + sizeof(SR); sbt.missRecordStrideInBytes = sizeof(SR); sbt.missRecordCount = 1; sbt.hitgroupRecordBase = drec + 2 * sizeof(SR); sbt.hitgroupRecordStrideInBytes = sizeof(SR); sbt.hitgroupRecordCount = 1;
    std::vector<Rec> hr(NW); std::vector<unsigned> hh(NQ); OptixTraversableHandle hR = 0, hU = 0, hF = 0;
    for (int st = 0; st < nsteps; ++st) {
        positions(g_mode ? fm[st] : fs[st], pts, bx); CK(cudaMemcpy((void *)db, bx.data(), bx.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy((void *)dpt, pts.data(), pts.size() * 4, cudaMemcpyHostToDevice));
        if (st == 0) OX(optixAccelBuild(ctx, 0, &aoU, &bi, 1, tmp, tmpsz, gasR, bsU.outputSizeInBytes, &hR, nullptr, 0));
        else { OptixAccelBuildOptions up = aoU; up.operation = OPTIX_BUILD_OPERATION_UPDATE; OX(optixAccelBuild(ctx, 0, &up, &bi, 1, tmp, tmpsz, gasR, bsU.outputSizeInBytes, &hR, nullptr, 0)); }
        OX(optixAccelBuild(ctx, 0, &aoU, &bi, 1, tmp, tmpsz, gasU, bsU.outputSizeInBytes, &hU, nullptr, 0)); OX(optixAccelBuild(ctx, 0, &aoF, &bi, 1, tmp, tmpsz, gasF, bsF.outputSizeInBytes, &hF, nullptr, 0)); CK(cudaDeviceSynchronize());
        for (int tree = 0; tree < 3; ++tree) {
            prm.handle = tree == 0 ? hR : tree == 1 ? hU : hF; CK(cudaMemcpy((void *)dprm, &prm, sizeof prm, cudaMemcpyHostToDevice));
            printf("{\"step\":%d,\"f\":%.2f,\"tree\":\"%c\",\"nq\":%u,\"launches\":[", st, g_mode ? fm[st] : fs[st], "RUF"[tree], NQ);
            for (int k = -1; k < repeat; ++k) {
                if (k == 0) for (int w = 0; w < 2; ++w) { OX(optixLaunch(pipe, 0, dprm, sizeof prm, &sbt, NQ, 1, 1)); CK(cudaDeviceSynchronize()); }
                OX(optixLaunch(pipe, 0, dprm, sizeof prm, &sbt, NQ, 1, 1)); CK(cudaDeviceSynchronize());
                CK(cudaMemcpy(hr.data(), prm.rec, NW * sizeof(Rec), cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hh.data(), prm.hits, NQ * 4, cudaMemcpyDeviceToHost));
                unsigned long long gmin = ~0ull, gmax = 0, calls = 0, nhit = 0; double cyc = 0;
                for (unsigned w = 0; w < NW; ++w) { gmin = std::min(gmin, hr[w].g0); gmax = std::max(gmax, hr[w].g1); calls += hr[w].calls; cyc += (double)(hr[w].c1 - hr[w].c0); }
                for (unsigned i = 0; i < NQ; ++i) nhit += hh[i];
                printf("%s{\"cold\":%d,\"span_ns\":%llu,\"calls\":%llu,\"hits\":%llu,\"warp_cyc_mean\":%.1f}", k > -1 ? "," : "", k == -1, gmax - gmin, calls, nhit, cyc / NW);
            }
            printf("]}\n"); fflush(stdout);
        }
    }
    return 0;
}
