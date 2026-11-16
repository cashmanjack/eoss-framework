// Load_1 host (OptiX only, no TTU code of our own): loaded point query test.
//   load PTX ORDER(0 lattice|1 random) NQ REPEAT OUT.bin     one NQ per process; prints one JSON line. NQ multiple of 32.
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
int main(int argc, char **argv)
{
    if (argc < 6) { fprintf(stderr, "usage: load PTX ORDER NQ REPEAT OUT\n"); return 2; }
    const int ord = atoi(argv[2]); const unsigned NQ = (unsigned)atoi(argv[3]); const int repeat = atoi(argv[4]);
    if (NQ % 32) { fprintf(stderr, "NQ must be a multiple of 32\n"); return 2; }
    const int NX = 50, NY = 50, NZ = 40, NP = NX * NY * NZ;
    std::vector<float> pts(3 * NP), bx(6 * NP);
    for (int k = 0; k < NZ; ++k) for (int j = 0; j < NY; ++j) for (int i = 0; i < NX; ++i) { int p = (k * NY + j) * NX + i; float c[3] = {(float)i, (float)j, (float)k};
        for (int a = 0; a < 3; ++a) { pts[3 * p + a] = c[a]; bx[6 * p + a] = c[a] - 0.4f; bx[6 * p + 3 + a] = c[a] + 0.4f; } }
    std::vector<unsigned> order(NQ); { std::vector<unsigned> perm(NP); std::iota(perm.begin(), perm.end(), 0u); if (ord) { std::mt19937 g(77); std::shuffle(perm.begin(), perm.end(), g); }
        for (unsigned i = 0; i < NQ; ++i) order[i] = perm[i % NP]; }
    CK(cudaFree(0)); OX(optixInit()); OptixDeviceContextOptions co{}; OptixDeviceContext ctx; OX(optixDeviceContextCreate(0, &co, &ctx));
    CUdeviceptr db, dpt, dord; CK(cudaMalloc((void **)&db, bx.size() * 4)); CK(cudaMemcpy((void *)db, bx.data(), bx.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc((void **)&dpt, pts.size() * 4)); CK(cudaMemcpy((void *)dpt, pts.data(), pts.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc((void **)&dord, NQ * 4)); CK(cudaMemcpy((void *)dord, order.data(), NQ * 4, cudaMemcpyHostToDevice));
    unsigned fl = OPTIX_GEOMETRY_FLAG_NONE; OptixBuildInput bi{}; bi.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES; bi.customPrimitiveArray.aabbBuffers = &db; bi.customPrimitiveArray.numPrimitives = NP; bi.customPrimitiveArray.strideInBytes = 24; bi.customPrimitiveArray.flags = &fl; bi.customPrimitiveArray.numSbtRecords = 1;
    OptixAccelBuildOptions ao{}; ao.buildFlags = OPTIX_BUILD_FLAG_NONE; ao.operation = OPTIX_BUILD_OPERATION_BUILD; OptixAccelBufferSizes bs{}; OX(optixAccelComputeMemoryUsage(ctx, &ao, &bi, 1, &bs));
    CUdeviceptr tmp, gas; CK(cudaMalloc((void **)&tmp, bs.tempSizeInBytes)); CK(cudaMalloc((void **)&gas, bs.outputSizeInBytes)); OptixTraversableHandle h;
    OX(optixAccelBuild(ctx, 0, &ao, &bi, 1, tmp, bs.tempSizeInBytes, gas, bs.outputSizeInBytes, &h, nullptr, 0)); CK(cudaDeviceSynchronize());
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
    Params prm{}; prm.handle = h; prm.n = NQ; prm.boxes = (const float *)db; prm.pts = (const float *)dpt; prm.order = (const unsigned *)dord;
    const unsigned NW = NQ / 32; CK(cudaMalloc((void **)&prm.rec, NW * sizeof(Rec))); CK(cudaMalloc((void **)&prm.hits, NQ * 4));
    CUdeviceptr dprm; CK(cudaMalloc((void **)&dprm, sizeof prm)); CK(cudaMemcpy((void *)dprm, &prm, sizeof prm, cudaMemcpyHostToDevice));
    OptixShaderBindingTable sbt{}; sbt.raygenRecord = drec; sbt.missRecordBase = drec + sizeof(SR); sbt.missRecordStrideInBytes = sizeof(SR); sbt.missRecordCount = 1; sbt.hitgroupRecordBase = drec + 2 * sizeof(SR); sbt.hitgroupRecordStrideInBytes = sizeof(SR); sbt.hitgroupRecordCount = 1;
    std::vector<Rec> hr(NW); std::vector<unsigned> hh(NQ);
    printf("{\"nq\":%u,\"order\":%d,\"launches\":[", NQ, ord);
    cudaEvent_t a0, a1; CK(cudaEventCreate(&a0)); CK(cudaEventCreate(&a1));
    for (int k = -1; k < repeat; ++k) {   // k = -1: the first (cold) launch, reported with flag cold; then the repeats
        if (k == 0) { for (int w = 0; w < 2; ++w) { OX(optixLaunch(pipe, 0, dprm, sizeof prm, &sbt, NQ, 1, 1)); CK(cudaDeviceSynchronize()); } }
        CK(cudaEventRecord(a0)); OX(optixLaunch(pipe, 0, dprm, sizeof prm, &sbt, NQ, 1, 1)); CK(cudaEventRecord(a1)); CK(cudaEventSynchronize(a1)); float ms; CK(cudaEventElapsedTime(&ms, a0, a1));
        CK(cudaMemcpy(hr.data(), prm.rec, NW * sizeof(Rec), cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hh.data(), prm.hits, NQ * 4, cudaMemcpyDeviceToHost));
        unsigned long long gmin = ~0ull, gmax = 0, calls = 0, nhit = 0; double cyc = 0, ns = 0; std::vector<double> wc(NW);
        for (unsigned w = 0; w < NW; ++w) { gmin = std::min(gmin, hr[w].g0); gmax = std::max(gmax, hr[w].g1); calls += hr[w].calls; wc[w] = (double)(hr[w].c1 - hr[w].c0); cyc += wc[w]; ns += (double)(hr[w].g1 - hr[w].g0); }
        for (unsigned i = 0; i < NQ; ++i) nhit += hh[i];
        std::sort(wc.begin(), wc.end());
        printf("%s{\"cold\":%d,\"event_ms\":%.5f,\"span_ns\":%llu,\"calls\":%llu,\"hits\":%llu,\"warp_cyc_mean\":%.1f,\"warp_cyc_med\":%.1f,\"warp_cyc_p90\":%.1f,\"warp_ns_mean\":%.1f,\"sms\":%d}", k > -1 ? "," : "", k == -1, ms, gmax - gmin, calls, nhit, cyc / NW, wc[NW / 2], wc[(size_t)(NW * 0.9)], ns / NW, 0);
    }
    printf("]}\n");
    FILE *f = fopen(argv[5], "wbx"); if (!f) { fprintf(stderr, "cannot create %s (exists?)\n", argv[5]); return 2; } fwrite(hr.data(), sizeof(Rec), NW, f); fclose(f);   // warp records of the last launch (raw, never overwritten)
    return 0;
}
