// Load_1 host (OptiX only, no TTU code of our own): loaded point query test.
//   load3 PTX DATASET NQ REPEAT IAS [SEQ WARM DUMPDIR]  SEQ 0 back to back, 1 update kernel before each launch, 2 update kernel + refit of GAS and instance group (+ host sync between, as Mochi);    Mochi dataset (Morton order as Mochi sorts it), GAS with the update flag only (as Mochi), optional instance level; prints one JSON line.
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
#include <queue>
#include <random>
#include <cmath>
#include <sstream>
#include <string>
#include <vector>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(2); } } while (0)
#define OX(x) do { OptixResult r_ = (x); if (r_ != OPTIX_SUCCESS) { fprintf(stderr, "%s:%d %s: %d\n", __FILE__, __LINE__, #x, (int)r_); exit(2); } } while (0)
struct Rec { unsigned long long c0, c1, g0, g1; unsigned sm, calls; };
struct Particle { float3 center; float3 velocity; float radius; float mass; };
struct Params { const Particle *particles; const float *boxes; float3 *acc; unsigned *callw; Rec *rec; OptixTraversableHandle handle; unsigned n; float stiffness, restitution; };
static std::string read_file(const char *p) { std::ifstream f(p); if (!f) { fprintf(stderr, "cannot read %s\n", p); exit(2); } std::stringstream s; s << f.rdbuf(); return s.str(); }


__global__ void upd(Particle *p, float3 *acc, float dt, int n) {   // Mochi's update kernel traffic: read particle and force, write particle, clear force (values unchanged: velocity 0)
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return; Particle q = p[i]; float3 a = acc[i]; q.velocity.x += 0.f * a.x * dt; q.center.x += q.velocity.x * dt; q.center.y += q.velocity.y * dt; q.center.z += q.velocity.z * dt; p[i] = q; acc[i] = make_float3(0.f, 0.f, 0.f); }
__global__ void bnd(const Particle *p, float *bx, int n) {   // Mochi's bounds program: centre -/+ 2 radius
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return; Particle q = p[i]; float r = q.radius; bx[6 * i] = q.center.x - 2 * r; bx[6 * i + 1] = q.center.y - 2 * r; bx[6 * i + 2] = q.center.z - 2 * r; bx[6 * i + 3] = q.center.x + 2 * r; bx[6 * i + 4] = q.center.y + 2 * r; bx[6 * i + 5] = q.center.z + 2 * r; }
static unsigned expandBits(unsigned v) { v = (v * 0x00010001u) & 0xFF0000FFu; v = (v * 0x00000101u) & 0x0F00F00Fu; v = (v * 0x00000011u) & 0xC30C30C3u; v = (v * 0x00000005u) & 0x49249249u; return v; }
int main(int argc, char **argv)
{
    if (argc < 6) { fprintf(stderr, "usage: load3 PTX DATASET NQ REPEAT IAS\n"); return 2; }
    const unsigned NQ = (unsigned)atoi(argv[3]); const int repeat = atoi(argv[4]), ias = atoi(argv[5]), seq = argc > 6 ? atoi(argv[6]) : 0, warm = argc > 7 ? atoi(argv[7]) : 0; const char *dumpdir = argc > 8 ? argv[8] : nullptr;
    if (NQ % 32 || NQ > 100000) { fprintf(stderr, "NQ multiple of 32 and at most 100000\n"); return 2; }
    std::vector<Particle> P; std::vector<std::pair<unsigned, unsigned>> mc;   // as Mochi: Morton code from x*1024 clamped, ids popped from a max priority queue (largest code first)
    { FILE *f = fopen(argv[2], "r"); if (!f) { fprintf(stderr, "no dataset\n"); return 2; } float v[5]; while (P.size() < 100000 && fscanf(f, "%f %f %f %f %f", v, v + 1, v + 2, v + 3, v + 4) == 5) { unsigned id = (unsigned)P.size(); P.push_back({{v[0], v[1], v[2]}, {0.f, 0.f, 0.f}, v[3], v[4]});
            auto cl = [](float a) { return std::min(std::max(a * 1024.0f, 0.0f), 1023.0f); }; mc.push_back({expandBits((unsigned)cl(v[0])) * 4 + expandBits((unsigned)cl(v[1])) * 2 + expandBits((unsigned)cl(v[2])), id}); } fclose(f); }
    const int flip = getenv("FLIP") ? atoi(getenv("FLIP")) : 0; fprintf(stderr, "FLIP %d\n", flip);
    const int NP = (int)P.size(); if (NP != 100000) { fprintf(stderr, "dataset has %d particles\n", NP); return 2; }
    std::priority_queue<std::pair<unsigned, unsigned>> pq; for (auto &x : mc) pq.push(x); std::vector<Particle> S; while (!pq.empty()) { S.push_back(P[pq.top().second]); pq.pop(); } P = S;
    std::vector<float> bx(6 * NP); for (int p = 0; p < NP; ++p) { const float r = P[p].radius; bx[6 * p] = P[p].center.x - 2 * r; bx[6 * p + 1] = P[p].center.y - 2 * r; bx[6 * p + 2] = P[p].center.z - 2 * r; bx[6 * p + 3] = P[p].center.x + 2 * r; bx[6 * p + 4] = P[p].center.y + 2 * r; bx[6 * p + 5] = P[p].center.z + 2 * r; }
    CK(cudaFree(0)); OX(optixInit()); OptixDeviceContextOptions co{}; OptixDeviceContext ctx; OX(optixDeviceContextCreate(0, &co, &ctx));
    CUdeviceptr db, dpt, dacc, dcw; CK(cudaMalloc((void **)&db, bx.size() * 4)); CK(cudaMemcpy((void *)db, bx.data(), bx.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMalloc((void **)&dpt, P.size() * sizeof(Particle))); CK(cudaMemcpy((void *)dpt, P.data(), P.size() * sizeof(Particle), cudaMemcpyHostToDevice));
    CK(cudaMalloc((void **)&dacc, NP * 12)); CK(cudaMemset((void *)dacc, 0, NP * 12)); const unsigned NW = NQ / 32; CK(cudaMalloc((void **)&dcw, NW * 4));
    unsigned fl = OPTIX_GEOMETRY_FLAG_NONE; OptixBuildInput bi{}; bi.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES; bi.customPrimitiveArray.aabbBuffers = &db; bi.customPrimitiveArray.numPrimitives = NP; bi.customPrimitiveArray.strideInBytes = 24; bi.customPrimitiveArray.flags = &fl; bi.customPrimitiveArray.numSbtRecords = 1;
    OptixAccelBuildOptions ao{}; ao.buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE; ao.operation = OPTIX_BUILD_OPERATION_BUILD; OptixAccelBufferSizes bs{}; OX(optixAccelComputeMemoryUsage(ctx, &ao, &bi, 1, &bs));
    CUdeviceptr tmp, gas; const size_t tmpsz = std::max(bs.tempSizeInBytes, bs.tempUpdateSizeInBytes); CK(cudaMalloc((void **)&tmp, tmpsz)); CK(cudaMalloc((void **)&gas, bs.outputSizeInBytes)); OptixTraversableHandle hG, hW; OX(optixAccelBuild(ctx, 0, &ao, &bi, 1, tmp, bs.tempSizeInBytes, gas, bs.outputSizeInBytes, &hG, nullptr, 0)); hW = hG;
    OptixBuildInput ib{}; CUdeviceptr di = 0, ias_ = 0, tmp2 = 0; size_t tmp2sz = 0, ias_sz = 0;
    if (ias) {   // as OWL: one instance (identity transform) of the GAS, built with the update flag
        OptixInstance inst{}; const float id[12] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0}; memcpy(inst.transform, id, sizeof id); inst.instanceId = 0; inst.sbtOffset = 0; inst.visibilityMask = 255; inst.flags = OPTIX_INSTANCE_FLAG_NONE; inst.traversableHandle = hG;
        CK(cudaMalloc((void **)&di, sizeof inst)); CK(cudaMemcpy((void *)di, &inst, sizeof inst, cudaMemcpyHostToDevice));
        ib.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES; ib.instanceArray.instances = di; ib.instanceArray.numInstances = 1; OptixAccelBufferSizes bs2{}; OX(optixAccelComputeMemoryUsage(ctx, &ao, &ib, 1, &bs2));
        tmp2sz = std::max(bs2.tempSizeInBytes, bs2.tempUpdateSizeInBytes); ias_sz = bs2.outputSizeInBytes; CK(cudaMalloc((void **)&tmp2, tmp2sz)); CK(cudaMalloc((void **)&ias_, bs2.outputSizeInBytes)); OX(optixAccelBuild(ctx, 0, &ao, &ib, 1, tmp2, bs2.tempSizeInBytes, ias_, bs2.outputSizeInBytes, &hW, nullptr, 0)); }
    CK(cudaDeviceSynchronize());
    OptixModuleCompileOptions mo{}; OptixPipelineCompileOptions po{}; po.traversableGraphFlags = ias ? OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING : OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS; po.numPayloadValues = 2; po.numAttributeValues = (flip & 1) ? 2 : 0;
    po.pipelineLaunchParamsVariableName = "params"; po.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM | ((flip & 2) ? OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE : 0);
    std::string ptx = read_file(argv[1]); char log[4096]; size_t ls = sizeof log; OptixModule m; { OptixResult rr_ = optixModuleCreate(ctx, &mo, &po, ptx.c_str(), ptx.size(), log, &ls, &m); if (rr_ != OPTIX_SUCCESS) { fprintf(stderr, "optixModuleCreate %d: %s\n", (int)rr_, log); return 2; } }
    OptixProgramGroupOptions go{}; OptixProgramGroupDesc d[3] = {}; d[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN; d[0].raygen.module = m; d[0].raygen.entryFunctionName = "__raygen__rg";
    d[1].kind = OPTIX_PROGRAM_GROUP_KIND_MISS; d[1].miss.module = m; d[1].miss.entryFunctionName = "__miss__ms";
    d[2].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP; d[2].hitgroup.moduleCH = m; d[2].hitgroup.entryFunctionNameCH = "__closesthit__ch"; d[2].hitgroup.moduleIS = m; d[2].hitgroup.entryFunctionNameIS = "__intersection__is";
    OptixProgramGroup pg[3]; ls = sizeof log; OX(optixProgramGroupCreate(ctx, d, 3, &go, log, &ls, pg));
    OptixPipelineLinkOptions lo{}; lo.maxTraceDepth = (flip & 4) ? 2 : 1; OptixPipeline pipe; ls = sizeof log; OX(optixPipelineCreate(ctx, &po, &lo, pg, 3, log, &ls, &pipe)); if (flip & 8) OX(optixPipelineSetStackSize(pipe, 2048, 2048, 2048, 2));
    const size_t RS = (flip & 16) ? 48 : 32;   // record size: header 32 bytes, plus 16 bytes of program data with F5 (as OWL: header + the program's struct)
    std::vector<char> rec(3 * RS, 0); for (int k = 0; k < 3; ++k) OX(optixSbtRecordPackHeader(pg[k], rec.data() + k * RS));
    if (flip & 16) { struct RGd { const Particle *p; OptixTraversableHandle h; } rg{(const Particle *)dpt, hW}; struct HGd { const Particle *p; float st, re; } hg{(const Particle *)dpt, 8e3f, 2.f * (1.f / sqrtf(1.f + (float)((M_PI / std::log(0.5)) * (M_PI / std::log(0.5)))))}; static_assert(sizeof(RGd) == 16 && sizeof(HGd) == 16, "sbt data 16 bytes"); memcpy(rec.data() + 32, &rg, 16); memcpy(rec.data() + 2 * RS + 32, &hg, 16); }
    CUdeviceptr drec; CK(cudaMalloc((void **)&drec, rec.size())); CK(cudaMemcpy((void *)drec, rec.data(), rec.size(), cudaMemcpyHostToDevice));
    Params prm{}; prm.n = NQ; prm.particles = (const Particle *)dpt; prm.boxes = (const float *)db; prm.acc = (float3 *)dacc; prm.callw = (unsigned *)dcw; prm.handle = hW; prm.stiffness = 8e3f; prm.restitution = 0.5f * 0 + 2.f * (1.f / sqrtf(1.f + (float)((M_PI / std::log(0.5)) * (M_PI / std::log(0.5)))));
    CK(cudaMalloc((void **)&prm.rec, NW * sizeof(Rec))); CUdeviceptr dprm; CK(cudaMalloc((void **)&dprm, sizeof prm)); CK(cudaMemcpy((void *)dprm, &prm, sizeof prm, cudaMemcpyHostToDevice));
    OptixShaderBindingTable sbt{}; sbt.raygenRecord = drec; sbt.missRecordBase = drec + RS; sbt.missRecordStrideInBytes = RS; sbt.missRecordCount = 1; sbt.hitgroupRecordBase = drec + 2 * RS; sbt.hitgroupRecordStrideInBytes = RS; sbt.hitgroupRecordCount = 1;
    std::vector<Rec> hr(NW); printf("{\"nq\":%u,\"ias\":%d,\"seq\":%d,\"warm\":%d,\"launches\":[", NQ, ias, seq, warm);
    auto sequence = [&]() {
        if (seq >= 1) { upd<<<(NP + 255) / 256, 256>>>((Particle *)dpt, (float3 *)dacc, 1e-4f, NP); CK(cudaDeviceSynchronize()); }
        if (seq >= 2) {
            bnd<<<(NP + 255) / 256, 256>>>((const Particle *)dpt, (float *)db, NP); CK(cudaDeviceSynchronize());
            OptixAccelBuildOptions up = ao; up.operation = OPTIX_BUILD_OPERATION_UPDATE; OptixTraversableHandle hh = hG; OX(optixAccelBuild(ctx, 0, &up, &bi, 1, tmp, tmpsz, gas, bs.outputSizeInBytes, &hh, nullptr, 0)); CK(cudaDeviceSynchronize());
            if (ias) { OptixTraversableHandle h2 = hW; OX(optixAccelBuild(ctx, 0, &up, &ib, 1, tmp2, tmp2sz, ias_, ias_sz, &h2, nullptr, 0)); CK(cudaDeviceSynchronize()); }
        } };
    for (int k = -warm - 1; k < repeat; ++k) {   // k < -1: warmup iterations (Mochi's 200), k = -1: first timed (cold flag), then repeat timed
        sequence(); CK(cudaMemset((void *)dcw, 0, NW * 4)); OX(optixLaunch(pipe, 0, dprm, sizeof prm, &sbt, NQ, 1, 1)); CK(cudaDeviceSynchronize());
        if (k < -1) continue; if (k == 0 && warm == 0) for (int w = 0; w < 2; ++w) { sequence(); CK(cudaMemset((void *)dcw, 0, NW * 4)); OX(optixLaunch(pipe, 0, dprm, sizeof prm, &sbt, NQ, 1, 1)); CK(cudaDeviceSynchronize()); }
        CK(cudaMemcpy(hr.data(), prm.rec, NW * sizeof(Rec), cudaMemcpyDeviceToHost));
        unsigned long long gmin = ~0ull, gmax = 0, calls = 0; double cyc = 0; std::vector<double> wc(NW); for (unsigned w = 0; w < NW; ++w) { gmin = std::min(gmin, hr[w].g0); gmax = std::max(gmax, hr[w].g1); calls += hr[w].calls; wc[w] = (double)(hr[w].c1 - hr[w].c0); cyc += wc[w]; } std::sort(wc.begin(), wc.end());
        printf("%s{\"cold\":%d,\"span_ns\":%llu,\"calls\":%llu,\"warp_cyc_mean\":%.1f,\"wc_p10\":%.0f,\"wc_med\":%.0f,\"wc_p90\":%.0f,\"wc_max\":%.0f}", k > -1 ? "," : "", k == -1, gmax - gmin, calls, cyc / NW, wc[NW / 10], wc[NW / 2], wc[NW * 9 / 10], wc[NW - 1]);
    }
    if (dumpdir) { size_t sz = bs.outputSizeInBytes; std::vector<char> hg(sz); CK(cudaMemcpy(hg.data(), (void *)gas, sz, cudaMemcpyDeviceToHost)); char fn[512]; snprintf(fn, 512, "%s/gas_f00.bin", dumpdir); FILE *fo = fopen(fn, "wbx"); if (!fo) { perror(fn); return 5; } fwrite(hg.data(), 1, sz, fo); fclose(fo);
        std::vector<Particle> hp(NP); CK(cudaMemcpy(hp.data(), (void *)dpt, NP * sizeof(Particle), cudaMemcpyDeviceToHost)); snprintf(fn, 512, "%s/particles_f00.bin", dumpdir); fo = fopen(fn, "wbx"); if (!fo) { perror(fn); return 5; } fwrite(hp.data(), sizeof(Particle), NP, fo); fclose(fo); }
    printf("]}\n"); return 0;
}
