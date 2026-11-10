// Setup_3 host: triangle GAS, an OptiX reference launch, then ttu_one<<<1, 32 * W>>>.
//
//   setup3 PTX WORDS RAY FILL OUT.json [options]
//     PTX    setup3_dev.ptx
//     WORDS  8 hex words: TTUST 0 then TTUST 0x80, low to high (made at build time)
//     RAY    hit | miss | t2 | back | moved | moved2 | hitB | tri:I   (every lane the same ray)
//            tri:I = straight down (+z) onto the centroid of triangle I
//            perlane = lane i onto the centroid of triangle (7919 i + 13) mod N (different paths)
//            jitter  = lane i onto a different point of triangle 0 (same path, different hit data)
//     FILL   zero | random   (value of the words OptiX leaves uninitialized: 0x180 high, 0x190)
//   options
//     --scene S      tri1 (default) | tri2 (2 triangles, 1 build input) | tri2g (2 build inputs) | grid:N
//     --warps W      warps in the single block, 1..21 (default 1); build_patch.py --regs 96 sets the register count
//     --repeat N     timed launches in this process, 1..100 (default 1)
//     --blocks G     blocks of W warps (default 1); with an --smfilter build only blocks on SM A or B run
//     --sms A,B      the SMs allowed to run (VIRTUALSMID; words 18, 19), default 0,0
//     --flush MB     before every ttu_one launch, a plain kernel reads MB megabytes (L2 eviction)
//     --dump-gas F   write the GAS bytes to F
//     --no-ttu       stop after the OptiX reference
//
// ttu_one as compiled here only loads 96 words into registers and stores them
// back. build_patch.py rewrites its body (stage 0: no TTU code; stage 1: one
// TTU group). Run the patched copy only through run.py.
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include "setup3_params.h"

#define N 96
#define MAXW 21   // 21 x 32 x 96 registers fit the 65536 register file
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(2); } } while (0)
#define OX(x) do { OptixResult r_ = (x); if (r_ != OPTIX_SUCCESS) { \
    fprintf(stderr, "%s:%d %s: %d\n", __FILE__, __LINE__, #x, (int)r_); exit(2); } } while (0)

// Template for build_patch.py: 96 loads into live registers, then 96 stores.
// Load k and store k use word k of this lane's record (= register k after patching).
extern "C" __global__ void ttu_one(const unsigned *in, unsigned *out)
{
    const unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
    const volatile unsigned *p = in + t * N;
    unsigned r[N];
#pragma unroll
    for (int k = 0; k < N; ++k) r[k] = p[k];
    __syncthreads();
    volatile unsigned *q = out + t * N;
#pragma unroll
    for (int k = 0; k < N; ++k) q[k] = r[k];
}

// Plain kernel: read a large buffer so the L2 (96 MB) holds none of the BVH.
extern "C" __global__ void flush_read(const uint4 *p, size_t n, unsigned *sink)
{
    unsigned acc = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        const uint4 v = p[i]; acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x12345678u) sink[0] = acc;   // keep the loads
}

static std::string read_file(const char *path)
{
    std::ifstream f(path); if (!f) { fprintf(stderr, "cannot read %s\n", path); exit(2); }
    std::stringstream ss; ss << f.rdbuf(); return ss.str();
}
static unsigned fbits(float x) { unsigned u; memcpy(&u, &x, 4); return u; }

struct Tri { float3 a, b, c; };

static std::vector<std::vector<Tri>> make_scene(const std::string &s)
{
    const Tri t0{{-0.5f, -0.5f, 0.f}, {0.5f, -0.5f, 0.f}, {0.f, 0.5f, 0.f}};
    const Tri t1{{1.5f, -0.5f, 0.f}, {2.5f, -0.5f, 0.f}, {2.f, 0.5f, 0.f}};
    if (s == "tri1") return {{t0}};
    if (s == "tri2") return {{t0, t1}};
    if (s == "tri2g") return {{t0}, {t1}};
    if (s.rfind("grid:", 0) == 0) {
        const int n = atoi(s.c_str() + 5);
        if (n < 1 || n > 1 << 20) { fprintf(stderr, "grid:N with 1 <= N <= 2^20\n"); exit(2); }
        const int k = (int)std::ceil(std::sqrt((double)n));
        const float c = 2.f / k;
        std::vector<Tri> v;
        for (int i = 0; i < n; ++i) {
            const float x = -1.f + (i % k) * c, y = -1.f + (i / k) * c;
            v.push_back({{x + 0.1f * c, y + 0.1f * c, 0.f}, {x + 0.9f * c, y + 0.1f * c, 0.f}, {x + 0.5f * c, y + 0.9f * c, 0.f}});
        }
        return {v};
    }
    fprintf(stderr, "unknown scene %s\n", s.c_str()); exit(2);
}

int main(int argc, char **argv)
{
    if (argc < 6) { fprintf(stderr, "usage: see header of setup3.cu\n"); return 2; }
    const bool random_fill = !strcmp(argv[4], "random");
    bool run_ttu = true; int repeat = 1, warps = 1, blocks = 1; unsigned sm_a = 0, sm_b = 0; double flush_mb = 0; std::string scene = "tri1"; const char *dump = nullptr;
    for (int a = 6; a < argc; ++a) {
        if (!strcmp(argv[a], "--no-ttu")) run_ttu = false;
        else if (!strcmp(argv[a], "--repeat") && a + 1 < argc) repeat = atoi(argv[++a]);
        else if (!strcmp(argv[a], "--warps") && a + 1 < argc) warps = atoi(argv[++a]);
        else if (!strcmp(argv[a], "--blocks") && a + 1 < argc) blocks = atoi(argv[++a]);
        else if (!strcmp(argv[a], "--sms") && a + 1 < argc) { if (sscanf(argv[++a], "%u,%u", &sm_a, &sm_b) != 2) { fprintf(stderr, "--sms A,B\n"); return 2; } }
        else if (!strcmp(argv[a], "--flush") && a + 1 < argc) flush_mb = atof(argv[++a]);
        else if (!strcmp(argv[a], "--scene") && a + 1 < argc) scene = argv[++a];
        else if (!strcmp(argv[a], "--dump-gas") && a + 1 < argc) dump = argv[++a];
        else { fprintf(stderr, "unknown option %s\n", argv[a]); return 2; }
    }
    if (repeat < 1 || repeat > 100 || warps < 1 || warps > MAXW || blocks < 1 || blocks > 1024 || flush_mb < 0 || flush_mb > 4096) {
        fprintf(stderr, "bad option value\n"); return 2;
    }
    unsigned words[8];
    { std::istringstream ws(read_file(argv[2])); for (auto &w : words) ws >> std::hex >> w; }
    { cudaFuncAttributes fa; CK(cudaFuncGetAttributes(&fa, ttu_one));
      printf("ttu_one: numRegs %d maxThreadsPerBlock %d localBytes %zu\n", fa.numRegs, fa.maxThreadsPerBlock, fa.localSizeBytes); }

    // ---- OptiX GAS ------------------------------------------------------------
    CK(cudaFree(0)); OX(optixInit());
    OptixDeviceContextOptions co{}; OptixDeviceContext ctx; OX(optixDeviceContextCreate(0, &co, &ctx));
    const auto geoms = make_scene(scene);
    const unsigned ng = (unsigned)geoms.size();
    std::vector<CUdeviceptr> dverts(ng); std::vector<OptixBuildInput> bis(ng);
    unsigned gflags = OPTIX_GEOMETRY_FLAG_NONE;
    std::vector<Tri> all;
    for (unsigned g = 0; g < ng; ++g) {
        const auto &v = geoms[g]; all.insert(all.end(), v.begin(), v.end());
        CK(cudaMalloc((void **)&dverts[g], v.size() * sizeof(Tri)));
        CK(cudaMemcpy((void *)dverts[g], v.data(), v.size() * sizeof(Tri), cudaMemcpyHostToDevice));
        OptixBuildInput &bi = bis[g]; bi = {}; bi.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        bi.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3; bi.triangleArray.numVertices = (unsigned)v.size() * 3;
        bi.triangleArray.vertexBuffers = &dverts[g]; bi.triangleArray.flags = &gflags; bi.triangleArray.numSbtRecords = 1;
    }
    OptixAccelBuildOptions ao{}; ao.buildFlags = OPTIX_BUILD_FLAG_NONE; ao.operation = OPTIX_BUILD_OPERATION_BUILD;
    OptixAccelBufferSizes bs{}; OX(optixAccelComputeMemoryUsage(ctx, &ao, bis.data(), ng, &bs));
    CUdeviceptr tmp, gas; CK(cudaMalloc((void **)&tmp, bs.tempSizeInBytes)); CK(cudaMalloc((void **)&gas, bs.outputSizeInBytes));
    OptixTraversableHandle h;
    OX(optixAccelBuild(ctx, 0, &ao, bis.data(), ng, tmp, bs.tempSizeInBytes, gas, bs.outputSizeInBytes, &h, nullptr, 0));
    CK(cudaDeviceSynchronize()); CK(cudaFree((void *)tmp));
    if (dump) {
        std::vector<char> b(bs.outputSizeInBytes);
        CK(cudaMemcpy(b.data(), (void *)gas, b.size(), cudaMemcpyDeviceToHost));
        FILE *f = fopen(dump, "wb"); fwrite(b.data(), 1, b.size(), f); fclose(f);
    }

    // ---- pipeline -------------------------------------------------------------
    OptixModuleCompileOptions mo{}; OptixPipelineCompileOptions po{};
    po.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    po.numPayloadValues = 7; po.numAttributeValues = 2;
    po.pipelineLaunchParamsVariableName = "params";
    po.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
    std::string ptx = read_file(argv[1]); char log[4096]; size_t ls = sizeof log; OptixModule m;
    OX(optixModuleCreate(ctx, &mo, &po, ptx.c_str(), ptx.size(), log, &ls, &m));
    OptixProgramGroupOptions go{}; OptixProgramGroupDesc d[3] = {};
    d[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN; d[0].raygen.module = m; d[0].raygen.entryFunctionName = "__raygen__rg";
    d[1].kind = OPTIX_PROGRAM_GROUP_KIND_MISS; d[1].miss.module = m; d[1].miss.entryFunctionName = "__miss__ms";
    d[2].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP; d[2].hitgroup.moduleCH = m; d[2].hitgroup.entryFunctionNameCH = "__closesthit__ch";
    OptixProgramGroup g[3]; ls = sizeof log; OX(optixProgramGroupCreate(ctx, d, 3, &go, log, &ls, g));
    OptixPipelineLinkOptions lo{}; lo.maxTraceDepth = 1; OptixPipeline pipe;
    ls = sizeof log; OX(optixPipelineCreate(ctx, &po, &lo, g, 3, log, &ls, &pipe));
    struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) Rec { char h[OPTIX_SBT_RECORD_HEADER_SIZE]; };
    std::vector<Rec> rec(2 + ng);   // raygen, miss, one hit record per build input
    OX(optixSbtRecordPackHeader(g[0], &rec[0])); OX(optixSbtRecordPackHeader(g[1], &rec[1]));
    for (unsigned k = 0; k < ng; ++k) OX(optixSbtRecordPackHeader(g[2], &rec[2 + k]));
    CUdeviceptr drec; CK(cudaMalloc((void **)&drec, rec.size() * sizeof(Rec)));
    CK(cudaMemcpy((void *)drec, rec.data(), rec.size() * sizeof(Rec), cudaMemcpyHostToDevice));
    OptixShaderBindingTable sbt{}; sbt.raygenRecord = drec;
    sbt.missRecordBase = drec + sizeof(Rec); sbt.missRecordStrideInBytes = sizeof(Rec); sbt.missRecordCount = 1;
    sbt.hitgroupRecordBase = drec + 2 * sizeof(Rec); sbt.hitgroupRecordStrideInBytes = sizeof(Rec); sbt.hitgroupRecordCount = ng;

    // ---- ray and OptiX reference --------------------------------------------------
    const unsigned LANES = 32u * warps;
    struct Named { const char *name; Ray8 r; };
    const Named named[] = {   // keep in sync with ref.py RAYS
        {"hit",    {{0.f, 0.f, -1.f}, 0.f, {0.f, 0.f, 1.f}, 1e16f}},
        {"miss",   {{2.f, 2.f, -1.f}, 0.f, {0.f, 0.f, 1.f}, 1e16f}},
        {"t2",     {{0.f, 0.f, -2.f}, 0.f, {0.f, 0.f, 1.f}, 1e16f}},
        {"back",   {{0.f, 0.f, 1.f}, 0.f, {0.f, 0.f, -1.f}, 1e16f}},
        {"moved",  {{0.2f, -0.3f, -1.f}, 0.f, {0.f, 0.f, 1.f}, 1e16f}},
        {"moved2", {{-0.3f, -0.4f, -1.f}, 0.f, {0.f, 0.f, 1.f}, 1e16f}},
        {"hitB",   {{2.f, 0.f, -1.f}, 0.f, {0.f, 0.f, 1.f}, 1e16f}},
    };
    Ray8 ray{};
    const bool perlane = !strcmp(argv[3], "perlane"), jitter = !strcmp(argv[3], "jitter");
    if (perlane || jitter) {
        ray = {{0.f, 0.f, -1.f}, 0.f, {0.f, 0.f, 1.f}, 1e16f};   // replaced per lane below
    } else if (!strncmp(argv[3], "tri:", 4)) {
        const int i = atoi(argv[3] + 4);
        if (i < 0 || i >= (int)all.size()) { fprintf(stderr, "tri:I out of range\n"); return 2; }
        const Tri &t = all[i];
        ray = {{(t.a.x + t.b.x + t.c.x) / 3.f, (t.a.y + t.b.y + t.c.y) / 3.f, -1.f}, 0.f, {0.f, 0.f, 1.f}, 1e16f};
    } else {
        const Ray8 *pick = nullptr;
        for (const Named &n : named) if (!strcmp(argv[3], n.name)) pick = &n.r;
        if (!pick) { fprintf(stderr, "unknown ray %s\n", argv[3]); return 2; }
        ray = *pick;
    }
    std::vector<Ray8> rays(LANES, ray);
    for (unsigned i = 0; i < LANES && (perlane || jitter); ++i) {
        const Tri &t = all[perlane ? (size_t)((7919ull * i + 13) % all.size()) : 0];
        float u = 1.f / 3, v = 1.f / 3;
        if (jitter) { u = ((i % 8) + 0.5f) / 16.f; v = (((i / 8) % 8) + 0.5f) / 16.f; }   // u, v < 0.5
        const float w = 1.f - u - v;
        rays[i] = {{w * t.a.x + u * t.b.x + v * t.c.x, w * t.a.y + u * t.b.y + v * t.c.y, -1.f}, 0.f, {0.f, 0.f, 1.f}, 1e16f};
    }
    Params prm{}; prm.handle = h; prm.n = LANES;
    CK(cudaMalloc((void **)&prm.rays, LANES * sizeof(Ray8)));
    CK(cudaMemcpy((void *)prm.rays, rays.data(), LANES * sizeof(Ray8), cudaMemcpyHostToDevice));
    CK(cudaMalloc((void **)&prm.out, LANES * sizeof(RefOut))); CK(cudaMalloc((void **)&prm.inst, 18 * 8));
    CUdeviceptr dprm; CK(cudaMalloc((void **)&dprm, sizeof prm)); CK(cudaMemcpy((void *)dprm, &prm, sizeof prm, cudaMemcpyHostToDevice));
    OX(optixLaunch(pipe, 0, dprm, sizeof prm, &sbt, LANES, 1, 1));
    CK(cudaDeviceSynchronize());
    std::vector<RefOut> ref(LANES); unsigned long long inst[18];
    CK(cudaMemcpy(ref.data(), prm.out, LANES * sizeof(RefOut), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(inst, prm.inst, sizeof inst, cudaMemcpyDeviceToHost));

    // ---- checks before anything is handed to ttu_one -----------------------------
    const unsigned long long A = inst[0];
    const unsigned ident[16] = {0, 1, 0, 0, 0x3f800000, 0, 0, 0, 0, 0x3f800000, 0, 0, 0, 0, 0x3f800000, 0};
    bool inst_ok = true; for (int k = 0; k < 16; ++k) inst_ok &= (unsigned)inst[1 + k] == ident[k];
    const unsigned c3_34 = (unsigned)inst[17];
    const bool c3_ok = c3_34 == 0x00011000u;   // the store words were computed for this value
    unsigned long long off = 0; CK(cudaMemcpy(&off, (void *)(gas + 0x20), 8, cudaMemcpyDeviceToHost));
    const unsigned long long root = (unsigned long long)gas + off;
    const bool handle_ok = (h & ~0x3fULL) == (unsigned long long)gas && (h & 0x3f) == 0xc && (gas & 0x7f) == 0;
    const bool align_ok = (A & 0x3f) == 0 && (root & 0x7f) == 0;

    FILE *jf = fopen(argv[5], "w");
    fprintf(jf, "{\n \"scene\": \"%s\", \"geoms\": %u, \"tris\": [", scene.c_str(), ng);
    if (all.size() <= 64)
        for (size_t i = 0; i < all.size(); ++i) {
            const Tri &t = all[i];
            fprintf(jf, "%s[[%.9g,%.9g,%.9g],[%.9g,%.9g,%.9g],[%.9g,%.9g,%.9g]]", i ? "," : "",
                    t.a.x, t.a.y, t.a.z, t.b.x, t.b.y, t.b.z, t.c.x, t.c.y, t.c.z);
        }
    fprintf(jf, "], \"ntris\": %zu,\n \"ray_name\": \"%s\", \"ray\": [[%.9g,%.9g,%.9g],[%.9g,%.9g,%.9g]],\n", all.size(), argv[3],
            ray.o[0], ray.o[1], ray.o[2], ray.d[0], ray.d[1], ray.d[2]);
    fprintf(jf, " \"gas\": \"0x%llx\", \"gas_size\": %zu, \"handle\": \"0x%llx\", \"root_off\": \"0x%llx\", \"root\": \"0x%llx\",\n",
            (unsigned long long)gas, (size_t)bs.outputSizeInBytes, (unsigned long long)h, off, root);
    fprintf(jf, " \"inst\": \"0x%llx\", \"inst_ok\": %s, \"c3_34\": \"0x%08x\", \"handle_ok\": %s, \"align_ok\": %s,\n", A,
            inst_ok ? "true" : "false", c3_34, handle_ok ? "true" : "false", align_ok ? "true" : "false");
    fprintf(jf, " \"lane_rays\": [");
    for (unsigned i = 0; i < LANES && i < 32; ++i)
        fprintf(jf, "%s[%.9g,%.9g,%.9g]", i ? "," : "", rays[i].o[0], rays[i].o[1], rays[i].o[2]);
    fprintf(jf, "],\n");
    fprintf(jf, " \"warps\": %d, \"flush_mb\": %g, \"fill\": \"%s\",\n \"ref\": [", warps, flush_mb, argv[4]);
    for (unsigned i = 0; i < LANES; ++i)
        fprintf(jf, "%s{\"hit\": %u, \"t\": \"0x%08x\", \"prim\": %u, \"u\": \"0x%08x\", \"v\": \"0x%08x\", \"front\": %u, \"gas_index\": %u}",
                i ? ", " : "", ref[i].hit, fbits(ref[i].t), ref[i].prim, fbits(ref[i].u), fbits(ref[i].v), ref[i].front, ref[i].gas_index);
    fprintf(jf, "]");
    printf("scene %s (%zu tris, %u inputs) gas 0x%llx size %zu root 0x%llx inst 0x%llx inst_ok %d c3_34 0x%x handle_ok %d align_ok %d\n",
           scene.c_str(), all.size(), ng, (unsigned long long)gas, (size_t)bs.outputSizeInBytes, root, A, inst_ok, c3_34, handle_ok, align_ok);
    printf("optix ref lane 0: hit %u t %g prim %u front %u gas_index %u\n", ref[0].hit, ref[0].t, ref[0].prim, ref[0].front, ref[0].gas_index);
    if (!inst_ok || !handle_ok || !align_ok || !c3_ok) {
        fprintf(jf, "\n}\n"); fclose(jf); fprintf(stderr, "check failed, ttu_one not launched\n"); return 3;
    }
    if (!run_ttu) { fprintf(jf, "\n}\n"); fclose(jf); return 0; }

    // ---- register image for ttu_one: word r of a thread = register r --------------
    const size_t TOTAL = (size_t)blocks * LANES;
    std::vector<unsigned> in(TOTAL * N, 0), out(TOTAL * N, 0x77777777u);
    for (size_t i = 0; i < TOTAL; ++i) {
        unsigned *R = &in[i * N];
        const Ray8 &y = rays[i % LANES];
        R[12] = fbits(y.o[2]); R[13] = fbits(y.tmin); R[82] = fbits(y.o[0]); R[83] = fbits(y.o[1]);   // 0x40
        R[16] = fbits(y.d[2]); R[17] = fbits(y.tmax); R[14] = fbits(y.d[0]); R[15] = fbits(y.d[1]);   // 0x50
        R[52] = R[12]; R[53] = R[13]; R[56] = R[82]; R[57] = R[83];                                   // 0xC0
        R[30] = R[16]; R[31] = R[17]; R[46] = R[14]; R[47] = R[15];                                   // 0xD0
        R[54] = 0; R[55] = 0;                                                                          // 0x100 high
        R[34] = (unsigned)(A >> 6); R[35] = ((((unsigned)(A >> 32) >> 6) | 0x402800u) & 0x1ffffffu) | 0x2000000u;
        R[70] = (unsigned)(root >> 7); R[71] = ((unsigned)(root >> 32) >> 7) | 0x2484000u;           // 0x180 low
        unsigned *unk[6] = {&R[32], &R[33], &R[2], &R[3], &R[44], &R[45]};                             // 0x180 high, 0x190
        std::mt19937 lane_rng(12345);   // same pattern in every lane
        for (auto u : unk) *u = random_fill ? (unsigned)lane_rng() : 0u;
        R[68] = words[0]; R[69] = words[1]; R[62] = words[2]; R[63] = words[3];                        // 0
        R[60] = words[4]; R[61] = words[5]; R[58] = words[6]; R[59] = words[7];                        // 0x80
        R[90] = 0;   // (R37 & 0xC0000) for flags NONE: P0 true
        R[18] = sm_a; R[19] = sm_b;   // used only by --smfilter builds
    }
    unsigned *din, *dout, *sink; uint4 *fbuf = nullptr; size_t fn = (size_t)(flush_mb * 1048576.0) / sizeof(uint4);
    CK(cudaMalloc((void **)&din, in.size() * 4)); CK(cudaMalloc((void **)&dout, out.size() * 4)); CK(cudaMalloc((void **)&sink, 4));
    CK(cudaMemcpy(din, in.data(), in.size() * 4, cudaMemcpyHostToDevice));
    if (fn) { CK(cudaMalloc((void **)&fbuf, fn * sizeof(uint4))); CK(cudaMemset(fbuf, 1, fn * sizeof(uint4))); }

    // Each launch: one group per lane. Clock words R84:R85 (before TTUOPEN) and R86:R87
    // (after all results), global timer R22:R23 / R24:R25, SM id R20, if patched in.
    // A block ran if its lane 0 stored anything (the output is filled with 0x77777777 first).
    auto ran = [&](const std::vector<unsigned> &o, int b) { return o[(size_t)b * LANES * N + 84] != 0x77777777u || o[(size_t)b * LANES * N + 12] != 0x77777777u; };
    auto u64 = [](const unsigned *R, int r) { return ((unsigned long long)R[r + 1] << 32) | R[r]; };
    cudaError_t e = cudaSuccess;
    std::vector<unsigned> first;
    std::vector<int> ran_blocks;
    fprintf(jf, ",\n \"blocks\": %d, \"sms\": [%u, %u],\n \"launches\": [", blocks, sm_a, sm_b);
    bool same_all = true;
    for (int it = 0; it < repeat && e == cudaSuccess; ++it) {
        CK(cudaMemcpy(dout, out.data(), out.size() * 4, cudaMemcpyHostToDevice));
        if (fn) { flush_read<<<142 * 8, 256>>>(fbuf, fn, sink); CK(cudaGetLastError()); }
        ttu_one<<<blocks, LANES>>>(din, dout);
        e = cudaGetLastError();                     // launch errors (e.g. too many resources)
        if (e == cudaSuccess) e = cudaDeviceSynchronize();
        if (e != cudaSuccess) break;
        std::vector<unsigned> o(out.size());
        CK(cudaMemcpy(o.data(), dout, o.size() * 4, cudaMemcpyDeviceToHost));
        if (it == 0) first = o;
        std::vector<int> rb;
        for (int b = 0; b < blocks; ++b) if (ran(o, b)) rb.push_back(b);
        if (it == 0) ran_blocks = rb;
        bool same = rb == ran_blocks;
        for (int b : rb)
            for (unsigned i = 0; i < LANES; ++i)
                for (int r : {2, 3, 30, 31, 32, 33, 34, 35, 44, 45, 56, 57, 58, 59, 60, 61, 62, 63, 68, 69, 70, 71})
                    same &= o[((size_t)b * LANES + i) * N + r] == first[((size_t)b * LANES + i) * N + r];
        same_all &= same;
        bool lanes_eq = true;
        fprintf(jf, "%s{\"same_as_first\": %s, \"ran\": %zu, \"per_block\": [", it ? ", " : "", same ? "true" : "false", rb.size());
        std::string wc0;
        for (size_t bi = 0; bi < rb.size(); ++bi) {
            const int b = rb[bi];
            const unsigned *B0 = &o[(size_t)b * LANES * N];
            fprintf(jf, "%s{\"block\": %d, \"sm\": %u, \"warp_clk\": [", bi ? ", " : "", b, B0[20]);
            std::string wc;
            for (int w = 0; w < warps; ++w) {
                const unsigned *L0 = &B0[(32 * w) * N];
                for (int l = 1; l < 32; ++l) {
                    const unsigned *Ll = &B0[(32 * w + l) * N];
                    lanes_eq &= Ll[84] == L0[84] && Ll[85] == L0[85] && Ll[86] == L0[86] && Ll[87] == L0[87];
                }
                char buf[64]; snprintf(buf, sizeof buf, "%s[%llu,%llu]", w ? "," : "", u64(L0, 84), u64(L0, 86)); wc += buf;
            }
            fprintf(jf, "%s], \"gt\": [", wc.c_str());
            for (int w = 0; w < warps; ++w) {
                const unsigned *L0 = &B0[(32 * w) * N];
                fprintf(jf, "%s[%llu,%llu]", w ? "," : "", u64(L0, 22), u64(L0, 24));
            }
            fprintf(jf, "]}");
            if (bi == 0) wc0 = wc;
        }
        fprintf(jf, "], \"warp_clk\": [%s], \"lanes_equal\": %s}", wc0.c_str(), lanes_eq ? "true" : "false");
        if (it == repeat - 1) out = o;
    }
    fprintf(jf, "]");
    printf("ttu_one: %s (%d launch(es) x %d block(s) x %d warp(s), %zu block(s) ran, flush %g MB, results same every launch: %d)\n",
           cudaGetErrorString(e), repeat, blocks, warps, ran_blocks.size(), flush_mb, same_all);
    const int b0 = ran_blocks.empty() ? 0 : ran_blocks[0];
    fprintf(jf, ",\n \"ttu_status\": \"%s\",\n \"in\": [", cudaGetErrorString(e));
    const unsigned keep = LANES < 32 ? LANES : 32;   // first warp of the first block that ran
    for (unsigned i = 0; i < keep; ++i) {
        fprintf(jf, "%s[", i ? ", " : "");
        for (unsigned k = 0; k < N; ++k) fprintf(jf, "%s%u", k ? "," : "", in[((size_t)b0 * LANES + i) * N + k]);
        fprintf(jf, "]");
    }
    fprintf(jf, "],\n \"out\": [");
    for (unsigned i = 0; i < keep; ++i) {
        fprintf(jf, "%s[", i ? ", " : "");
        for (unsigned k = 0; k < N; ++k) fprintf(jf, "%s%u", k ? "," : "", out[((size_t)b0 * LANES + i) * N + k]);
        fprintf(jf, "]");
    }
    // lane 0 of every warp of every block that ran, last launch
    fprintf(jf, "],\n \"warp_out\": [");
    { bool firstw = true;
      for (int b : ran_blocks)
        for (int w = 0; w < warps; ++w) {
            fprintf(jf, "%s[", firstw ? "" : ", "); firstw = false;
            for (unsigned k = 0; k < N; ++k) fprintf(jf, "%s%u", k ? "," : "", out[((size_t)b * LANES + 32 * w) * N + k]);
            fprintf(jf, "]");
        } }
    fprintf(jf, "]\n}\n"); fclose(jf);
    return e == cudaSuccess ? 0 : 1;   // pipeline and GAS stay alive until here
}
