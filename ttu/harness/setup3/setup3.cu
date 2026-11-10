// Setup_3 host: one triangle GAS, an OptiX reference launch, then ttu_one<<<1, 32>>>.
//
//   setup3 PTX WORDS RAYS FILL OUT.json [--no-ttu] [--repeat N]
//     PTX    setup3_dev.ptx
//     WORDS  8 hex words: TTUST 0 then TTUST 0x80, low to high (made at build time)
//     RAYS   hit | miss | t2 | back | moved | moved2   (all 32 lanes the same ray; see ref.py)
//     FILL   zero | random   (value of the words OptiX leaves uninitialized: 0x180 high, 0x190)
//
// ttu_one as compiled here only loads 96 words into registers and stores them
// back. build_patch.py rewrites its body (stage 0: no TTU code; stage 1: one
// TTU group). Run the patched copy only through run.py.
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
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
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(2); } } while (0)
#define OX(x) do { OptixResult r_ = (x); if (r_ != OPTIX_SUCCESS) { \
    fprintf(stderr, "%s:%d %s: %d\n", __FILE__, __LINE__, #x, (int)r_); exit(2); } } while (0)

// Template for build_patch.py: 96 loads into live registers, then 96 stores.
// Load k and store k use word k of this lane's record (= register k after patching).
extern "C" __global__ void __launch_bounds__(32) ttu_one(const unsigned *in, unsigned *out)
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

static std::string read_file(const char *path)
{
    std::ifstream f(path); if (!f) { fprintf(stderr, "cannot read %s\n", path); exit(2); }
    std::stringstream ss; ss << f.rdbuf(); return ss.str();
}
static unsigned fbits(float x) { unsigned u; memcpy(&u, &x, 4); return u; }

int main(int argc, char **argv)
{
    if (argc < 6) { fprintf(stderr, "usage: setup3 PTX WORDS RAYS FILL OUT.json [--no-ttu]\n"); return 2; }
    const bool random_fill = !strcmp(argv[4], "random");
    bool run_ttu = true; int repeat = 1;
    for (int a = 6; a < argc; ++a) {
        if (!strcmp(argv[a], "--no-ttu")) run_ttu = false;
        else if (!strcmp(argv[a], "--repeat") && a + 1 < argc) repeat = atoi(argv[++a]);
    }
    if (repeat < 1 || repeat > 100) { fprintf(stderr, "repeat 1..100\n"); return 2; }
    unsigned words[8];
    { std::istringstream ws(read_file(argv[2])); for (auto &w : words) ws >> std::hex >> w; }

    // ---- OptiX: one triangle GAS (same as memdump) --------------------------
    CK(cudaFree(0)); OX(optixInit());
    OptixDeviceContextOptions co{}; OptixDeviceContext ctx; OX(optixDeviceContextCreate(0, &co, &ctx));
    float3 v[3] = {{-0.5f, -0.5f, 0.f}, {0.5f, -0.5f, 0.f}, {0.f, 0.5f, 0.f}};
    CUdeviceptr dv; CK(cudaMalloc((void **)&dv, sizeof v)); CK(cudaMemcpy((void *)dv, v, sizeof v, cudaMemcpyHostToDevice));
    unsigned gflags = OPTIX_GEOMETRY_FLAG_NONE;
    OptixBuildInput bi{}; bi.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    bi.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3; bi.triangleArray.numVertices = 3;
    bi.triangleArray.vertexBuffers = &dv; bi.triangleArray.flags = &gflags; bi.triangleArray.numSbtRecords = 1;
    OptixAccelBuildOptions ao{}; ao.buildFlags = OPTIX_BUILD_FLAG_NONE; ao.operation = OPTIX_BUILD_OPERATION_BUILD;
    OptixAccelBufferSizes bs{}; OX(optixAccelComputeMemoryUsage(ctx, &ao, &bi, 1, &bs));
    CUdeviceptr tmp, gas; CK(cudaMalloc((void **)&tmp, bs.tempSizeInBytes)); CK(cudaMalloc((void **)&gas, bs.outputSizeInBytes));
    OptixTraversableHandle h;
    OX(optixAccelBuild(ctx, 0, &ao, &bi, 1, tmp, bs.tempSizeInBytes, gas, bs.outputSizeInBytes, &h, nullptr, 0));
    CK(cudaDeviceSynchronize()); CK(cudaFree((void *)tmp));

    OptixModuleCompileOptions mo{}; OptixPipelineCompileOptions po{};
    po.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    po.numPayloadValues = 5; po.numAttributeValues = 2;
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
    Rec rec[3]; for (int k = 0; k < 3; ++k) OX(optixSbtRecordPackHeader(g[k], &rec[k]));
    CUdeviceptr drec; CK(cudaMalloc((void **)&drec, sizeof rec)); CK(cudaMemcpy((void *)drec, rec, sizeof rec, cudaMemcpyHostToDevice));
    OptixShaderBindingTable sbt{}; sbt.raygenRecord = drec;
    sbt.missRecordBase = drec + sizeof(Rec); sbt.missRecordStrideInBytes = sizeof(Rec); sbt.missRecordCount = 1;
    sbt.hitgroupRecordBase = drec + 2 * sizeof(Rec); sbt.hitgroupRecordStrideInBytes = sizeof(Rec); sbt.hitgroupRecordCount = 1;

    // ---- rays and OptiX reference ---------------------------------------------
    const unsigned LANES = 32;
    struct Named { const char *name; Ray8 r; };
    const Named named[] = {   // keep in sync with ref.py RAYS
        {"hit",    {{0.f, 0.f, -1.f}, 0.f, {0.f, 0.f, 1.f}, 1e16f}},
        {"miss",   {{2.f, 2.f, -1.f}, 0.f, {0.f, 0.f, 1.f}, 1e16f}},
        {"t2",     {{0.f, 0.f, -2.f}, 0.f, {0.f, 0.f, 1.f}, 1e16f}},
        {"back",   {{0.f, 0.f, 1.f}, 0.f, {0.f, 0.f, -1.f}, 1e16f}},
        {"moved",  {{0.2f, -0.3f, -1.f}, 0.f, {0.f, 0.f, 1.f}, 1e16f}},
        {"moved2", {{-0.3f, -0.4f, -1.f}, 0.f, {0.f, 0.f, 1.f}, 1e16f}},
    };
    const Ray8 *pick = nullptr;
    for (const Named &n : named) if (!strcmp(argv[3], n.name)) pick = &n.r;
    if (!pick) { fprintf(stderr, "unknown ray %s\n", argv[3]); return 2; }
    const Ray8 ray = *pick;
    std::vector<Ray8> rays(LANES, ray);
    Params prm{}; prm.handle = h; prm.n = LANES;
    CK(cudaMalloc((void **)&prm.rays, LANES * sizeof(Ray8)));
    CK(cudaMemcpy((void *)prm.rays, rays.data(), LANES * sizeof(Ray8), cudaMemcpyHostToDevice));
    CK(cudaMalloc((void **)&prm.out, LANES * sizeof(RefOut))); CK(cudaMalloc((void **)&prm.inst, 17 * 8));
    CUdeviceptr dprm; CK(cudaMalloc((void **)&dprm, sizeof prm)); CK(cudaMemcpy((void *)dprm, &prm, sizeof prm, cudaMemcpyHostToDevice));
    OX(optixLaunch(pipe, 0, dprm, sizeof prm, &sbt, LANES, 1, 1));
    CK(cudaDeviceSynchronize());
    std::vector<RefOut> ref(LANES); unsigned long long inst[17];
    CK(cudaMemcpy(ref.data(), prm.out, LANES * sizeof(RefOut), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(inst, prm.inst, sizeof inst, cudaMemcpyDeviceToHost));

    // ---- checks before anything is handed to ttu_one -----------------------------
    const unsigned long long A = inst[0];
    const unsigned ident[16] = {0, 1, 0, 0, 0x3f800000, 0, 0, 0, 0, 0x3f800000, 0, 0, 0, 0, 0x3f800000, 0};
    bool inst_ok = true; for (int k = 0; k < 16; ++k) inst_ok &= (unsigned)inst[1 + k] == ident[k];
    unsigned long long off = 0; CK(cudaMemcpy(&off, (void *)(gas + 0x20), 8, cudaMemcpyDeviceToHost));
    const unsigned long long root = (unsigned long long)gas + off;
    const bool handle_ok = (h & ~0x3fULL) == (unsigned long long)gas && (h & 0x3f) == 0xc && (gas & 0x7f) == 0;
    const bool align_ok = (A & 0x3f) == 0 && (root & 0x7f) == 0;

    FILE *jf = fopen(argv[5], "w");
    fprintf(jf, "{\n \"gas\": \"0x%llx\", \"gas_size\": %zu, \"handle\": \"0x%llx\", \"root_off\": \"0x%llx\", \"root\": \"0x%llx\",\n",
            (unsigned long long)gas, (size_t)bs.outputSizeInBytes, (unsigned long long)h, off, root);
    fprintf(jf, " \"inst\": \"0x%llx\", \"inst_ok\": %s, \"handle_ok\": %s, \"align_ok\": %s,\n", A,
            inst_ok ? "true" : "false", handle_ok ? "true" : "false", align_ok ? "true" : "false");
    fprintf(jf, " \"ray\": \"%s\", \"fill\": \"%s\",\n \"ref\": [", argv[3], argv[4]);
    for (unsigned i = 0; i < LANES; ++i)
        fprintf(jf, "%s{\"hit\": %u, \"t\": \"0x%08x\", \"prim\": %u, \"u\": \"0x%08x\", \"v\": \"0x%08x\"}", i ? ", " : "",
                ref[i].hit, fbits(ref[i].t), ref[i].prim, fbits(ref[i].u), fbits(ref[i].v));
    fprintf(jf, "]");
    printf("gas 0x%llx handle 0x%llx root 0x%llx inst 0x%llx inst_ok %d handle_ok %d align_ok %d\n",
           (unsigned long long)gas, (unsigned long long)h, root, A, inst_ok, handle_ok, align_ok);
    printf("optix ref lane 0: hit %u t %g prim %u\n", ref[0].hit, ref[0].t, ref[0].prim);
    if (!inst_ok || !handle_ok || !align_ok) { fprintf(jf, "\n}\n"); fclose(jf); fprintf(stderr, "check failed, ttu_one not launched\n"); return 3; }
    if (!run_ttu) { fprintf(jf, "\n}\n"); fclose(jf); return 0; }

    // ---- register image for ttu_one: word r of a lane = register r --------------
    std::vector<unsigned> in(LANES * N, 0), out(LANES * N, 0x77777777u);
    for (unsigned i = 0; i < LANES; ++i) {
        unsigned *R = &in[i * N];
        const Ray8 &y = rays[i];
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
    }
    unsigned *din, *dout;
    CK(cudaMalloc((void **)&din, in.size() * 4)); CK(cudaMalloc((void **)&dout, out.size() * 4));
    CK(cudaMemcpy(din, in.data(), in.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dout, out.data(), out.size() * 4, cudaMemcpyHostToDevice));
    // Each launch: one group per lane. Launch 0 is cold; the rest find the BVH in L2.
    // Clock words R84:R85 (before TTUOPEN) and R86:R87 (after results), if patched in.
    cudaError_t e = cudaSuccess;
    std::vector<unsigned> first;
    fprintf(jf, ",\n \"launches\": [");
    bool same_all = true;
    for (int it = 0; it < repeat && e == cudaSuccess; ++it) {
        CK(cudaMemcpy(dout, out.data(), out.size() * 4, cudaMemcpyHostToDevice));
        ttu_one<<<1, LANES>>>(din, dout);
        e = cudaDeviceSynchronize();
        if (e != cudaSuccess) break;
        std::vector<unsigned> o(out.size());
        CK(cudaMemcpy(o.data(), dout, o.size() * 4, cudaMemcpyDeviceToHost));
        if (it == 0) first = o;
        bool same = true;
        for (unsigned i = 0; i < LANES; ++i)
            for (int r : {2, 3, 30, 31, 32, 33, 34, 35, 44, 45, 56, 57, 58, 59, 60, 61, 62, 63, 68, 69, 70, 71})
                same &= o[i * N + r] == first[i * N + r];
        same_all &= same;
        fprintf(jf, "%s{\"same_as_first\": %s, \"clk\": [", it ? ", " : "", same ? "true" : "false");
        for (unsigned i = 0; i < LANES; ++i) {
            unsigned long long a = ((unsigned long long)o[i * N + 85] << 32) | o[i * N + 84];
            unsigned long long b = ((unsigned long long)o[i * N + 87] << 32) | o[i * N + 86];
            fprintf(jf, "%s%lld", i ? "," : "", (long long)(b - a));
        }
        fprintf(jf, "]}");
        if (it == repeat - 1) out = o;
    }
    fprintf(jf, "]");
    printf("ttu_one: %s (%d launch(es), results same every launch: %d)\n", cudaGetErrorString(e), repeat, same_all);
    fprintf(jf, ",\n \"ttu_status\": \"%s\",\n \"in\": [", cudaGetErrorString(e));
    for (unsigned i = 0; i < LANES; ++i) {
        fprintf(jf, "%s[", i ? ", " : "");
        for (unsigned k = 0; k < N; ++k) fprintf(jf, "%s%u", k ? "," : "", in[i * N + k]);
        fprintf(jf, "]");
    }
    fprintf(jf, "],\n \"out\": [");
    for (unsigned i = 0; i < LANES; ++i) {
        fprintf(jf, "%s[", i ? ", " : "");
        for (unsigned k = 0; k < N; ++k) fprintf(jf, "%s%u", k ? "," : "", out[i * N + k]);
        fprintf(jf, "]");
    }
    fprintf(jf, "]\n}\n"); fclose(jf);
    return e == cudaSuccess ? 0 : 1;   // pipeline and GAS stay alive until here
}
