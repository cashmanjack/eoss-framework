// Host of the rehosted RTNN style radius search (R3 to R5).
//   rtnn_host PTX WORDS OUT.json --mode real|inject [--pts N] [--r R] [--seed S] [--warps W] [--blocks B] [--repeat K] [--optix-repeat K] [--optix-only] [--stats-only]
// One query per lane (B blocks x W warps x 32 lanes). The data points (N, uniform in the unit cube), the AABB custom primitive GAS (box = point +- r) and an OptiX
// reference pipeline (rtnn_dev.cu) are built in this process; then the patched ttu_one (build_rtnn.py) runs the same queries in a loop of TTU sessions with the
// inline distance test. Gate: per query the neighbour count and set equal in ours, OptiX and a CPU grid search with the same float expression.
//   inject (stage 0 builds, --rounds 1): each lane gets a fake candidate answer (tag, hit flag, pointer to a record) in its input image; the TTU group is NOPs.
// The patched kernel runs only through ttu/rehost/rtnn_run.py (check_seq marker, idle GPU, timeout).
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
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include "rtnn_params.h"

#define N 96
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(2); } } while (0)
#define OX(x) do { OptixResult r_ = (x); if (r_ != OPTIX_SUCCESS) { fprintf(stderr, "%s:%d %s: %d\n", __FILE__, __LINE__, #x, (int)r_); exit(2); } } while (0)

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
    if (t == 0xffffffffu) {   // never true: only adds instruction slots for the builder
#pragma unroll
        for (int k = 0; k < 160; ++k) asm volatile("nanosleep.u32 0;");
    }
}

static std::string read_file(const char *path) { std::ifstream f(path); std::stringstream s; s << f.rdbuf(); return s.str(); }
static unsigned fbits(float x) { unsigned u; memcpy(&u, &x, 4); return u; }
static float bitsf(unsigned u) { float f; memcpy(&f, &u, 4); return f; }
static double now_ms() { return 0; }

template <class T> static T *dalloc_lo32(size_t n)   // device buffer that does not straddle a 4 GB boundary (the kernel does 32 bit pointer arithmetic on it)
{
    for (int tries = 0; tries < 8; ++tries) {
        T *p; CK(cudaMalloc((void **)&p, n * sizeof(T)));
        if ((((unsigned long long)p) & 0xffffffffull) + n * sizeof(T) + 4096 < 0x100000000ull) return p;
    }
    fprintf(stderr, "no allocation inside one 4 GB window\n"); exit(2);
}

int main(int argc, char **argv)
{
    if (argc < 4) { fprintf(stderr, "usage: rtnn_host PTX WORDS OUT.json [options]\n"); return 2; }
    std::string mode = "real"; int npts = 4096, warps = 1, blocks = 1, repeat = 1, orep = 0; unsigned seed = 7; float r = 0.08f; bool optix_only = false, stats_only = false, no_ttu = false, trace = false, serial = false; const char *gas_file = nullptr; int inject_rounds = 1;
    for (int a = 4; a < argc; ++a) {
        if (!strcmp(argv[a], "--mode") && a + 1 < argc) mode = argv[++a];
        else if (!strcmp(argv[a], "--pts") && a + 1 < argc) npts = atoi(argv[++a]);
        else if (!strcmp(argv[a], "--r") && a + 1 < argc) r = (float)atof(argv[++a]);
        else if (!strcmp(argv[a], "--seed") && a + 1 < argc) seed = (unsigned)atoi(argv[++a]);
        else if (!strcmp(argv[a], "--warps") && a + 1 < argc) warps = atoi(argv[++a]);
        else if (!strcmp(argv[a], "--blocks") && a + 1 < argc) blocks = atoi(argv[++a]);
        else if (!strcmp(argv[a], "--repeat") && a + 1 < argc) repeat = atoi(argv[++a]);
        else if (!strcmp(argv[a], "--optix-repeat") && a + 1 < argc) orep = atoi(argv[++a]);
        else if (!strcmp(argv[a], "--optix-only")) optix_only = true;
        else if (!strcmp(argv[a], "--stats-only")) stats_only = true;
        else if (!strcmp(argv[a], "--dump-gas") && a + 1 < argc) gas_file = argv[++a];   // the whole GAS of this scene (private raw file)
        else if (!strcmp(argv[a], "--inject-rounds") && a + 1 < argc) inject_rounds = atoi(argv[++a]);   // inject mode: the stage 0 build runs this many rounds; round 1 sees answer 1, rounds 2.. see answer 2 again
        else if (!strcmp(argv[a], "--serial")) serial = true;   // for a build_rtnn.py --lane0 kernel: one launch of one warp per query, lane 0 of each warp is the only lane that runs
        else if (!strcmp(argv[a], "--trace")) trace = true;   // for a build_rtnn.py --trace kernel: per session answer words in a per lane trace area (64 sessions of 128 bytes) behind the out records
        else { fprintf(stderr, "unknown option %s\n", argv[a]); return 2; }
    }
    if (warps < 1 || warps > 16 || blocks < 1 || blocks > 4096 || repeat < 1 || repeat > 100 || npts < 1 || (mode != "real" && mode != "inject")) { fprintf(stderr, "bad option value\n"); return 2; }
    const unsigned LANES = 32u * warps, TOTAL = LANES * blocks;
    unsigned words[8]; { std::istringstream ws(read_file(argv[2])); for (auto &w : words) ws >> std::hex >> w; }
    CK(cudaFree(0));
    // ---- data ----
    std::mt19937 rng(seed); std::uniform_real_distribution<float> U(0.f, 1.f);
    std::vector<float4> P(npts), Q(TOTAL);
    for (auto &p : P) p = make_float4(U(rng), U(rng), U(rng), 0.f);
    for (auto &q : Q) q = make_float4(U(rng), U(rng), U(rng), 0.f);
    const float r2 = r * r;
    float4 *dP = dalloc_lo32<float4>(npts), *dQ; CK(cudaMalloc((void **)&dQ, (size_t)TOTAL * sizeof(float4)));
    CK(cudaMemcpy(dP, P.data(), npts * sizeof(float4), cudaMemcpyHostToDevice)); CK(cudaMemcpy(dQ, Q.data(), (size_t)TOTAL * sizeof(float4), cudaMemcpyHostToDevice));
    std::vector<float> bb; bb.reserve((size_t)npts * 6);
    for (auto &p : P) { bb.push_back(p.x - r); bb.push_back(p.y - r); bb.push_back(p.z - r); bb.push_back(p.x + r); bb.push_back(p.y + r); bb.push_back(p.z + r); }
    // ---- OptiX GAS and pipeline ----
    OptixDeviceContextOptions co{}; OptixDeviceContext ctx; OX(optixInit()); OX(optixDeviceContextCreate(0, &co, &ctx));
    CUdeviceptr dbb; CK(cudaMalloc((void **)&dbb, bb.size() * 4)); CK(cudaMemcpy((void *)dbb, bb.data(), bb.size() * 4, cudaMemcpyHostToDevice));
    unsigned gflags = OPTIX_GEOMETRY_FLAG_NONE; OptixBuildInput bi{}; bi.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    bi.customPrimitiveArray.aabbBuffers = &dbb; bi.customPrimitiveArray.numPrimitives = npts; bi.customPrimitiveArray.strideInBytes = 24; bi.customPrimitiveArray.flags = &gflags; bi.customPrimitiveArray.numSbtRecords = 1;
    OptixAccelBuildOptions ao{}; ao.buildFlags = OPTIX_BUILD_FLAG_NONE; ao.operation = OPTIX_BUILD_OPERATION_BUILD;
    OptixAccelBufferSizes bs{}; OX(optixAccelComputeMemoryUsage(ctx, &ao, &bi, 1, &bs));
    CUdeviceptr tmp, gas; CK(cudaMalloc((void **)&tmp, bs.tempSizeInBytes));
    { std::vector<CUdeviceptr> held; for (int tries = 0; tries < 16; ++tries) { CK(cudaMalloc((void **)&gas, bs.outputSizeInBytes)); if (((unsigned long long)gas & 0xffffffffull) + bs.outputSizeInBytes + 4096 < 0x100000000ull) break; held.push_back(gas); }   // the kernel advances the record pointer in 32 bits: keep the GAS inside one 4 GB window
      for (auto h_ : held) cudaFree((void *)h_); }
    OptixTraversableHandle h; OX(optixAccelBuild(ctx, 0, &ao, &bi, 1, tmp, bs.tempSizeInBytes, gas, bs.outputSizeInBytes, &h, nullptr, 0)); CK(cudaDeviceSynchronize()); CK(cudaFree((void *)tmp));
    OptixModuleCompileOptions mo{}; OptixPipelineCompileOptions po{};
    po.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS; po.numPayloadValues = 1; po.numAttributeValues = 2; po.pipelineLaunchParamsVariableName = "params";
    po.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;
    std::string ptx = read_file(argv[1]); char log[4096]; size_t ls = sizeof log; OptixModule m;
    OX(optixModuleCreate(ctx, &mo, &po, ptx.c_str(), ptx.size(), log, &ls, &m));
    OptixProgramGroupOptions go{}; OptixProgramGroupDesc d[3] = {};
    d[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN; d[0].raygen.module = m; d[0].raygen.entryFunctionName = "__raygen__rg";
    d[1].kind = OPTIX_PROGRAM_GROUP_KIND_MISS; d[1].miss.module = m; d[1].miss.entryFunctionName = "__miss__ms";
    d[2].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP; d[2].hitgroup.moduleIS = m; d[2].hitgroup.entryFunctionNameIS = "__intersection__is";
    OptixProgramGroup g[3]; ls = sizeof log; OX(optixProgramGroupCreate(ctx, d, 3, &go, log, &ls, g));
    OptixPipelineLinkOptions lo{}; lo.maxTraceDepth = 1; OptixPipeline pipe; ls = sizeof log; OX(optixPipelineCreate(ctx, &po, &lo, g, 3, log, &ls, &pipe));
    struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) Rec { char h[OPTIX_SBT_RECORD_HEADER_SIZE]; };
    Rec rec[3]; for (int k = 0; k < 3; ++k) OX(optixSbtRecordPackHeader(g[k], &rec[k]));
    CUdeviceptr drec; CK(cudaMalloc((void **)&drec, sizeof rec)); CK(cudaMemcpy((void *)drec, rec, sizeof rec, cudaMemcpyHostToDevice));
    OptixShaderBindingTable sbt{}; sbt.raygenRecord = drec; sbt.missRecordBase = drec + sizeof(Rec); sbt.missRecordStrideInBytes = sizeof(Rec); sbt.missRecordCount = 1;
    sbt.hitgroupRecordBase = drec + 2 * sizeof(Rec); sbt.hitgroupRecordStrideInBytes = sizeof(Rec); sbt.hitgroupRecordCount = 1;
    RParams prm{}; prm.q = dQ; prm.pts = dP; prm.r2 = r2; prm.n = TOTAL; prm.handle = h;
    CK(cudaMalloc((void **)&prm.candlog, (size_t)TOTAL * 256 * 4)); CK(cudaMemset(prm.candlog, 0, (size_t)TOTAL * 256 * 4)); CK(cudaMalloc((void **)&prm.nbr, (size_t)TOTAL * 65 * 4)); CK(cudaMalloc((void **)&prm.ncand, (size_t)TOTAL * 4)); CK(cudaMalloc((void **)&prm.inst, 18 * 8));
    CK(cudaMemset(prm.nbr, 0, (size_t)TOTAL * 65 * 4)); CK(cudaMemset(prm.ncand, 0, (size_t)TOTAL * 4)); CK(cudaMemset(prm.inst, 0, 18 * 8));
    CUdeviceptr dprm; CK(cudaMalloc((void **)&dprm, sizeof prm)); CK(cudaMemcpy((void *)dprm, &prm, sizeof prm, cudaMemcpyHostToDevice));
    OX(optixLaunch(pipe, 0, dprm, sizeof prm, &sbt, TOTAL, 1, 1)); CK(cudaDeviceSynchronize());
    std::vector<unsigned> onbr((size_t)TOTAL * 65), oncand(TOTAL); unsigned long long inst[18];
    std::vector<unsigned> ocl((size_t)TOTAL * 256); CK(cudaMemcpy(ocl.data(), prm.candlog, ocl.size() * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(onbr.data(), prm.nbr, onbr.size() * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(oncand.data(), prm.ncand, TOTAL * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(inst, prm.inst, sizeof inst, cudaMemcpyDeviceToHost));
    std::vector<float> optix_ms;
    if (orep > 0) {
        cudaEvent_t a0, a1; CK(cudaEventCreate(&a0)); CK(cudaEventCreate(&a1));
        for (int k = 0; k < orep; ++k) {
            CK(cudaMemset(prm.nbr, 0, (size_t)TOTAL * 65 * 4)); CK(cudaMemset(prm.ncand, 0, (size_t)TOTAL * 4)); CK(cudaDeviceSynchronize());
            CK(cudaEventRecord(a0)); OX(optixLaunch(pipe, 0, dprm, sizeof prm, &sbt, TOTAL, 1, 1)); CK(cudaEventRecord(a1)); CK(cudaEventSynchronize(a1));
            float ms; CK(cudaEventElapsedTime(&ms, a0, a1)); optix_ms.push_back(ms);
        }
    }
    // ---- checks before anything is handed to ttu_one ----
    const unsigned long long A = inst[0]; const unsigned ident[16] = {0, 1, 0, 0, 0x3f800000, 0, 0, 0, 0, 0x3f800000, 0, 0, 0, 0, 0x3f800000, 0};
    bool inst_ok = true; for (int k = 0; k < 16; ++k) inst_ok &= (unsigned)inst[1 + k] == ident[k];
    const unsigned c3_34 = (unsigned)inst[17]; const bool c3_ok = c3_34 == 0x00011000u;
    unsigned long long off = 0; CK(cudaMemcpy(&off, (void *)(gas + 0x20), 8, cudaMemcpyDeviceToHost)); const unsigned long long root = (unsigned long long)gas + off;
    const bool handle_ok = (h & ~0x3fULL) == (unsigned long long)gas && (h & 0x3f) == 0xc && (gas & 0x7f) == 0, align_ok = (A & 0x3f) == 0 && (root & 0x7f) == 0;
    unsigned rec_off = 0; CK(cudaMemcpy(&rec_off, (void *)(gas + 0xa0), 4, cudaMemcpyDeviceToHost));
    if (gas_file) { std::vector<char> gb(bs.outputSizeInBytes); CK(cudaMemcpy(gb.data(), (void *)gas, gb.size(), cudaMemcpyDeviceToHost)); FILE *gf = fopen(gas_file, "wb"); fwrite(gb.data(), 1, gb.size(), gf); fclose(gf); }
    if (((unsigned long long)gas & 0xffffffffull) + bs.outputSizeInBytes >= 0x100000000ull) { fprintf(stderr, "the GAS straddles a 4 GB boundary (the record pointer is advanced in 32 bits)\n"); return 2; }
    printf("rtnn: %d points r %g, %u queries (%d blocks x %d warps), gas 0x%llx size %zu root 0x%llx inst 0x%llx inst_ok %d c3_34 0x%x handle_ok %d align_ok %d rec_off 0x%x\n", npts, r, TOTAL, blocks, warps,
           (unsigned long long)gas, (size_t)bs.outputSizeInBytes, root, A, inst_ok, c3_34, handle_ok, align_ok, rec_off);
    // ---- CPU reference (same float expression, uniform grid) ----
    const int G = std::max(1, std::min(256, (int)std::floor(1.0 / std::max(r, 1e-4f)))); std::vector<std::vector<int>> cell((size_t)G * G * G);
    auto cidx = [&](float v) { int c = (int)(v * G); return std::min(G - 1, std::max(0, c)); };
    for (int i = 0; i < npts; ++i) cell[((size_t)cidx(P[i].x) * G + cidx(P[i].y)) * G + cidx(P[i].z)].push_back(i);
    auto cpu_nbrs = [&](unsigned qi, size_t *ambig) {
        std::vector<unsigned> res; const float4 q = Q[qi]; const int cx = cidx(q.x), cy = cidx(q.y), cz = cidx(q.z); const int span = 1 + (int)std::ceil(r * G);
        for (int x = std::max(0, cx - span); x <= std::min(G - 1, cx + span); ++x) for (int y = std::max(0, cy - span); y <= std::min(G - 1, cy + span); ++y) for (int z = std::max(0, cz - span); z <= std::min(G - 1, cz + span); ++z)
            for (int i : cell[((size_t)x * G + y) * G + z]) {
                const float4 p = P[i]; const float dx = p.x - q.x, dy = p.y - q.y, dz = p.z - q.z; const float d2 = fmaf(dz, dz, fmaf(dy, dy, dx * dx));
                if (d2 <= r2) res.push_back(i);
                const double D = (double)dx * dx + (double)dy * dy + (double)dz * dz; if (ambig && std::fabs(D - (double)r2) < 1e-6 * r2) ++*ambig;
            }
        std::sort(res.begin(), res.end()); return res;
    };
    if (!inst_ok || !c3_ok || !handle_ok || !align_ok) { fprintf(stderr, "check failed, ttu_one not launched\n"); return 3; }
    FILE *jf = fopen(argv[3], "w");
    size_t onb_total = 0, ocand_total = 0; for (unsigned i = 0; i < TOTAL; ++i) { onb_total += onbr[(size_t)i * 65]; ocand_total += oncand[i]; }
    fprintf(jf, "{\n \"mode\": \"%s\", \"pts\": %d, \"r\": %.9g, \"seed\": %u, \"queries\": %u, \"blocks\": %d, \"warps\": %d, \"gas\": \"0x%llx\", \"gas_size\": %zu, \"rec_off\": %u, \"c3_34\": \"0x%08x\",\n",
            mode.c_str(), npts, r, seed, TOTAL, blocks, warps, (unsigned long long)gas, (size_t)bs.outputSizeInBytes, rec_off, c3_34);
    fprintf(jf, " \"optix_ms\": [");
    for (size_t k = 0; k < optix_ms.size(); ++k) fprintf(jf, "%s%.6f", k ? "," : "", optix_ms[k]);
    fprintf(jf, "], \"optix_neighbours\": %zu, \"optix_candidates\": %zu", onb_total, ocand_total);
    if (optix_only || (no_ttu)) { fprintf(jf, "\n}\n"); fclose(jf); printf("optix only: neighbours %zu candidates %zu\n", onb_total, ocand_total); return 0; }
    // ---- register image ----
    const size_t IMG = (size_t)TOTAL * N;
    const size_t TRW = trace ? (size_t)TOTAL * 2048 : 0;   // trace words (8 KB per lane)
    unsigned *dout = dalloc_lo32<unsigned>(IMG + TRW), *din; CK(cudaMalloc((void **)&din, IMG * 4));
    unsigned long long *drecs = nullptr; std::vector<unsigned> fake_prim(TOTAL, 0);
    unsigned *dr = nullptr; if (mode == "inject") { dr = dalloc_lo32<unsigned>((size_t)TOTAL * 256); }   // 2 runs of up to 64 records (2 words each) per lane
    std::vector<unsigned> in(IMG, 0), outv(IMG + TRW, 0x77777777u);
    const unsigned long long st0 = root; const unsigned start_flags = 0x2484000u;
    static const unsigned CNTS[7] = {1, 2, 3, 8, 17, 31, 63}; std::vector<std::vector<unsigned>> fake_run(TOTAL), fake_run2(TOTAL); std::vector<unsigned> cnt1(TOTAL, 0), cnt2(TOTAL, 0);
    // inject: lane i gets answer 1 for CNTS[i % 7] records in round 1 and answer 2 (CNTS[(i / 7 + 3) % 7] records, the next consecutive run) in every later round
    if (mode == "inject") { std::vector<unsigned> rv((size_t)TOTAL * 256, 0); std::mt19937 r2g(seed + 99);
        for (unsigned i = 0; i < TOTAL; ++i) { cnt1[i] = CNTS[i % 7]; cnt2[i] = CNTS[(i / 7 + 3) % 7];
            for (unsigned j = 0; j < cnt1[i]; ++j) { const unsigned p = (unsigned)(r2g() % npts); fake_run[i].push_back(p); rv[(size_t)i * 256 + 2 * j] = p; }
            for (unsigned j = 0; j < cnt2[i]; ++j) { const unsigned p = (unsigned)(r2g() % npts); fake_run2[i].push_back(p); rv[(size_t)i * 256 + 128 + 2 * j] = p; } }
        CK(cudaMemcpy(dr, rv.data(), rv.size() * 4, cudaMemcpyHostToDevice)); }
    for (unsigned i = 0; i < TOTAL; ++i) {
        unsigned *R = &in[(size_t)i * N]; const float4 q = Q[i];
        R[12] = fbits(q.z); R[13] = 0; R[82] = fbits(q.x); R[83] = fbits(q.y);   // 0x40: origin xy in R82:R83, z and tmin in R12:R13
        R[16] = fbits(1.f); R[17] = fbits(1e-16f); R[14] = 0; R[15] = 0;           // 0x50: direction (0, 0, 1), tmax
        R[52] = R[12]; R[53] = R[13]; R[56] = R[82]; R[57] = R[83]; R[30] = R[16]; R[31] = R[17]; R[46] = R[14]; R[47] = R[15];
        R[54] = 0; R[55] = 0; R[34] = (unsigned)(A >> 6); R[35] = ((((unsigned)(A >> 32) >> 6) | 0x402800u) & 0x1ffffffu) | 0x2000000u;
        R[70] = (unsigned)(st0 >> 7); R[71] = ((unsigned)(st0 >> 32) >> 7) | start_flags;
        R[68] = words[0]; R[69] = words[1]; R[62] = words[2]; R[63] = words[3]; R[60] = words[4]; R[61] = words[5]; R[58] = words[6]; R[59] = words[7];
        // the trace address is absolute: the template's stores are [R.64] with a memory descriptor, not base + offset
        if (trace) { const unsigned long long tro = (unsigned long long)dout + (unsigned long long)IMG * 4 + (unsigned long long)i * 8192; R[18] = (unsigned)tro; R[19] = (unsigned)(tro >> 32); }
        R[90] = 0; R[6] = (unsigned)(unsigned long long)dP; R[7] = (unsigned)((unsigned long long)dP >> 32); R[8] = fbits(r2);
        if (mode == "inject") { const unsigned long long ra = (unsigned long long)(dr + 256 * i) >> 3, rb = (unsigned long long)(dr + 256 * i + 128) >> 3; R[33] = 0xa0000000u | cnt1[i]; R[3] = 1; R[44] = (unsigned)ra; R[45] = (unsigned)(ra >> 32);
            R[59] = 0xa0000000u | cnt2[i]; R[60] = (unsigned)rb; R[61] = (unsigned)(rb >> 32); }   // answer 2 reaches R33, R44:R45 through the swap (R33 <- R59, R44:R45 <- R60:R61), R3 <- R71
    }
    CK(cudaMemcpy(din, in.data(), IMG * 4, cudaMemcpyHostToDevice));
    // the kernel adds to the out pointer in 32 bits (neighbour cursor): refuse a window that crosses 4 GB
    if ((((unsigned long long)dout) & 0xffffffffull) + (IMG + TRW) * 4 >= 0x100000000ull) { fprintf(stderr, "out buffer straddles a 4 GB boundary\n"); return 2; }
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1)); std::vector<float> ttu_ms; cudaError_t e = cudaSuccess;
    for (int it = 0; it < repeat && e == cudaSuccess; ++it) {
        std::vector<unsigned> fill(IMG + TRW, 0x77777777u); CK(cudaMemcpy(dout, fill.data(), (IMG + TRW) * 4, cudaMemcpyHostToDevice));
        CK(cudaDeviceSynchronize()); CK(cudaEventRecord(e0));
        if (serial) { for (unsigned q = 0; q < TOTAL && e == cudaSuccess; ++q) { ttu_one<<<1, 32>>>(din + (size_t)q * N, dout + (size_t)q * N); e = cudaGetLastError(); if (e == cudaSuccess) e = cudaDeviceSynchronize(); } }
        else ttu_one<<<blocks, LANES>>>(din, dout);
        CK(cudaEventRecord(e1)); e = cudaGetLastError(); if (e == cudaSuccess) e = cudaEventSynchronize(e1);
        if (e == cudaSuccess) { float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); ttu_ms.push_back(ms); }
    }
    if (e == cudaSuccess) CK(cudaMemcpy(outv.data(), dout, (IMG + TRW) * 4, cudaMemcpyDeviceToHost));
    fprintf(jf, ",\n \"ttu_status\": \"%s\", \"ttu_ms\": [", cudaGetErrorString(e));
    for (size_t k = 0; k < ttu_ms.size(); ++k) fprintf(jf, "%s%.6f", k ? "," : "", ttu_ms[k]);
    fprintf(jf, "]");
    // ---- compare ----
    size_t bad_ours_optix = 0, bad_ours_cpu = 0, bad_optix_cpu = 0, ambiguous = 0, unfinished = 0, capped = 0, checked = 0, same_order = 0, ses_eq_cand = 0, ses_cand_p1 = 0, ses_other = 0; long long first_bad = -1;
    std::vector<unsigned> ses(TOTAL), cnt(TOTAL), sess_cyc(TOTAL), tot_cyc(TOTAL), left(TOTAL); std::vector<std::vector<unsigned>> expv(TOTAL);
    for (unsigned i = 0; i < TOTAL && e == cudaSuccess; ++i) {
        const unsigned *O = &outv[(size_t)i * N];
        cnt[i] = O[6]; ses[i] = O[7]; sess_cyc[i] = O[9] - O[8]; tot_cyc[i] = O[11] - O[10]; left[i] = O[13];
        std::vector<unsigned> ours; for (unsigned k = 0; k < std::min<unsigned>(cnt[i], 64); ++k) ours.push_back(O[16 + k]);
        std::vector<unsigned> ox; for (unsigned k = 0; k < std::min<unsigned>(onbr[(size_t)i * 65], 64); ++k) ox.push_back(onbr[(size_t)i * 65 + 1 + k]);
        std::vector<unsigned> cpu;
        if (mode == "inject") {   // expected: the fake candidate's primitive, if within r (same expression)
            auto inside = [&](unsigned pp) { const float4 p = P[pp], q = Q[i]; const float dx = p.x - q.x, dy = p.y - q.y, dz = p.z - q.z; return fmaf(dz, dz, fmaf(dy, dy, dx * dx)) <= r2; };
            for (unsigned pp : fake_run[i]) if (inside(pp)) cpu.push_back(pp);
            for (int rr = 1; rr < inject_rounds; ++rr) for (unsigned pp : fake_run2[i]) if (inside(pp)) cpu.push_back(pp);
            ox = cpu;   // no OptiX reference in inject mode
            if (O[7] != (unsigned)inject_rounds) ++unfinished;
        } else { cpu = cpu_nbrs(i, &ambiguous); }
        expv[i] = cpu;
        const bool ocap = onbr[(size_t)i * 65] > 64; capped += ocap; ++checked;
        std::vector<unsigned> a = ours, b = ox, c = cpu; std::sort(a.begin(), a.end()); std::sort(b.begin(), b.end()); std::sort(c.begin(), c.end());
        const bool eq_ob = mode == "inject" ? true : (cnt[i] == onbr[(size_t)i * 65] && a == b), eq_oc = (cnt[i] == cpu.size() || (ocap && cnt[i] > 64)) && (ocap || a == c), eq_xc = mode == "inject" ? true : (onbr[(size_t)i * 65] == cpu.size() && (ocap || b == c));
        if (mode == "inject") { if (!(cnt[i] == cpu.size() && ours == std::vector<unsigned>(cpu.begin(), cpu.begin() + std::min<size_t>(cpu.size(), 64)))) { ++bad_ours_cpu; if (first_bad < 0) first_bad = i; if (bad_ours_cpu <= 12) printf("bad lane %u (warp %u, lane %u, count1 %u): ours cnt %u [%u %u %u ...] expect %zu\n", i, i / 32, i % 32, cnt1[i], cnt[i], a.size() > 0 ? a[0] : 0u, a.size() > 1 ? a[1] : 0u, a.size() > 2 ? a[2] : 0u, cpu.size()); } continue; }
        bad_ours_optix += !eq_ob; bad_ours_cpu += !eq_oc; bad_optix_cpu += !eq_xc; if ((!eq_ob || !eq_oc) && first_bad < 0) first_bad = i;
        same_order += (ours == ox);
        if (ses[i] == oncand[i]) ++ses_eq_cand; else if (ses[i] == oncand[i] + 1) ++ses_cand_p1; else ++ses_other;
        unfinished += left[i] == 0 && (O[85] | O[86] | O[87] | O[88] | O[89] | O[90] | O[91] | O[92]) != 0;   // last answer's pending words (slots 80..92)
    }
    auto pct = [&](std::vector<unsigned> v, double f) { if (v.empty()) return 0u; std::sort(v.begin(), v.end()); return v[(size_t)(f * (v.size() - 1))]; };
    double sum_ses = 0, sum_eff_num = 0, sum_eff_den = 0, sum_cyc = 0, sum_sess = 0;
    for (unsigned i = 0; i < TOTAL && e == cudaSuccess; ++i) { sum_ses += ses[i]; sum_cyc += tot_cyc[i]; sum_sess += sess_cyc[i]; }
    for (unsigned w = 0; w < TOTAL / 32 && e == cudaSuccess; ++w) { unsigned mx = 0, sm = 0; for (int l = 0; l < 32; ++l) { mx = std::max(mx, ses[w * 32 + l]); sm += ses[w * 32 + l]; } sum_eff_num += sm; sum_eff_den += 32.0 * mx; }
    fprintf(jf, ",\n \"checked\": %zu, \"bad_ours_vs_optix\": %zu, \"bad_ours_vs_cpu\": %zu, \"bad_optix_vs_cpu\": %zu, \"ambiguous_boundary_pairs\": %zu, \"capped_queries\": %zu, \"unfinished\": %zu, \"first_bad\": %lld,\n",
            checked, bad_ours_optix, bad_ours_cpu, bad_optix_cpu, ambiguous, capped, unfinished, first_bad);
    fprintf(jf, " \"order_equal\": %zu, \"sessions_eq_candidates\": %zu, \"sessions_eq_candidates_plus1\": %zu, \"sessions_other\": %zu,\n", same_order, ses_eq_cand, ses_cand_p1, ses_other);
    fprintf(jf, " \"mean_sessions\": %.6f, \"sessions_p50\": %u, \"sessions_p90\": %u, \"sessions_max\": %u, \"lane_efficiency\": %.6f, \"mean_neighbours\": %.6f,\n",
            sum_ses / TOTAL, pct(ses, 0.5), pct(ses, 0.9), pct(ses, 1.0), sum_eff_den > 0 ? sum_eff_num / sum_eff_den : 0.0, (double)onb_total / TOTAL);
    fprintf(jf, " \"loop_cycles_mean\": %.1f, \"loop_cycles_p50\": %u, \"loop_cycles_p90\": %u, \"session_cycles_mean\": %.1f, \"session_cycles_p50\": %u, \"session_cycles_p90\": %u",
            sum_cyc / TOTAL, pct(tot_cyc, 0.5), pct(tot_cyc, 0.9), sum_sess / TOTAL, pct(sess_cyc, 0.5), pct(sess_cyc, 0.9));
    if (!stats_only) {   // per lane detail for the first 64 lanes
        fprintf(jf, ",\n \"lanes\": [");
        for (unsigned i = 0; i < std::min<unsigned>(TOTAL, 64) && e == cudaSuccess; ++i) {
            const unsigned *O = &outv[(size_t)i * N]; fprintf(jf, "%s{\"cnt\": %u, \"ses\": %u, \"optix_cnt\": %u, \"optix_cand\": %u, \"sess_cyc\": %u, \"loop_cyc\": %u, \"left\": %u, \"ours\": [", i ? "," : "", cnt[i], ses[i], onbr[(size_t)i * 65], oncand[i], sess_cyc[i], tot_cyc[i], left[i]);
            for (unsigned k = 0; k < std::min<unsigned>(cnt[i], 64); ++k) fprintf(jf, "%s%u", k ? "," : "", O[16 + k]);
            fprintf(jf, "], \"expect\": ["); for (size_t k = 0; k < expv[i].size() && k < 64; ++k) fprintf(jf, "%s%u", k ? "," : "", expv[i][k]);
            fprintf(jf, "], \"optix\": ["); for (unsigned k = 0; k < std::min<unsigned>(onbr[(size_t)i * 65], 64); ++k) fprintf(jf, "%s%u", k ? "," : "", onbr[(size_t)i * 65 + 1 + k]);
            fprintf(jf, "], \"last\": [%u,\"%08x\",\"%08x\",\"%08x\",\"%08x\"], \"dbg\": [", O[80], O[81], O[82], O[83], O[84]);
            for (int k = 20; k < 32; ++k) fprintf(jf, "%s\"%08x\"", k > 20 ? "," : "", O[k]);
            fprintf(jf, "]}");
        }
        fprintf(jf, "]");
    }
    { std::vector<unsigned> rt((size_t)npts * 2); CK(cudaMemcpy(rt.data(), (void *)(gas + rec_off), rt.size() * 4, cudaMemcpyDeviceToHost));   // the GAS record table (word 0 = primitive index) and OptiX's full candidate sequence per query
      fprintf(jf, ",\n \"rec_prim\": ["); for (int k = 0; k < npts; ++k) fprintf(jf, "%s%u", k ? "," : "", rt[(size_t)k * 2]);
      fprintf(jf, "],\n \"optix_cands\": ["); for (unsigned i = 0; i < std::min<unsigned>(TOTAL, 64); ++i) { fprintf(jf, "%s[", i ? "," : ""); for (unsigned k = 0; k < std::min<unsigned>(oncand[i], 256); ++k) fprintf(jf, "%s%u", k ? "," : "", ocl[(size_t)i * 256 + k]); fprintf(jf, "]"); }
      fprintf(jf, "]"); }
    if (trace && e == cudaSuccess) {   // the answer registers of every session of the first 64 lanes: [R2,R3,R30,R31,R32,R33,R34,R35,R44,R45,R56..R63,R68,R69,R70,R71]
        fprintf(jf, ",\n \"trace\": [");
        for (unsigned i = 0; i < std::min<unsigned>(TOTAL, 64); ++i) {
            fprintf(jf, "%s[", i ? "," : "");
            for (unsigned k = 0; k < std::min<unsigned>(ses[i], 64); ++k) { fprintf(jf, "%s[", k ? "," : ""); for (int w = 0; w < 22; ++w) fprintf(jf, "%s\"%08x\"", w ? "," : "", outv[IMG + (size_t)i * 2048 + k * 32 + w]); fprintf(jf, "]"); }
            fprintf(jf, "]");
        }
        fprintf(jf, "]");
    }
    fprintf(jf, "\n}\n"); fclose(jf);
    printf("ttu_one: %s | checked %zu | ours vs optix bad %zu | ours vs cpu bad %zu | optix vs cpu bad %zu | ambiguous %zu | unfinished %zu | sessions mean %.3f max %u | lane eff %.3f | neighbours %.2f/query\n",
           cudaGetErrorString(e), checked, bad_ours_optix, bad_ours_cpu, bad_optix_cpu, ambiguous, unfinished, sum_ses / TOTAL, pct(ses, 1.0), sum_eff_den > 0 ? sum_eff_num / sum_eff_den : 0.0, (double)onb_total / TOTAL);
    printf("cycles per query (mean): loop %.0f, in sessions %.0f | ttu ms %s | optix ms %s\n", sum_cyc / TOTAL, sum_sess / TOTAL, ttu_ms.empty() ? "-" : std::to_string(ttu_ms.back()).c_str(), optix_ms.empty() ? "-" : std::to_string(optix_ms.back()).c_str());
    return e == cudaSuccess && bad_ours_optix == 0 && bad_ours_cpu == 0 && bad_optix_cpu == 0 && unfinished == 0 ? 0 : 4;
}
