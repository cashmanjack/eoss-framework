// Setup_3 host: triangle GAS, an OptiX reference launch, then ttu_one<<<1, 32 * W>>>.
//
//   setup3 PTX WORDS RAY FILL OUT.json [options]
//     PTX    setup3_dev.ptx
//     WORDS  8 hex words: TTUST 0 then TTUST 0x80, low to high (made at build time)
//     RAY    hit | miss | t2 | back | moved | moved2 | hitB | tri:I   (every lane the same ray)
//            tri:I = straight down (+z) onto the centroid of triangle I
//            perlane = lane i onto the centroid of triangle (7919 i + 13) mod N (different paths)
//            jitter  = lane i onto a different point of triangle 0 (same path, different hit data)
//            random  = lane i: its own random origin in [-1,1]^3 and random direction
//            random1 = every lane traces lane 0's random ray (coherent control)
//            randomz = lane i: its own random origin, direction +z
//            file:F  = rays from F (8 float32 each: origin, tmin, direction, tmax); the count must be a
//                      multiple of 32 W; rays [32 W c, 32 W (c + 1)) run as chunk c, one chunk after another
//            pairs:F = lane i reads line i (wrapping) of F, "src dst": origin above the centroid of
//                      triangle src (z - 1), aimed at the centroid of triangle dst (src == dst: +z ray)
//     FILL   zero | random   (value of the words OptiX leaves uninitialized: 0x180 high, 0x190)
//   options
//     --scene S      tri1 (default) | tri2 (2 triangles, 1 build input) | tri2g (2 build inputs) | grid:N
//                    | soup:N[:E] (N random small triangles in [-1,1]^3, edge E/cbrt(N), E default 3)
//                    | par:N:s|o[:DELTA] (full traversal: N triangles parallel to a +z ray at x = y = 0,
//                      DELTA (default 1e-3) off it, stacked along z (s) or all overlapping (o); every box, no hit)
//                    | layers:N:L[:DZ] (L copies of grid:N at z = DZ k, DZ default 0.5)
//                    | stack:L (L copies of triangle 0 at z = 0.01 k, k = 0..L-1, all on the hit ray)
//     --warps W      warps in the single block, 1..21 (default 1); build_patch.py --regs 96 sets the register count
//     --repeat N     timed launches in this process, 1..100 (default 1)
//     --blocks G     blocks of W warps (default 1); with an --smfilter build only blocks on SM A or B run
//     --sms A,B      the SMs allowed to run (VIRTUALSMID; words 18, 19), default 0,0
//     --flush MB     before every ttu_one launch, a plain kernel reads MB megabytes (L2 eviction)
//     --dump-gas F   write the GAS bytes to F
//     --thrash       allocate a 4 MB buffer; words 26, 27 = buffer + 128 x lane (for build_patch --thrash)
//     --resume K     then up to K follow up queries per lane, built as OptiX's loop builds them
//     --dump-tris F  write all triangles (9 floats each) to F, for the CPU reference
//     --start OFF[:FLAGS]  first query starts at the node at GAS + OFF (hex, 128 byte aligned) instead
//                    of the GAS root: 0x180 low = node >> 7, high word (node_hi >> 7) | FLAGS (hex,
//                    default 0x2484000 = the root word's flags). Resumed queries use the TTU's own words.
//     --start-file F one slot path per ray (same order as the rays; "root" allowed), FLAGS as --start
//     --start-path P[:FLAGS]  same, but the node is named by its slot path from the root (e.g. 3,5 = the
//                    internal child in slot 3 of the root, then its slot 5; "root" = the root). OptiX
//                    places nodes in a different memory order in each process; the slot path is stable.
//     --passes P     run the whole chunk sequence P times (1..8, default 1); pass p chunk c > 0 writes
//                    OUT.json.p<p>c<c> (pass 0 chunk 0 is OUT.json itself)
//     --flush MB also flushes before every resume round
//     --active L --idle mask|tmax0   only lanes 0..L-1 of every warp carry a query (1 <= L <= 32). mask: the
//                    other lanes get SM words 18, 19 = 0xffffffff, so an --smfilter build makes them exit
//                    before any TTU instruction (run with --sms of the block's SM, 0 for one block);
//                    tmax0: they run the group with tmax 0 (riding along like finished resume lanes);
//                    tneg: tmax -1; empty: tmin 1, tmax 0
//     --sched M      run the whole ray set with a scheduler instead of chunks (needs an --smfilter build,
//                    --sms 0,0): round 0 = every ray from its start node in file order; then rounds of
//                    resumed queries (tmax = best t, 0x180/0x190 from the returned pending entries) until no
//                    ray has pending entries (at most --resume K rounds). M = inplace: rays keep their lane,
//                    finished lanes are masked; compact: the rays still running are packed into full warps in
//                    their previous order; regroup: packed after sorting by the pending node pointer they resume
//                    at. Padding lanes are masked. --passes P repeats the whole schedule; the last is reported.
//     --smem BYTES   ttu_one launches reserve BYTES of dynamic shared memory (unused), which shrinks L1
//                    (Memory_1; measured with chase.cu: any reservation gives L1 about 16..24 KB, none about 90 KB)
//     --optix-repeat K  time K more OptiX reference launches with CUDA events (Throughput_7), after one untimed
//     --no-ttu       stop after the OptiX reference
//
// ttu_one as compiled here only loads 96 words into registers and stores them
// back. build_patch.py rewrites its body (stage 0: no TTU code; stage 1: one
// TTU group). Run the patched copy only through run.py.
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
#include <set>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include "setup3_params.h"

#define N 96
#define MAXW 32   // one block; the register count build_patch declares sets the real limit
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
    if (t == 0xffffffffu) {   // never true: only adds instruction slots for build_patch.py
#pragma unroll
        for (int k = 0; k < 160; ++k) asm volatile("nanosleep.u32 0;");
    }
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
        if (n < 1 || n > 1 << 23) { fprintf(stderr, "grid:N with 1 <= N <= 2^23\n"); exit(2); }
        const int k = (int)std::ceil(std::sqrt((double)n));
        const float c = 2.f / k;
        std::vector<Tri> v;
        for (int i = 0; i < n; ++i) {
            const float x = -1.f + (i % k) * c, y = -1.f + (i / k) * c;
            v.push_back({{x + 0.1f * c, y + 0.1f * c, 0.f}, {x + 0.9f * c, y + 0.1f * c, 0.f}, {x + 0.5f * c, y + 0.9f * c, 0.f}});
        }
        return {v};
    }
    if (s.rfind("par:", 0) == 0) {   // full traversal scene for a ray along +z at x = y = 0 (Traversal_2a)
        // triangle k lies in the plane y = x tan(th_k) + DELTA, x in [-0.5, 0.5]: its box holds the ray's line,
        // but the plane is parallel to the ray and DELTA away, so the ray never hits it.
        // s: stacked along z (k-th at z in [2k/N - 1, 2(k+1)/N - 1]); o: all overlapping at z in [-1, 1].
        int n = 0; char mode = 's'; float delta = 1e-3f;
        if (sscanf(s.c_str() + 4, "%d:%c:%f", &n, &mode, &delta) < 2 || n < 1 || n > (1 << 22) || (mode != 's' && mode != 'o') || !(delta > 0.f)) {
            fprintf(stderr, "par:N:s|o[:DELTA]\n"); exit(2); }
        std::vector<Tri> v; std::mt19937 g(99); std::uniform_real_distribution<float> U(0.5f, 2.f);
        for (int k = 0; k < n; ++k) {
            const float t = U(g) * (k % 2 ? -1.f : 1.f);
            const float z0 = mode == 's' ? 2.f * k / n - 1.f : -1.f, z1 = mode == 's' ? 2.f * (k + 1) / n - 1.f : 1.f;
            v.push_back({{-0.5f, -0.5f * t + delta, z0}, {0.5f, 0.5f * t + delta, z0}, {0.f, delta, z1}});
        }
        return {v};
    }
    if (s.rfind("layers:", 0) == 0) {   // L copies of grid:N at z = DZ k, k = 0..L-1 (triangle index = k N + i)
        int n = 0, L = 0; float dz = 0.5f;
        if (sscanf(s.c_str() + 7, "%d:%d:%f", &n, &L, &dz) < 2 || n < 1 || L < 1 || (long long)n * L > (1 << 23) || !(dz > 0.f)) {
            fprintf(stderr, "layers:N:L[:DZ]\n"); exit(2); }
        auto g = make_scene("grid:" + std::to_string(n))[0];
        std::vector<Tri> v;
        for (int k = 0; k < L; ++k) for (Tri t : g) { t.a.z = t.b.z = t.c.z = dz * k; v.push_back(t); }
        return {v};
    }
    if (s.rfind("stack:", 0) == 0) {
        const int L = atoi(s.c_str() + 6);
        if (L < 1 || L > 4096) { fprintf(stderr, "stack:L with 1 <= L <= 4096\n"); exit(2); }
        std::vector<Tri> v;
        for (int k = 0; k < L; ++k) { Tri t = t0; t.a.z = t.b.z = t.c.z = 0.01f * k; v.push_back(t); }
        return {v};
    }
    if (s.rfind("soup:", 0) == 0) {
        const int n = atoi(s.c_str() + 5);
        if (n < 1 || n > 1 << 23) { fprintf(stderr, "soup:N with 1 <= N <= 2^23\n"); exit(2); }
        const size_t colon = s.find(':', 5);
        const float E = colon == std::string::npos ? 3.f : (float)atof(s.c_str() + colon + 1);
        if (!(E > 0.f && E <= 100.f)) { fprintf(stderr, "soup:N:E with 0 < E <= 100\n"); exit(2); }
        const float e = E / std::cbrt((float)n);            // edge length: E sets the overlap
        std::mt19937 g(1234); std::uniform_real_distribution<float> U(-1.f, 1.f);
        std::vector<Tri> v; v.reserve(n);
        for (int i = 0; i < n; ++i) {
            const float3 c{U(g), U(g), U(g)};
            auto pt = [&]() { return float3{c.x + 0.5f * e * U(g), c.y + 0.5f * e * U(g), c.z + 0.5f * e * U(g)}; };
            v.push_back({pt(), pt(), pt()});
        }
        return {v};
    }
    fprintf(stderr, "unknown scene %s\n", s.c_str()); exit(2);
}

int main(int argc, char **argv)
{
    if (argc < 6) { fprintf(stderr, "usage: see header of setup3.cu\n"); return 2; }
    const bool random_fill = !strcmp(argv[4], "random");
    bool run_ttu = true, thrash = false; int resume = 0; const char *dump_tris = nullptr; long long start_off = -1; unsigned start_flags = 0x2484000u; std::string start_path; bool use_path = false; const char *start_file = nullptr; int optix_repeat = 0; int smem = 0; int passes = 1; std::string sched; int active = 32; std::string idle = "mask"; int repeat = 1, warps = 1, blocks = 1; unsigned sm_a = 0, sm_b = 0; double flush_mb = 0; std::string scene = "tri1"; const char *dump = nullptr;
    for (int a = 6; a < argc; ++a) {
        if (!strcmp(argv[a], "--no-ttu")) run_ttu = false;
        else if (!strcmp(argv[a], "--thrash")) thrash = true;
        else if (!strcmp(argv[a], "--repeat") && a + 1 < argc) repeat = atoi(argv[++a]);
        else if (!strcmp(argv[a], "--passes") && a + 1 < argc) passes = atoi(argv[++a]);
        else if (!strcmp(argv[a], "--smem") && a + 1 < argc) smem = atoi(argv[++a]);
        else if (!strcmp(argv[a], "--optix-repeat") && a + 1 < argc) optix_repeat = atoi(argv[++a]);
        else if (!strcmp(argv[a], "--active") && a + 1 < argc) active = atoi(argv[++a]);
        else if (!strcmp(argv[a], "--sched") && a + 1 < argc) sched = argv[++a];
        else if (!strcmp(argv[a], "--idle") && a + 1 < argc) idle = argv[++a];
        else if (!strcmp(argv[a], "--warps") && a + 1 < argc) warps = atoi(argv[++a]);
        else if (!strcmp(argv[a], "--blocks") && a + 1 < argc) blocks = atoi(argv[++a]);
        else if (!strcmp(argv[a], "--sms") && a + 1 < argc) { if (sscanf(argv[++a], "%u,%u", &sm_a, &sm_b) != 2) { fprintf(stderr, "--sms A,B\n"); return 2; } }
        else if (!strcmp(argv[a], "--flush") && a + 1 < argc) flush_mb = atof(argv[++a]);
        else if (!strcmp(argv[a], "--scene") && a + 1 < argc) scene = argv[++a];
        else if (!strcmp(argv[a], "--dump-gas") && a + 1 < argc) dump = argv[++a];
        else if (!strcmp(argv[a], "--resume") && a + 1 < argc) resume = atoi(argv[++a]);
        else if (!strcmp(argv[a], "--dump-tris") && a + 1 < argc) dump_tris = argv[++a];
        else if (!strcmp(argv[a], "--start-file") && a + 1 < argc) start_file = argv[++a];
        else if (!strcmp(argv[a], "--start-path") && a + 1 < argc) {
            std::string v = argv[++a]; const size_t c = v.find(':'); use_path = true;
            if (c != std::string::npos) { char *end; start_flags = (unsigned)strtoul(v.c_str() + c + 1, &end, 16); v = v.substr(0, c);
                if (*end || (start_flags & 0xffu)) { fprintf(stderr, "--start-path P[:FLAGS]\n"); return 2; } }
            start_path = v == "root" ? "" : v;
        }
        else if (!strcmp(argv[a], "--start") && a + 1 < argc) {
            const char *v = argv[++a]; char *end; start_off = strtoll(v, &end, 16);
            if (*end == ':') start_flags = (unsigned)strtoul(end + 1, &end, 16);
            if (*end || start_off < 0 || (start_off & 0x7f) || (start_flags & 0xffu)) { fprintf(stderr, "--start OFF[:FLAGS], OFF 128 byte aligned, FLAGS low byte 0\n"); return 2; }
        }
        else { fprintf(stderr, "unknown option %s\n", argv[a]); return 2; }
    }
    if ((!sched.empty() && sched != "inplace" && sched != "compact" && sched != "regroup") || active < 1 || active > 32 || (idle != "mask" && idle != "tmax0" && idle != "tneg" && idle != "empty") || passes < 1 || passes > 8 || repeat < 1 || repeat > 100 || warps < 1 || warps > MAXW || blocks < 1 || blocks > 1024 || flush_mb < 0 || flush_mb > 4096) {
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
    if (dump_tris) { FILE *f = fopen(dump_tris, "wb"); fwrite(all.data(), sizeof(Tri), all.size(), f); fclose(f); }
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
    const bool rnd = !strcmp(argv[3], "random"), rnd1 = !strcmp(argv[3], "random1"), rndz = !strcmp(argv[3], "randomz");
    std::vector<std::pair<int, int>> pairs;
    if (!strncmp(argv[3], "pairs:", 6)) {
        std::istringstream ps(read_file(argv[3] + 6)); int a, b;
        while (ps >> a >> b) {
            if (a < 0 || b < 0 || a >= (int)all.size() || b >= (int)all.size()) { fprintf(stderr, "pairs: index out of range\n"); return 2; }
            pairs.push_back({a, b});
        }
        if (pairs.empty()) { fprintf(stderr, "pairs: empty file\n"); return 2; }
    }
    if (perlane || jitter || rnd || rnd1 || rndz || !pairs.empty() || !strncmp(argv[3], "file:", 5)) {
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
    if (!strncmp(argv[3], "file:", 5)) {
        FILE *rf = fopen(argv[3] + 5, "rb"); if (!rf) { fprintf(stderr, "cannot read %s\n", argv[3] + 5); return 2; }
        std::vector<Ray8> fr; Ray8 y; while (fread(&y, sizeof y, 1, rf) == 1) fr.push_back(y); fclose(rf);
        if (fr.empty() || fr.size() % LANES || fr.size() > (1u << 20)) { fprintf(stderr, "file: ray count %zu is not a multiple of %u\n", fr.size(), LANES); return 2; }
        rays = fr;
    }
    const unsigned NR = (unsigned)rays.size(), nchunks = NR / LANES;
    for (unsigned i = 0; i < LANES && (rnd || rnd1 || rndz); ++i) {
        std::mt19937 g(777 + (rnd1 ? 0 : i)); std::uniform_real_distribution<float> U(-1.f, 1.f);
        float3 o{U(g), U(g), U(g)}, d;
        do { d = {U(g), U(g), U(g)}; } while (d.x * d.x + d.y * d.y + d.z * d.z < 0.01f || d.x * d.x + d.y * d.y + d.z * d.z > 1.f);
        if (rndz) d = {0.f, 0.f, 1.f};
        const float l = std::sqrt(d.x * d.x + d.y * d.y + d.z * d.z);
        rays[i] = {{o.x, o.y, o.z}, 0.f, {d.x / l, d.y / l, d.z / l}, 1e16f};
    }
    for (unsigned i = 0; i < LANES && (perlane || jitter); ++i) {
        const Tri &t = all[perlane ? (size_t)((7919ull * i + 13) % all.size()) : 0];
        float u = 1.f / 3, v = 1.f / 3;
        if (jitter) { u = ((i % 8) + 0.5f) / 16.f; v = (((i / 8) % 8) + 0.5f) / 16.f; }   // u, v < 0.5
        const float w = 1.f - u - v;
        rays[i] = {{w * t.a.x + u * t.b.x + v * t.c.x, w * t.a.y + u * t.b.y + v * t.c.y, -1.f}, 0.f, {0.f, 0.f, 1.f}, 1e16f};
    }
    for (unsigned i = 0; i < LANES && !pairs.empty(); ++i) {
        auto cen = [&](int k) { const Tri &t = all[k]; return float3{(t.a.x + t.b.x + t.c.x) / 3.f, (t.a.y + t.b.y + t.c.y) / 3.f, (t.a.z + t.b.z + t.c.z) / 3.f}; };
        const auto &pr = pairs[i % pairs.size()];
        const float3 s = cen(pr.first), t = cen(pr.second);
        const float3 o{s.x, s.y, s.z - 1.f}, d{t.x - o.x, t.y - o.y, t.z - o.z};
        const float l = std::sqrt(d.x * d.x + d.y * d.y + d.z * d.z);
        rays[i] = {{o.x, o.y, o.z}, 0.f, {d.x / l, d.y / l, d.z / l}, 1e16f};
    }
    Params prm{}; prm.handle = h; prm.n = NR;
    CK(cudaMalloc((void **)&prm.rays, NR * sizeof(Ray8)));
    CK(cudaMemcpy((void *)prm.rays, rays.data(), NR * sizeof(Ray8), cudaMemcpyHostToDevice));
    CK(cudaMalloc((void **)&prm.out, NR * sizeof(RefOut))); CK(cudaMalloc((void **)&prm.inst, 18 * 8));
    CUdeviceptr dprm; CK(cudaMalloc((void **)&dprm, sizeof prm)); CK(cudaMemcpy((void *)dprm, &prm, sizeof prm, cudaMemcpyHostToDevice));
    OX(optixLaunch(pipe, 0, dprm, sizeof prm, &sbt, NR, 1, 1));
    CK(cudaDeviceSynchronize());
    std::vector<float> optix_ms;
    if (optix_repeat > 0) {   // Throughput_7: the same launch again, timed (the first launch above warms caches and JIT)
        cudaEvent_t a0, a1; CK(cudaEventCreate(&a0)); CK(cudaEventCreate(&a1));
        for (int k = 0; k < optix_repeat; ++k) {
            CK(cudaEventRecord(a0)); OX(optixLaunch(pipe, 0, dprm, sizeof prm, &sbt, NR, 1, 1)); CK(cudaEventRecord(a1));
            CK(cudaEventSynchronize(a1)); float ms; CK(cudaEventElapsedTime(&ms, a0, a1)); optix_ms.push_back(ms);
        }
    }
    std::vector<RefOut> ref(NR); unsigned long long inst[18];
    CK(cudaMemcpy(ref.data(), prm.out, NR * sizeof(RefOut), cudaMemcpyDeviceToHost));
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
    std::vector<unsigned char> G;
    auto walk_path = [&](const std::string &path) -> long long {   // slot path from the root -> byte offset, -1 if invalid
        if (G.empty()) { G.resize(bs.outputSizeInBytes); CK(cudaMemcpy(G.data(), (void *)gas, G.size(), cudaMemcpyDeviceToHost)); }
        long long cur = (long long)off; std::istringstream ps(path == "root" ? std::string() : path); std::string tok;
        while (std::getline(ps, tok, ',')) {
            const int want = atoi(tok.c_str()); int rel; memcpy(&rel, &G[cur + 0x1c], 4); int k = 0; bool found = false;
            for (int sl = 0; sl < 12 && !found; ++sl) {
                const unsigned char *b = &G[cur + 0x20 + 8 * sl];
                if (b[4] == 0xff && b[5] == 0 && b[6] == 0 && b[7] == 0) continue;   // empty slot
                if (b[7] == 0) { if (sl == want) { cur += 0x80LL * (rel + k); found = true; } ++k; }
            }
            if (!found || cur < 0 || cur + 0x80 > (long long)G.size()) return -1;
        }
        return cur;
    };
    if (use_path) {   // walk the slot path in this process's GAS (format: see ttu/bench/BVH_5/gas.py)
        start_off = walk_path(start_path.empty() ? "root" : start_path);
        if (start_off < 0) { fprintf(stderr, "--start-path: not a path of internal children\n"); return 2; }
    }
    const bool start_ok = start_off < 0 || (start_off >= (long long)off && start_off + 0x80 <= (long long)bs.outputSizeInBytes);
    const unsigned long long start = start_off < 0 ? root : (unsigned long long)gas + (unsigned long long)start_off;
    std::vector<unsigned long long> ray_start(NR, start);   // --start-file: one slot path per ray
    size_t n_nonroot = 0;
    if (start_file) {
        std::istringstream sf(read_file(start_file)); std::string line; size_t k = 0; std::map<std::string, long long> memo;
        while (std::getline(sf, line) && k < NR) {
            auto it = memo.find(line); long long o = it != memo.end() ? it->second : (memo[line] = walk_path(line));
            if (o < 0) { fprintf(stderr, "--start-file: line %zu is not a path of internal children\n", k + 1); return 2; }
            ray_start[k++] = (unsigned long long)gas + (unsigned long long)o; n_nonroot += o != (long long)off;
        }
        if (k != NR) { fprintf(stderr, "--start-file: %zu lines for %u rays\n", k, NR); return 2; }
    }

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
    fprintf(jf, " \"optix_ms\": [");
    for (size_t k = 0; k < optix_ms.size(); ++k) fprintf(jf, "%s%.6f", k ? "," : "", optix_ms[k]);
    fprintf(jf, "],\n");
    fprintf(jf, " \"start_file\": \"%s\", \"rays_not_from_root\": %zu,", start_file ? start_file : "", n_nonroot);
    fprintf(jf, " \"start_path\": \"%s\",", use_path ? (start_path.empty() ? "root" : start_path.c_str()) : "");
    fprintf(jf, " \"start_off\": \"0x%llx\", \"start\": \"0x%llx\", \"start_flags\": \"0x%x\", \"start_ok\": %s,\n",
            start_off < 0 ? off : (unsigned long long)start_off, start, start_flags, start_ok ? "true" : "false");
    fprintf(jf, " \"lane_rays\": [");
    for (unsigned i = 0; i < NR; ++i)
        fprintf(jf, "%s[%.9g,%.9g,%.9g,%.9g,%.9g,%.9g]", i ? "," : "", rays[i].o[0], rays[i].o[1], rays[i].o[2], rays[i].d[0], rays[i].d[1], rays[i].d[2]);
    fprintf(jf, "],\n");
    fprintf(jf, " \"warps\": %d, \"flush_mb\": %g, \"fill\": \"%s\",\n \"ref\": [", warps, flush_mb, argv[4]);
    for (unsigned i = 0; i < NR; ++i)
        fprintf(jf, "%s{\"hit\": %u, \"t\": \"0x%08x\", \"prim\": %u, \"u\": \"0x%08x\", \"v\": \"0x%08x\", \"front\": %u, \"gas_index\": %u}",
                i ? ", " : "", ref[i].hit, fbits(ref[i].t), ref[i].prim, fbits(ref[i].u), fbits(ref[i].v), ref[i].front, ref[i].gas_index);
    fprintf(jf, "]");
    printf("scene %s (%zu tris, %u inputs) gas 0x%llx size %zu root 0x%llx inst 0x%llx inst_ok %d c3_34 0x%x handle_ok %d align_ok %d\n",
           scene.c_str(), all.size(), ng, (unsigned long long)gas, (size_t)bs.outputSizeInBytes, root, A, inst_ok, c3_34, handle_ok, align_ok);
    printf("optix ref lane 0: hit %u t %g prim %u front %u gas_index %u\n", ref[0].hit, ref[0].t, ref[0].prim, ref[0].front, ref[0].gas_index);
    if (!inst_ok || !handle_ok || !align_ok || !c3_ok || !start_ok) {
        fprintf(jf, "\n}\n"); fclose(jf); fprintf(stderr, "check failed, ttu_one not launched\n"); return 3;
    }
    if (!run_ttu) { fprintf(jf, "\n}\n"); fclose(jf); return 0; }

    if (!sched.empty()) {   // ---- scheduler over the whole ray set (Memory_4a) ----
        auto init_regs = [&](unsigned *R, const Ray8 &y, unsigned long long st0) {   // same image as the chunk loop
            std::fill(R, R + N, 0u);
            R[12] = fbits(y.o[2]); R[13] = fbits(y.tmin); R[82] = fbits(y.o[0]); R[83] = fbits(y.o[1]);
            R[16] = fbits(y.d[2]); R[17] = fbits(y.tmax); R[14] = fbits(y.d[0]); R[15] = fbits(y.d[1]);
            R[52] = R[12]; R[53] = R[13]; R[56] = R[82]; R[57] = R[83];
            R[30] = R[16]; R[31] = R[17]; R[46] = R[14]; R[47] = R[15];
            R[34] = (unsigned)(A >> 6); R[35] = ((((unsigned)(A >> 32) >> 6) | 0x402800u) & 0x1ffffffu) | 0x2000000u;
            R[70] = (unsigned)(st0 >> 7); R[71] = ((unsigned)(st0 >> 32) >> 7) | start_flags;
            R[68] = words[0]; R[69] = words[1]; R[62] = words[2]; R[63] = words[3];
            R[60] = words[4]; R[61] = words[5]; R[58] = words[6]; R[59] = words[7];
            R[18] = sm_a; R[19] = sm_b;
        };
        auto pend = [](const unsigned *R) { return (R[56] | R[57] | R[58] | R[59] | R[60] | R[61] | R[70] | R[71]) != 0; };
        auto absf = [](unsigned u) { u &= 0x7fffffffu; float f; memcpy(&f, &u, 4); return f; };
        const unsigned LL = (unsigned)blocks * LANES;   // lanes per launch (--blocks G: G blocks, one per SM)
        auto sm_of = [&](unsigned lane) -> unsigned {     // the SM a block lands on (measured for Throughput_8);
            if (blocks == 1) return sm_a;                   // a wrong guess makes the lane exit, which the store check catches
            const unsigned b = lane / LANES; return b < 71 ? 2 * b : 2 * (b - 71) + 1;
        };
        cudaEvent_t ev0, ev1; CK(cudaEventCreate(&ev0)); CK(cudaEventCreate(&ev1)); std::vector<float> launch_ms;
        unsigned *din, *dout, *sink; uint4 *fbuf = nullptr; const size_t fn = (size_t)(flush_mb * 1048576.0) / sizeof(uint4);
        CK(cudaMalloc((void **)&din, (size_t)LL * N * 4)); CK(cudaMalloc((void **)&dout, (size_t)LL * N * 4)); CK(cudaMalloc((void **)&sink, 4));
        if (fn) { CK(cudaMalloc((void **)&fbuf, fn * sizeof(uint4))); CK(cudaMemset(fbuf, 1, fn * sizeof(uint4))); }
        cudaError_t e = cudaSuccess;
        std::vector<unsigned> img((size_t)NR * N); std::vector<int> nq, done; std::vector<float> best_t; std::vector<unsigned> best_p;
        std::vector<double> lat; std::string rounds_json; int round = 0; long long launches = 0;
        for (int pass = 0; pass < passes && e == cudaSuccess; ++pass) {
            for (unsigned r = 0; r < NR; ++r) init_regs(&img[(size_t)r * N], rays[r], ray_start[r]);
            nq.assign(NR, 0); done.assign(NR, 0); best_t.assign(NR, -1.f); best_p.assign(NR, 0); lat.assign(NR, 0.0);
            std::vector<unsigned long long> key(NR, 0); rounds_json.clear(); launches = 0; launch_ms.clear();
            for (round = 0; round <= resume && e == cudaSuccess; ++round) {
                std::vector<long long> slots;   // ray per lane, -1 = masked
                if (round == 0 || sched == "inplace") {
                    for (unsigned c = 0; c * LL < NR; ++c) {
                        bool any = false; for (unsigned i = 0; i < LL && c * LL + i < NR; ++i) any |= !done[c * LL + i];
                        if (any) for (unsigned i = 0; i < LL; ++i) { const unsigned r = c * LL + i; slots.push_back(r >= NR || done[r] ? -1 : (long long)r); }
                    }
                } else {
                    std::vector<unsigned> act; for (unsigned r = 0; r < NR; ++r) if (!done[r]) act.push_back(r);
                    if (sched == "regroup") std::stable_sort(act.begin(), act.end(), [&](unsigned a, unsigned b) { return key[a] < key[b]; });
                    for (unsigned r : act) slots.push_back(r);
                    while (slots.size() % LL) slots.push_back(-1);
                }
                if (slots.empty()) break;
                double kpw = 0; int nw = 0; std::set<unsigned long long> allk;   // distinct resume nodes per warp (rounds >= 1)
                for (size_t w0 = 0; w0 < slots.size(); w0 += 32) {
                    std::set<unsigned long long> ks; for (size_t l = w0; l < w0 + 32; ++l) if (slots[l] >= 0) { ks.insert(key[slots[l]]); allk.insert(key[slots[l]]); }
                    if (!ks.empty()) { kpw += ks.size(); ++nw; }
                }
                char hb[192]; snprintf(hb, sizeof hb, "%s{\"round\": %d, \"nodes_per_warp\": %.2f, \"distinct_nodes\": %zu, \"launches\": [", round ? ", " : "", round, round ? kpw / nw : 1.0, round ? allk.size() : (size_t)1); rounds_json += hb;
                std::vector<unsigned> hin((size_t)LL * N), hout((size_t)LL * N);
                for (size_t L0 = 0; L0 < slots.size() && e == cudaSuccess; L0 += LL) {
                    int nact = 0;
                    for (unsigned i = 0; i < LL; ++i) {
                        const long long r = slots[L0 + i]; unsigned *R = &hin[(size_t)i * N];
                        if (r >= 0) { memcpy(R, &img[(size_t)r * N], N * 4); R[18] = R[19] = sm_of(i); ++nact; }
                        else { memcpy(R, &img[0], N * 4); R[18] = R[19] = 0xffffffffu; }   // masked: exits before TTUOPEN
                    }
                    CK(cudaMemcpy(din, hin.data(), hin.size() * 4, cudaMemcpyHostToDevice));
                    std::fill(hout.begin(), hout.end(), 0x77777777u);
                    CK(cudaMemcpy(dout, hout.data(), hout.size() * 4, cudaMemcpyHostToDevice));
                    if (fn) { flush_read<<<142 * 8, 256>>>(fbuf, fn, sink); CK(cudaGetLastError()); }
                    CK(cudaEventRecord(ev0));
                    ttu_one<<<blocks, LANES>>>(din, dout); ++launches;
                    CK(cudaEventRecord(ev1));
                    e = cudaGetLastError(); if (e == cudaSuccess) e = cudaDeviceSynchronize();
                    { float ms = 0; if (e == cudaSuccess) CK(cudaEventElapsedTime(&ms, ev0, ev1)); launch_ms.push_back(ms); }
                    if (e != cudaSuccess) break;
                    CK(cudaMemcpy(hout.data(), dout, hout.size() * 4, cudaMemcpyDeviceToHost));
                    std::string wj;
                    for (int w = 0; w < warps * blocks; ++w) {
                        int l0 = -1; for (int l = 0; l < 32 && l0 < 0; ++l) if (slots[L0 + 32 * w + l] >= 0) l0 = l;
                        if (l0 < 0) continue;
                        const unsigned *O = &hout[(size_t)(32 * w + l0) * N];
                        const unsigned long long s0 = ((unsigned long long)O[85] << 32) | O[84], s1 = ((unsigned long long)O[87] << 32) | O[86];
                        char b[64]; snprintf(b, sizeof b, "%s[%llu,%llu]", wj.empty() ? "" : ",", s0, s1); wj += b;
                        for (int l = 0; l < 32; ++l) { const long long r = slots[L0 + 32 * w + l]; if (r >= 0) lat[r] += (double)(s1 - s0); }
                    }
                    char lb[96]; snprintf(lb, sizeof lb, "%s{\"active\": %d, \"ms\": %.6f, \"warp_clk\": [", L0 ? ", " : "", nact, launch_ms.back()); rounds_json += lb; rounds_json += (blocks == 1 ? wj : std::string()) + "]}";
                    for (unsigned i = 0; i < LL; ++i) {
                        const long long r = slots[L0 + i]; if (r < 0) continue;
                        const unsigned *O = &hout[(size_t)i * N]; unsigned *R = &img[(size_t)r * N];
                        if (O[12] == 0x77777777u && O[84] == 0x77777777u) { fprintf(stderr, "ray %lld: lane did not store\n", r); e = cudaErrorUnknown; break; }
                        ++nq[r];
                        if (O[3]) { const float t = absf(O[45]); if (best_t[r] < 0 || t < best_t[r]) { best_t[r] = t; best_p[r] = O[44] & 0x1fffffffu; } }
                        if (!pend(O)) { done[r] = 1; continue; }
                        key[r] = ((unsigned long long)(O[57] & 0xffu) << 32) | O[56];   // node the resume starts at (address >> 7)
                        R[17] = R[31] = fbits(best_t[r] >= 0.f ? best_t[r] : rays[r].tmax);
                        R[70] = O[56]; R[71] = O[57]; R[32] = O[58]; R[33] = O[59];
                        R[44] = O[60]; R[45] = O[61]; R[2] = O[70]; R[3] = O[71];
                    }
                }
                rounds_json += "]}";
            }
        }
        size_t ndone = 0; int maxq = 0; for (unsigned r = 0; r < NR; ++r) { ndone += done[r]; maxq = std::max(maxq, nq[r]); }
        double tot_ms = 0; for (float m : launch_ms) tot_ms += m;
        fprintf(jf, ",\n \"blocks\": %d, \"total_ms\": %.6f", blocks, tot_ms);
        fprintf(jf, ",\n \"sched\": \"%s\", \"passes\": %d, \"launches_total\": %lld, \"rounds_used\": %d, \"status\": \"%s\",\n \"rounds\": [%s],\n \"ray_queries\": [",
                sched.c_str(), passes, launches, round, cudaGetErrorString(e), rounds_json.c_str());
        for (unsigned r = 0; r < NR; ++r) fprintf(jf, "%s%d", r ? "," : "", done[r] ? nq[r] : -nq[r]);
        fprintf(jf, "],\n \"ray_lat\": [");
        for (unsigned r = 0; r < NR; ++r) fprintf(jf, "%s%.0f", r ? "," : "", lat[r]);
        fprintf(jf, "],\n \"ray_best\": [");
        for (unsigned r = 0; r < NR; ++r) fprintf(jf, "%s[\"0x%08x\", %u]", r ? "," : "", best_t[r] >= 0 ? fbits(best_t[r]) : 0xffffffffu, best_p[r]);
        fprintf(jf, "]\n}\n"); fclose(jf);
        printf("sched %s: %s, %d round(s), %lld launches (last pass), %zu of %u rays finished, max queries %d\n", sched.c_str(), cudaGetErrorString(e), round, launches, ndone, NR, maxq);
        return e == cudaSuccess && ndone == NR ? 0 : 1;
    }
    fprintf(jf, ",\n \"smem\": %d", smem);
    fprintf(jf, ",\n \"nchunks\": %u, \"passes\": %d, \"active\": %d, \"idle\": \"%s\"", nchunks, passes, active, idle.c_str());
    cudaError_t e_all = cudaSuccess;
    for (int pc = 0; pc < passes * (int)nchunks && e_all == cudaSuccess; ++pc) {
    const int pass = pc / (int)nchunks, chunk = pc % (int)nchunks;
    const Ray8 *crays = &rays[(size_t)chunk * LANES];
    if (pc > 0) {   // next chunk file
        const std::string cn = std::string(argv[5]) + ".p" + std::to_string(pass) + "c" + std::to_string(chunk);
        jf = fopen(cn.c_str(), "w"); fprintf(jf, "{\"pass\": %d, \"chunk\": %d", pass, chunk);
    } else fprintf(jf, ",\n \"pass\": 0, \"chunk\": 0");
    // ---- register image for ttu_one: word r of a thread = register r --------------
    const size_t TOTAL = (size_t)blocks * LANES;
    std::vector<unsigned> in(TOTAL * N, 0), out(TOTAL * N, 0x77777777u);
    for (size_t i = 0; i < TOTAL; ++i) {
        unsigned *R = &in[i * N];
        const Ray8 &y = crays[i % LANES];
        R[12] = fbits(y.o[2]); R[13] = fbits(y.tmin); R[82] = fbits(y.o[0]); R[83] = fbits(y.o[1]);   // 0x40
        R[16] = fbits(y.d[2]); R[17] = fbits(y.tmax); R[14] = fbits(y.d[0]); R[15] = fbits(y.d[1]);   // 0x50
        R[52] = R[12]; R[53] = R[13]; R[56] = R[82]; R[57] = R[83];                                   // 0xC0
        R[30] = R[16]; R[31] = R[17]; R[46] = R[14]; R[47] = R[15];                                   // 0xD0
        R[54] = 0; R[55] = 0;                                                                          // 0x100 high
        R[34] = (unsigned)(A >> 6); R[35] = ((((unsigned)(A >> 32) >> 6) | 0x402800u) & 0x1ffffffu) | 0x2000000u;
        const unsigned long long st0 = ray_start[(size_t)chunk * LANES + i % LANES];
        R[70] = (unsigned)(st0 >> 7); R[71] = ((unsigned)(st0 >> 32) >> 7) | start_flags;           // 0x180 low (root by default)
        unsigned *unk[6] = {&R[32], &R[33], &R[2], &R[3], &R[44], &R[45]};                             // 0x180 high, 0x190
        std::mt19937 lane_rng(12345);   // same pattern in every lane
        for (auto u : unk) *u = random_fill ? (unsigned)lane_rng() : 0u;
        R[68] = words[0]; R[69] = words[1]; R[62] = words[2]; R[63] = words[3];                        // 0
        R[60] = words[4]; R[61] = words[5]; R[58] = words[6]; R[59] = words[7];                        // 0x80
        R[90] = 0;   // (R37 & 0xC0000) for flags NONE: P0 true
        R[18] = sm_a; R[19] = sm_b;   // used only by --smfilter builds
        if ((int)(i % 32) >= active) {
            if (idle == "mask") R[18] = R[19] = 0xffffffffu;   // no SM has this id: the lane exits before TTUOPEN
            else if (idle == "tmax0") R[17] = R[31] = 0u;        // tmax 0 in 0x50 and 0xD0
            else if (idle == "tneg") R[17] = R[31] = fbits(-1.f); // tmax -1
            else { R[13] = R[53] = fbits(1.f); R[17] = R[31] = 0u; }   // empty: tmin 1 > tmax 0 (0x40 and 0xC0 high)
        }
    }
    unsigned char *tbuf = nullptr;
    if (thrash) {
        CK(cudaMalloc((void **)&tbuf, 4u << 20)); CK(cudaMemset(tbuf, 0, 4u << 20));
        for (size_t i = 0; i < TOTAL; ++i) {
            const unsigned long long a = (unsigned long long)tbuf + 128ull * (i % 32);
            in[i * N + 26] = (unsigned)a; in[i * N + 27] = (unsigned)(a >> 32);
        }
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
        if (smem && it == 0) CK(cudaFuncSetAttribute(ttu_one, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        ttu_one<<<blocks, LANES, smem>>>(din, dout);
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
    // ---- resume: follow up queries per lane, built the way OptiX's traversal loop builds them ----
    // OptiX (fnaddr-0xd430) sets tmax = |R45| (best t), then loops to fnaddr-0x10000, which stores
    // 0x180 = {R56:R57, R58:R59} and 0x190 = {R60:R61, R70:R71} from the previous TTULD 0x3C0 / 0x3D0.
    if (resume > 0 && e == cudaSuccess && blocks == 1) {
        auto pend = [](const unsigned *R) { return (R[56] | R[57] | R[58] | R[59] | R[60] | R[61] | R[70] | R[71]) != 0; };
        auto absf = [](unsigned u) { u &= 0x7fffffffu; float f; memcpy(&f, &u, 4); return f; };
        std::vector<unsigned> cur = in, prev = out;
        std::vector<int> nq(LANES, 1), done(LANES, 0), npend0(LANES, 0);
        std::vector<float> best_t(LANES, -1.f); std::vector<unsigned> best_p(LANES, 0); std::vector<unsigned long long> best_leaf(LANES, 0);
        auto take = [&](unsigned i, const unsigned *R) {
            if (R[3]) { const float t = absf(R[45]); if (best_t[i] < 0 || t < best_t[i]) { best_t[i] = t; best_p[i] = R[44] & 0x1fffffffu; best_leaf[i] = ((unsigned long long)R[31] << 32) | R[30]; } }
        };
        auto npairs = [](const unsigned *R) { return (int)((R[56] | R[57]) != 0) + ((R[58] | R[59]) != 0) + ((R[60] | R[61]) != 0) + ((R[70] | R[71]) != 0); };
        for (unsigned i = 0; i < LANES; ++i) { take(i, &prev[i * N]); done[i] = !pend(&prev[i * N]); npend0[i] = npairs(&prev[i * N]); }
        fprintf(jf, ",\n \"resume_rounds\": [");
        int round = 0;
        for (; round < resume; ++round) {
            bool any = false; for (unsigned i = 0; i < LANES; ++i) any |= !done[i];
            if (!any) break;
            for (unsigned i = 0; i < LANES; ++i) {
                unsigned *R = &cur[i * N]; const unsigned *Pv = &prev[i * N];
                if (done[i]) { R[17] = R[31] = 0u; continue; }          // finished lane: tmax 0, no work
                const float tmax = best_t[i] >= 0.f ? best_t[i] : crays[i % LANES].tmax;
                R[17] = R[31] = fbits(tmax);
                R[70] = Pv[56]; R[71] = Pv[57]; R[32] = Pv[58]; R[33] = Pv[59];       // 0x180
                R[44] = Pv[60]; R[45] = Pv[61]; R[2] = Pv[70]; R[3] = Pv[71];         // 0x190
            }
            CK(cudaMemcpy(din, cur.data(), cur.size() * 4, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dout, out.data(), out.size() * 4, cudaMemcpyHostToDevice));
            std::vector<unsigned> fill(out.size(), 0x77777777u);
            CK(cudaMemcpy(dout, fill.data(), fill.size() * 4, cudaMemcpyHostToDevice));
            if (fn) { flush_read<<<142 * 8, 256>>>(fbuf, fn, sink); CK(cudaGetLastError()); }
            ttu_one<<<1, LANES>>>(din, dout);
            e = cudaGetLastError(); if (e == cudaSuccess) e = cudaDeviceSynchronize();
            if (e != cudaSuccess) break;
            std::vector<unsigned> o(out.size());
            CK(cudaMemcpy(o.data(), dout, o.size() * 4, cudaMemcpyDeviceToHost));
            int active = 0, still = 0;
            fprintf(jf, "%s{\"round\": %d, \"warp_clk\": [", round ? ", " : "", round + 1);
            for (int w = 0; w < warps; ++w) {
                const unsigned *L0 = &o[(32 * w) * N];
                fprintf(jf, "%s[%llu,%llu]", w ? "," : "", ((unsigned long long)L0[85] << 32) | L0[84], ((unsigned long long)L0[87] << 32) | L0[86]);
            }
            for (unsigned i = 0; i < LANES; ++i) {
                if (done[i]) continue;
                ++active; ++nq[i]; take(i, &o[i * N]);
                done[i] = !pend(&o[i * N]); still += !done[i];
            }
            fprintf(jf, "], \"active\": %d, \"still_pending\": %d}", active, still);
            prev = o;
        }
        fprintf(jf, "],\n \"resume_status\": \"%s\", \"resume_rounds_used\": %d,\n \"lane_queries\": [", cudaGetErrorString(e), round);
        for (unsigned i = 0; i < LANES; ++i) fprintf(jf, "%s%d", i ? "," : "", done[i] ? nq[i] : -nq[i]);
        fprintf(jf, "],\n \"lane_pending_pairs_first\": [");
        for (unsigned i = 0; i < LANES; ++i) fprintf(jf, "%s%d", i ? "," : "", npend0[i]);
        fprintf(jf, "],\n \"lane_best\": [");
        for (unsigned i = 0; i < LANES; ++i) fprintf(jf, "%s[\"0x%08x\", %u]", i ? "," : "", best_t[i] >= 0 ? fbits(best_t[i]) : 0xffffffffu, best_p[i]);
        fprintf(jf, "],\n \"lane_leaf_off\": [");   // TTULD 0x310 high of the best hit: leaf block (low 48 bits) minus GAS base, slot bits dropped
        for (unsigned i = 0; i < LANES; ++i) fprintf(jf, "%s%lld", i ? "," : "", best_t[i] >= 0 ? (long long)((best_leaf[i] & 0xffffffffffffULL) - (unsigned long long)gas) : -1LL);
        fprintf(jf, "]");
        int maxq = 0; size_t ndone = 0; for (unsigned i = 0; i < LANES; ++i) { maxq = std::max(maxq, nq[i]); ndone += done[i]; }
        printf("resume: %s, %d round(s), %zu of %u lanes finished, max queries per lane %d\n", cudaGetErrorString(e), round, ndone, LANES, maxq);
    }
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
    CK(cudaFree(din)); CK(cudaFree(dout)); CK(cudaFree(sink)); if (fbuf) CK(cudaFree(fbuf)); if (tbuf) CK(cudaFree(tbuf));
    e_all = e;
    }
    return e_all == cudaSuccess ? 0 : 1;   // pipeline and GAS stay alive until here
}
