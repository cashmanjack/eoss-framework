// Treelet_1: kill test for TPC pinned treelet queues, OptiX proxy (no TTU code anywhere).
//   treelet1 tlsize PTX WORK N K                      build the K^3 treelet GASs of the N triangle soup, print their sizes
//   treelet1 prep   PTX WORK N S                      S random origin and direction rays and S depth 3 bounce rays, hit only, written to WORK (natural order)
//   treelet1 run    PTX WORK N K SET WPS ROUNDS OUT.json ARM...      timed arms, rotated order, every result checked against the full GAS closest hit
//   treelet1 profile PTX WORK N K SET WPS 0 OUT.json ARM...          every arm twice (warm, measured), for ncu
// ARM = LAYOUT_START: LAYOUT nat (natural order, one launch index per ray), grp (rays sorted by treelet, same launch), pn (persistent warps, one queue, natural),
//   pg (persistent, one queue, grouped), pp (persistent, grouped, one queue per TPC, warp takes the queue of its own TPC, steals only when it is empty).
//   START R = every ray traces the full GAS from its root; T = every ray traces the GAS of the treelet holding its closest hit (oracle key, as Memory_4).
// OUT.json is opened with mode x (never overwritten).
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <nvml.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include "treelet1_params.h"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(2); } } while (0)
#define OX(x) do { OptixResult r_ = (x); if (r_ != OPTIX_SUCCESS) { fprintf(stderr, "%s:%d %s: %d\n", __FILE__, __LINE__, #x, (int)r_); exit(2); } } while (0)
struct Tri { float3 a, b, c; };
static std::string read_file(const char *p) { std::ifstream f(p); if (!f) { fprintf(stderr, "cannot read %s\n", p); exit(2); } std::stringstream s; s << f.rdbuf(); return s.str(); }
template <class T> static std::vector<T> read_bin(const std::string &p) { FILE *f = fopen(p.c_str(), "rb"); if (!f) { fprintf(stderr, "cannot read %s\n", p.c_str()); exit(2); } fseek(f, 0, SEEK_END); size_t n = ftell(f) / sizeof(T); fseek(f, 0, SEEK_SET); std::vector<T> v(n); if (fread(v.data(), sizeof(T), n, f) != n) exit(2); fclose(f); return v; }
template <class T> static void write_bin(const std::string &p, const std::vector<T> &v) { FILE *f = fopen(p.c_str(), "wbx"); if (!f) { fprintf(stderr, "refusing to overwrite or cannot write %s\n", p.c_str()); exit(2); } fwrite(v.data(), sizeof(T), v.size(), f); fclose(f); }

static OptixDeviceContext ctx;
struct Gas { OptixTraversableHandle h; CUdeviceptr buf; size_t bytes; };

static OptixBuildInput tri_input(CUdeviceptr *vb, unsigned ntri, unsigned *flag)
{
    OptixBuildInput bi{}; bi.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES; *flag = OPTIX_GEOMETRY_FLAG_NONE;
    bi.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3; bi.triangleArray.numVertices = ntri * 3; bi.triangleArray.vertexBuffers = vb;
    bi.triangleArray.flags = flag; bi.triangleArray.numSbtRecords = 1; return bi;
}
static size_t al128(size_t x) { return (x + 127) & ~(size_t)127; }

// one GAS, compacted
static Gas build_one(CUdeviceptr verts, unsigned ntri)
{
    unsigned fl; CUdeviceptr vb = verts; OptixBuildInput bi = tri_input(&vb, ntri, &fl);
    OptixAccelBuildOptions ao{}; ao.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION; ao.operation = OPTIX_BUILD_OPERATION_BUILD;
    OptixAccelBufferSizes bs{}; OX(optixAccelComputeMemoryUsage(ctx, &ao, &bi, 1, &bs));
    CUdeviceptr tmp, out, dsz; CK(cudaMalloc((void **)&tmp, bs.tempSizeInBytes)); CK(cudaMalloc((void **)&out, bs.outputSizeInBytes)); CK(cudaMalloc((void **)&dsz, 8));
    OptixAccelEmitDesc em{}; em.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE; em.result = dsz;
    OptixTraversableHandle h0; OX(optixAccelBuild(ctx, 0, &ao, &bi, 1, tmp, bs.tempSizeInBytes, out, bs.outputSizeInBytes, &h0, &em, 1));
    CK(cudaDeviceSynchronize()); size_t csz; CK(cudaMemcpy(&csz, (void *)dsz, 8, cudaMemcpyDeviceToHost));
    Gas g; g.bytes = csz; CK(cudaMalloc((void **)&g.buf, csz)); OX(optixAccelCompact(ctx, 0, h0, g.buf, csz, &g.h)); CK(cudaDeviceSynchronize());
    CK(cudaFree((void *)tmp)); CK(cudaFree((void *)out)); CK(cudaFree((void *)dsz)); return g;
}

// treelet GASs, vertices already sorted by treelet: toff[t] = first triangle; all compacted into one buffer
struct Treelets { std::vector<OptixTraversableHandle> h; std::vector<size_t> bytes; CUdeviceptr buf; size_t total; };
static Treelets build_treelets(CUdeviceptr sorted_verts, const std::vector<unsigned> &toff)
{
    const unsigned T = (unsigned)toff.size() - 1; Treelets r; r.h.resize(T); r.bytes.resize(T);
    OptixAccelBuildOptions ao{}; ao.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION; ao.operation = OPTIX_BUILD_OPERATION_BUILD;
    std::vector<OptixAccelBufferSizes> bs(T); std::vector<size_t> off(T); size_t tot = 0, maxtmp = 0;
    for (unsigned t = 0; t < T; ++t) {
        unsigned fl; CUdeviceptr vb = sorted_verts + (CUdeviceptr)toff[t] * 36; OptixBuildInput bi = tri_input(&vb, toff[t + 1] - toff[t], &fl);
        OX(optixAccelComputeMemoryUsage(ctx, &ao, &bi, 1, &bs[t])); off[t] = tot; tot += al128(bs[t].outputSizeInBytes); maxtmp = std::max(maxtmp, (size_t)bs[t].tempSizeInBytes);
    }
    CUdeviceptr big, tmp, dsz; CK(cudaMalloc((void **)&big, tot)); CK(cudaMalloc((void **)&tmp, maxtmp)); CK(cudaMalloc((void **)&dsz, 8ull * T));
    std::vector<OptixTraversableHandle> h0(T);
    for (unsigned t = 0; t < T; ++t) {
        unsigned fl; CUdeviceptr vb = sorted_verts + (CUdeviceptr)toff[t] * 36; OptixBuildInput bi = tri_input(&vb, toff[t + 1] - toff[t], &fl);
        OptixAccelEmitDesc em{}; em.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE; em.result = dsz + 8ull * t;
        OX(optixAccelBuild(ctx, 0, &ao, &bi, 1, tmp, bs[t].tempSizeInBytes, big + off[t], bs[t].outputSizeInBytes, &h0[t], &em, 1));
    }
    CK(cudaDeviceSynchronize()); std::vector<size_t> csz(T); CK(cudaMemcpy(csz.data(), (void *)dsz, 8ull * T, cudaMemcpyDeviceToHost));
    std::vector<size_t> off2(T); size_t tot2 = 0; for (unsigned t = 0; t < T; ++t) { off2[t] = tot2; tot2 += al128(csz[t]); r.bytes[t] = csz[t]; }
    CK(cudaMalloc((void **)&r.buf, tot2)); r.total = tot2;
    for (unsigned t = 0; t < T; ++t) OX(optixAccelCompact(ctx, 0, h0[t], r.buf + off2[t], csz[t], &r.h[t]));
    CK(cudaDeviceSynchronize()); CK(cudaFree((void *)big)); CK(cudaFree((void *)tmp)); CK(cudaFree((void *)dsz)); return r;
}

static std::vector<Tri> gen_scene(unsigned n_tri)   // same generator as setup3.cu soup:N and morton.cu (edge 3 / cbrt(N), seed 1234)
{
    const float e = 3.f / std::cbrt((float)n_tri); std::mt19937 g(1234); std::uniform_real_distribution<float> U(-1.f, 1.f);
    std::vector<Tri> v; v.reserve(n_tri);
    for (unsigned i = 0; i < n_tri; ++i) { const float3 c{U(g), U(g), U(g)}; auto pt = [&]() { return float3{c.x + 0.5f * e * U(g), c.y + 0.5f * e * U(g), c.z + 0.5f * e * U(g)}; }; v.push_back({pt(), pt(), pt()}); }
    return v;
}
static unsigned morton3(unsigned x, unsigned y, unsigned z) { unsigned c = 0; for (int b = 0; b < 10; ++b) c |= (((x >> b) & 1u) << (3 * b)) | (((y >> b) & 1u) << (3 * b + 1)) | (((z >> b) & 1u) << (3 * b + 2)); return c; }

// ---- pipeline ----
struct Pipe { OptixPipeline pipe; OptixShaderBindingTable sbt[2]; CUdeviceptr drec; };
static Pipe make_pipe(const char *ptxfile)
{
    OptixModuleCompileOptions mo{}; OptixPipelineCompileOptions po{};
    po.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS; po.numPayloadValues = 7; po.numAttributeValues = 2;
    po.pipelineLaunchParamsVariableName = "params"; po.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
    std::string ptx = read_file(ptxfile); char log[4096]; size_t ls = sizeof log; OptixModule m; { OptixResult mr = optixModuleCreate(ctx, &mo, &po, ptx.c_str(), ptx.size(), log, &ls, &m); if (mr != OPTIX_SUCCESS) { fprintf(stderr, "optixModuleCreate %d: %s\n", (int)mr, log); exit(2); } }
    OptixProgramGroupOptions go{}; OptixProgramGroupDesc d[4] = {};
    d[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN; d[0].raygen.module = m; d[0].raygen.entryFunctionName = "__raygen__direct";
    d[1].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN; d[1].raygen.module = m; d[1].raygen.entryFunctionName = "__raygen__persist";
    d[2].kind = OPTIX_PROGRAM_GROUP_KIND_MISS; d[2].miss.module = m; d[2].miss.entryFunctionName = "__miss__ms";
    d[3].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP; d[3].hitgroup.moduleCH = m; d[3].hitgroup.entryFunctionNameCH = "__closesthit__ch";
    OptixProgramGroup pg[4]; ls = sizeof log; OX(optixProgramGroupCreate(ctx, d, 4, &go, log, &ls, pg));
    OptixPipelineLinkOptions lo{}; lo.maxTraceDepth = 1; Pipe p; ls = sizeof log; OX(optixPipelineCreate(ctx, &po, &lo, pg, 4, log, &ls, &p.pipe));
    struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) Rec { char h[OPTIX_SBT_RECORD_HEADER_SIZE]; };
    Rec rec[4]; for (int k = 0; k < 4; ++k) OX(optixSbtRecordPackHeader(pg[k], &rec[k]));
    CK(cudaMalloc((void **)&p.drec, sizeof rec)); CK(cudaMemcpy((void *)p.drec, rec, sizeof rec, cudaMemcpyHostToDevice));
    for (int k = 0; k < 2; ++k) { p.sbt[k] = {}; p.sbt[k].raygenRecord = p.drec + k * sizeof(Rec); p.sbt[k].missRecordBase = p.drec + 2 * sizeof(Rec); p.sbt[k].missRecordStrideInBytes = sizeof(Rec); p.sbt[k].missRecordCount = 1;
        p.sbt[k].hitgroupRecordBase = p.drec + 3 * sizeof(Rec); p.sbt[k].hitgroupRecordStrideInBytes = sizeof(Rec); p.sbt[k].hitgroupRecordCount = 1; }
    return p;
}

static std::vector<RefOut> trace_full(Pipe &pp, OptixTraversableHandle h, const std::vector<Ray8> &rays)
{
    Params prm{}; prm.handle = h; prm.n = (unsigned)rays.size(); prm.use_tl = 0;
    CK(cudaMalloc((void **)&prm.rays, rays.size() * sizeof(Ray8))); CK(cudaMemcpy((void *)prm.rays, rays.data(), rays.size() * sizeof(Ray8), cudaMemcpyHostToDevice));
    CK(cudaMalloc((void **)&prm.out, rays.size() * sizeof(RefOut)));
    CUdeviceptr dprm; CK(cudaMalloc((void **)&dprm, sizeof prm)); CK(cudaMemcpy((void *)dprm, &prm, sizeof prm, cudaMemcpyHostToDevice));
    OX(optixLaunch(pp.pipe, 0, dprm, sizeof prm, &pp.sbt[0], (rays.size() + 31) / 32 * 32, 1, 1)); CK(cudaDeviceSynchronize());
    std::vector<RefOut> o(rays.size()); CK(cudaMemcpy(o.data(), prm.out, o.size() * sizeof(RefOut), cudaMemcpyDeviceToHost));
    CK(cudaFree((void *)prm.rays)); CK(cudaFree(prm.out)); CK(cudaFree((void *)dprm)); return o;
}

static Ray8 bounce_ray(const Tri &t, const Ray8 &r, float tt, std::mt19937 &g)
{
    std::uniform_real_distribution<float> U(0.f, 1.f);
    float ux = t.b.x - t.a.x, uy = t.b.y - t.a.y, uz = t.b.z - t.a.z, vx = t.c.x - t.a.x, vy = t.c.y - t.a.y, vz = t.c.z - t.a.z;
    float nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx; const float il = 1.f / std::sqrt(nx * nx + ny * ny + nz * nz + 1e-30f); nx *= il; ny *= il; nz *= il;
    if (nx * r.d[0] + ny * r.d[1] + nz * r.d[2] > 0.f) { nx = -nx; ny = -ny; nz = -nz; }
    const float u1 = U(g), u2 = U(g), rr = std::sqrt(u1), ph = 6.2831853f * u2, x = rr * std::cos(ph), y = rr * std::sin(ph), z = std::sqrt(std::max(0.f, 1.f - u1));
    const float ax = std::fabs(nx) > 0.9f ? 0.f : 1.f, ay = std::fabs(nx) > 0.9f ? 1.f : 0.f; float tx = ay * nz, ty = -ax * nz, tz = ax * ny - ay * nx; const float it = 1.f / std::sqrt(tx * tx + ty * ty + tz * tz + 1e-30f); tx *= it; ty *= it; tz *= it;
    const float bx = ny * tz - nz * ty, by = nz * tx - nx * tz, bz = nx * ty - ny * tx;
    Ray8 o; o.o[0] = r.o[0] + tt * r.d[0] + 1e-4f * nx; o.o[1] = r.o[1] + tt * r.d[1] + 1e-4f * ny; o.o[2] = r.o[2] + tt * r.d[2] + 1e-4f * nz;
    o.tmin = 0.f; o.d[0] = x * tx + y * bx + z * nx; o.d[1] = x * ty + y * by + z * ny; o.d[2] = x * tz + y * bz + z * nz; o.tmax = 1e16f; return o;
}

struct Arm { std::string name; int layout; int persistent; int pinned; int use_tl; };   // layout 0 natural, 1 grouped
static Arm parse_arm(const std::string &s)
{
    const size_t u = s.find('_'); const std::string a = s.substr(0, u), st = s.substr(u + 1); Arm r{s, 0, 0, 0, st == "T"};
    if (a == "nat") {} else if (a == "grp") r.layout = 1; else if (a == "pn") r.persistent = 1; else if (a == "pg") { r.persistent = 1; r.layout = 1; } else if (a == "pp") { r.persistent = 1; r.layout = 1; r.pinned = 1; }
    else { fprintf(stderr, "bad arm %s\n", s.c_str()); exit(2); }
    return r;
}

int main(int argc, char **argv)
{
    if (argc < 6) { fprintf(stderr, "usage: see the top of treelet1.cu\n"); return 2; }
    const std::string mode = argv[1], work = argv[3]; const unsigned N = (unsigned)atoi(argv[4]);
    CK(cudaFree(0)); OX(optixInit()); OptixDeviceContextOptions co{}; OX(optixDeviceContextCreate(0, &co, &ctx));
    cudaDeviceProp dp; CK(cudaGetDeviceProperties(&dp, 0)); const unsigned nsm = dp.multiProcessorCount, nq = nsm / 2;
    std::vector<Tri> tris = gen_scene(N);
    Pipe pp = make_pipe(argv[2]);

    if (mode == "prep") {
        const unsigned S = (unsigned)atoi(argv[5]);
        CUdeviceptr dv; CK(cudaMalloc((void **)&dv, tris.size() * sizeof(Tri))); CK(cudaMemcpy((void *)dv, tris.data(), tris.size() * sizeof(Tri), cudaMemcpyHostToDevice));
        Gas g = build_one(dv, N); printf("{\"full_gas_bytes\": %zu, \"tris\": %u}\n", g.bytes, N); fflush(stdout);
        std::mt19937 rg(777); std::uniform_real_distribution<float> U(-1.f, 1.f); std::normal_distribution<float> Nn(0.f, 1.f);
        std::vector<Ray8> r0(S);
        for (unsigned i = 0; i < S; ++i) { float dx, dy, dz, l; do { dx = Nn(rg); dy = Nn(rg); dz = Nn(rg); l = std::sqrt(dx * dx + dy * dy + dz * dz); } while (l < 1e-6f);
            r0[i] = Ray8{{U(rg), U(rg), U(rg)}, 0.f, {dx / l, dy / l, dz / l}, 1e16f}; }
        auto save = [&](const char *tag, const std::vector<Ray8> &r) {   // trace on the full GAS, keep the hits, truncate to a multiple of 32
            std::vector<RefOut> o = trace_full(pp, g.h, r); std::vector<Ray8> kr; std::vector<RefOut> ko;
            for (size_t i = 0; i < r.size(); ++i) if (o[i].hit) { kr.push_back(r[i]); ko.push_back(o[i]); }
            kr.resize(kr.size() / 32 * 32); ko.resize(kr.size());
            write_bin(work + "/" + tag + "_rays.bin", kr); write_bin(work + "/" + tag + "_ref.bin", ko);
            printf("{\"set\": \"%s\", \"generated\": %zu, \"hits_kept\": %zu}\n", tag, r.size(), kr.size()); fflush(stdout); };
        save("rand", r0);
        std::mt19937 rb(4242); std::vector<Ray8> cur = r0;   // depth 3 bounce chain from the same random primaries, dead paths dropped, order of the paths kept
        for (int depth = 0; depth < 3; ++depth) {
            std::vector<RefOut> o = trace_full(pp, g.h, cur); std::vector<Ray8> nx;
            for (size_t i = 0; i < cur.size(); ++i) if (o[i].hit) nx.push_back(bounce_ray(tris[o[i].prim], cur[i], o[i].t, rb));
            printf("{\"bounce\": %d, \"alive_in\": %zu, \"alive_out\": %zu}\n", depth + 1, cur.size(), nx.size()); cur.swap(nx);
        }
        save("b3", cur); return 0;
    }

    // ---- treelet partition: K^3 cells (centroid), treelet id = Morton rank of the cell ----
    const unsigned K = (unsigned)atoi(argv[5]); const unsigned C = K * K * K;
    std::vector<unsigned> cell_of(N), cellcount(C, 0), code(C), rank_of_cell(C);
    for (unsigned i = 0; i < N; ++i) {
        const Tri &t = tris[i]; const float cx = (t.a.x + t.b.x + t.c.x) / 3.f, cy = (t.a.y + t.b.y + t.c.y) / 3.f, cz = (t.a.z + t.b.z + t.c.z) / 3.f;
        auto q = [&](float v) { int k = (int)std::floor((v + 1.f) * 0.5f * K); return (unsigned)std::min<int>(std::max<int>(k, 0), K - 1); };
        const unsigned x = q(cx), y = q(cy), z = q(cz); cell_of[i] = (z * K + y) * K + x; ++cellcount[cell_of[i]];
    }
    std::vector<unsigned> ord(C); std::iota(ord.begin(), ord.end(), 0);
    for (unsigned c = 0; c < C; ++c) code[c] = morton3(c % K, (c / K) % K, c / (K * K));
    std::stable_sort(ord.begin(), ord.end(), [&](unsigned a, unsigned b) { return code[a] < code[b]; });
    for (unsigned r = 0; r < C; ++r) rank_of_cell[ord[r]] = r;
    std::vector<unsigned> toff(C + 1, 0); for (unsigned r = 0; r < C; ++r) toff[r + 1] = toff[r] + cellcount[ord[r]];
    std::vector<unsigned> tl_of_tri(N), tri_sorted(N), fillp(toff.begin(), toff.end() - 1);
    for (unsigned i = 0; i < N; ++i) { const unsigned t = rank_of_cell[cell_of[i]]; tl_of_tri[i] = t; tri_sorted[fillp[t]++] = i; }
    std::vector<Tri> sv(N); for (unsigned j = 0; j < N; ++j) sv[j] = tris[tri_sorted[j]];
    CUdeviceptr dsv; CK(cudaMalloc((void **)&dsv, (size_t)N * sizeof(Tri))); CK(cudaMemcpy((void *)dsv, sv.data(), (size_t)N * sizeof(Tri), cudaMemcpyHostToDevice));
    Treelets tl = build_treelets(dsv, toff);
    { std::vector<size_t> b = tl.bytes; std::sort(b.begin(), b.end());
      printf("{\"treelets\": %u, \"tris_per_treelet_mean\": %.1f, \"gas_bytes_min\": %zu, \"gas_bytes_median\": %zu, \"gas_bytes_max\": %zu, \"gas_bytes_total\": %zu, \"sm\": %u, \"tpc_queues\": %u}\n",
             C, (double)N / C, b.front(), b[b.size() / 2], b.back(), tl.total, nsm, nq); fflush(stdout); }
    if (mode == "tlsize") return 0;

    // ---- run / profile ----
    const std::string set = argv[6]; const unsigned WPS = (unsigned)atoi(argv[7]); const int ROUNDS = atoi(argv[8]); const char *outjson = argv[9];
    std::vector<Arm> arms; for (int a = 10; a < argc; ++a) arms.push_back(parse_arm(argv[a]));
    CUdeviceptr dfull; CK(cudaMalloc((void **)&dfull, tris.size() * sizeof(Tri))); CK(cudaMemcpy((void *)dfull, tris.data(), tris.size() * sizeof(Tri), cudaMemcpyHostToDevice));
    Gas full = build_one(dfull, N);
    std::vector<Ray8> rays = read_bin<Ray8>(work + "/" + set + "_rays.bin"); std::vector<RefOut> ref = read_bin<RefOut>(work + "/" + set + "_ref.bin"); const unsigned n = (unsigned)rays.size();
    // layouts: 0 natural, 1 grouped by treelet (stable)
    std::vector<unsigned> src[2], tlv[2]; std::vector<Ray8> ry[2];
    src[0].resize(n); std::iota(src[0].begin(), src[0].end(), 0u);
    src[1] = src[0]; std::stable_sort(src[1].begin(), src[1].end(), [&](unsigned a, unsigned b) { return tl_of_tri[ref[a].prim] < tl_of_tri[ref[b].prim]; });
    for (int l = 0; l < 2; ++l) { ry[l].resize(n); tlv[l].resize(n); for (unsigned i = 0; i < n; ++i) { ry[l][i] = rays[src[l][i]]; tlv[l][i] = tl_of_tri[ref[src[l][i]].prim]; } }
    // TPC queues over the grouped layout: equal ray counts, cut at treelet boundaries
    std::vector<unsigned> tstart(C + 1, 0); for (unsigned i = 0; i < n; ++i) ++tstart[tlv[1][i] + 1]; for (unsigned t = 0; t < C; ++t) tstart[t + 1] += tstart[t];
    std::vector<unsigned> qbeg(nq), qend(nq);
    for (unsigned q = 0; q < nq; ++q) { auto cut = [&](unsigned k) { if (k == 0) return 0u; if (k == nq) return n; const unsigned tgt = (unsigned)((unsigned long long)k * n / nq); return *std::lower_bound(tstart.begin(), tstart.end(), tgt); }; qbeg[q] = cut(q); qend[q] = cut(q + 1); }
    // device buffers
    CUdeviceptr d_ry[2], d_tl[2], d_h, d_qend, d_qend1, d_head, d_stats, d_smhist, d_warpsm, d_prm; RefOut *d_out;
    for (int l = 0; l < 2; ++l) { CK(cudaMalloc((void **)&d_ry[l], n * sizeof(Ray8))); CK(cudaMemcpy((void *)d_ry[l], ry[l].data(), n * sizeof(Ray8), cudaMemcpyHostToDevice)); CK(cudaMalloc((void **)&d_tl[l], n * 4)); CK(cudaMemcpy((void *)d_tl[l], tlv[l].data(), n * 4, cudaMemcpyHostToDevice)); }
    CK(cudaMalloc((void **)&d_h, 8ull * C)); CK(cudaMemcpy((void *)d_h, tl.h.data(), 8ull * C, cudaMemcpyHostToDevice));
    CK(cudaMalloc((void **)&d_qend, 4ull * nq)); CK(cudaMemcpy((void *)d_qend, qend.data(), 4ull * nq, cudaMemcpyHostToDevice)); CK(cudaMalloc((void **)&d_qend1, 4)); CK(cudaMemcpy((void *)d_qend1, &n, 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc((void **)&d_head, 4ull * nq)); CK(cudaMalloc((void **)&d_stats, 16)); CK(cudaMalloc((void **)&d_smhist, 1024)); const unsigned maxwarps = 4096; CK(cudaMalloc((void **)&d_warpsm, 4ull * maxwarps)); CK(cudaMalloc((void **)&d_prm, sizeof(Params)));
    CK(cudaMalloc((void **)&d_out, (size_t)n * sizeof(RefOut)));
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    nvmlDevice_t nd; char bus[32]; CK(cudaDeviceGetPCIBusId(bus, sizeof bus, 0)); bool nv_ok = nvmlInit_v2() == NVML_SUCCESS && nvmlDeviceGetHandleByPciBusId_v2(bus, &nd) == NVML_SUCCESS;
    const unsigned PW = nsm * WPS;   // persistent warps launched
    if (PW > maxwarps) { fprintf(stderr, "WPS too large\n"); return 2; }

    struct Res { std::vector<double> ms; std::vector<unsigned> clk; unsigned stats[4]; unsigned smmin, smmax; unsigned wmin, wmax; unsigned long bad, tbad; };
    std::vector<Res> res(arms.size());
    auto launch = [&](const Arm &a, Res &r, bool timed, bool verify) {
        Params prm{}; prm.rays = (const Ray8 *)d_ry[a.layout]; prm.out = d_out; prm.handle = full.h; prm.tl_handle = (const OptixTraversableHandle *)d_h; prm.tl = (const unsigned *)d_tl[a.layout]; prm.n = n; prm.use_tl = a.use_tl; prm.chunk = getenv("T1_CHUNK") ? (unsigned)atoi(getenv("T1_CHUNK")) : 8u;
        prm.head = (unsigned *)d_head; prm.qend = (const unsigned *)(a.pinned ? d_qend : d_qend1); prm.nq = a.pinned ? nq : 1; prm.pinned = a.pinned; prm.stats = (unsigned *)d_stats; prm.smhist = (unsigned *)d_smhist; prm.warp_sm = (unsigned *)d_warpsm;
        std::vector<unsigned> hb(nq, 0); if (a.pinned) hb = qbeg;
        CK(cudaMemcpy((void *)d_head, hb.data(), 4ull * nq, cudaMemcpyHostToDevice)); CK(cudaMemset((void *)d_stats, 0, 16)); CK(cudaMemset((void *)d_smhist, 0, 1024)); CK(cudaMemset((void *)d_warpsm, 0xff, 4ull * maxwarps));
        CK(cudaMemset(d_out, 0xff, (size_t)n * sizeof(RefOut)));   // the same 268 MB write before every launch, for every arm
        CK(cudaMemcpy((void *)d_prm, &prm, sizeof prm, cudaMemcpyHostToDevice)); CK(cudaDeviceSynchronize());
        CK(cudaEventRecord(e0));
        if (a.persistent) OX(optixLaunch(pp.pipe, 0, d_prm, sizeof prm, &pp.sbt[1], PW * 32, 1, 1)); else OX(optixLaunch(pp.pipe, 0, d_prm, sizeof prm, &pp.sbt[0], (n + 31) / 32 * 32, 1, 1));
        CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
        float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); unsigned c = 0; if (nv_ok) nvmlDeviceGetClockInfo(nd, NVML_CLOCK_SM, &c);
        if (timed) { r.ms.push_back(ms); r.clk.push_back(c); }
        if (a.persistent) {
            CK(cudaMemcpy(r.stats, (void *)d_stats, 16, cudaMemcpyDeviceToHost)); std::vector<unsigned> sh(256), ws(maxwarps); CK(cudaMemcpy(sh.data(), (void *)d_smhist, 1024, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(ws.data(), (void *)d_warpsm, 4ull * maxwarps, cudaMemcpyDeviceToHost));
            std::vector<unsigned> wc(256, 0); for (unsigned w = 0; w < PW; ++w) if (ws[w] < 256) ++wc[ws[w]];
            r.smmin = r.wmin = 1u << 30; r.smmax = r.wmax = 0; for (unsigned s = 0; s < nsm; ++s) { r.smmin = std::min(r.smmin, sh[s]); r.smmax = std::max(r.smmax, sh[s]); r.wmin = std::min(r.wmin, wc[s]); r.wmax = std::max(r.wmax, wc[s]); }
        }
        if (verify) {
            std::vector<RefOut> o(n); CK(cudaMemcpy(o.data(), d_out, (size_t)n * sizeof(RefOut), cudaMemcpyDeviceToHost)); unsigned long bad = 0, tbad = 0;
            for (unsigned i = 0; i < n; ++i) { const RefOut &x = o[i], &y = ref[src[a.layout][i]]; const unsigned gp = (x.hit && a.use_tl) ? tri_sorted[toff[tlv[a.layout][i]] + x.prim] : x.prim;
                if (x.hit != y.hit || gp != y.prim) { ++bad; if (getenv("T1_DEBUG") && bad <= 12) { unsigned q = 0; while (q + 1 < nq && qend[q] <= i) ++q; fprintf(stderr, "BAD %s i %u (lane %u) hit 0x%x queue %u off %u | pad prev %x next %x | lane0 pad %x\n", a.name.c_str(), i, i & 31, x.hit, q, i - qbeg[q], i ? o[i-1].pad : 0u, i + 1 < n ? o[i+1].pad : 0u, o[i & ~31u].pad); } } else if (x.t != y.t) ++tbad; }
            r.bad += bad; r.tbad += tbad;
        }
        return ms;
    };
    FILE *jf = fopen(outjson, "wx"); if (!jf) { fprintf(stderr, "refusing to overwrite or cannot write %s\n", outjson); return 2; }
    fprintf(jf, "{\"set\": \"%s\", \"rays\": %u, \"tris\": %u, \"K\": %u, \"treelets\": %u, \"sm\": %u, \"tpc_queues\": %u, \"wps\": %u, \"rounds\": %d, \"full_gas_bytes\": %zu, \"treelet_gas_total\": %zu, \"treelet_gas_median\": %zu,\n \"queues\": [", set.c_str(), n, N, K, C, nsm, nq, WPS, ROUNDS, full.bytes, tl.total, [&] { auto b = tl.bytes; std::sort(b.begin(), b.end()); return b[b.size() / 2]; }());
    for (unsigned q = 0; q < nq; ++q) fprintf(jf, "%s[%u,%u]", q ? "," : "", qbeg[q], qend[q]);
    fprintf(jf, "],\n \"arms\": [\n");
    if (mode == "profile") {
        for (size_t j = 0; j < arms.size(); ++j) { launch(arms[j], res[j], false, true); printf("PROFILE %zu %s warm\n", j, arms[j].name.c_str()); launch(arms[j], res[j], true, false); printf("PROFILE %zu %s measured ms %.3f\n", j, arms[j].name.c_str(), res[j].ms.back()); fflush(stdout); }
    } else {
        for (size_t j = 0; j < arms.size(); ++j) launch(arms[j], res[j], false, true);        // untimed first launch of every arm, checked
        for (int r = 0; r < ROUNDS; ++r) for (size_t k = 0; k < arms.size(); ++k) { const size_t j = (k + r) % arms.size(); launch(arms[j], res[j], true, r == ROUNDS - 1); }   // rotated order, last round checked again
    }
    for (size_t j = 0; j < arms.size(); ++j) {
        Res &r = res[j]; std::vector<double> m = r.ms; std::sort(m.begin(), m.end());
        fprintf(jf, "  {\"arm\": \"%s\", \"ms\": [", arms[j].name.c_str()); for (size_t i = 0; i < r.ms.size(); ++i) fprintf(jf, "%s%.4f", i ? "," : "", r.ms[i]);
        fprintf(jf, "], \"clk_mhz\": ["); for (size_t i = 0; i < r.clk.size(); ++i) fprintf(jf, "%s%u", i ? "," : "", r.clk[i]);
        fprintf(jf, "], \"median_ms\": %.4f, \"min_ms\": %.4f, \"bad\": %lu, \"tbad\": %lu", m.empty() ? 0.0 : m[m.size() / 2], m.empty() ? 0.0 : m[0], r.bad, r.tbad);
        if (arms[j].persistent) fprintf(jf, ", \"own_chunks\": %u, \"stolen_chunks\": %u, \"lane_mismatch\": %u, \"chunks\": %u, \"sm_chunks_min\": %u, \"sm_chunks_max\": %u, \"warps_per_sm_min\": %u, \"warps_per_sm_max\": %u", r.stats[0], r.stats[1], r.stats[2], r.stats[3], r.smmin, r.smmax, r.wmin, r.wmax);
        fprintf(jf, "}%s\n", j + 1 < arms.size() ? "," : "");
    }
    fprintf(jf, " ]}\n"); fclose(jf); return 0;
}
