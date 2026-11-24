// Treelet_2: end to end treelet queue pipelines with grouping cost, Morton baselines, a treelet box inspector, real scenes. OptiX only (no TTU code anywhere).
//   treelet2 tlsize  PTX WORK SCENE SEED PART                      SCENE soup:N | file:PATH.tris ; PART grid:K | chunk:T ; prints treelet sizes
//   treelet2 prep    PTX WORK SCENE SEED PART TAG KIND NRAYS       KIND rand | b3 | cam3:px,py,pz,tx,ty,tz,fovdeg ; writes WORK/TAG_rays.bin and TAG_ref.bin (hit rays, multiple of 32, natural order)
//   treelet2 run     PTX WORK SCENE SEED PART TAG WPS ROUNDS OUT.json ARM...
// Presorted arms (no grouping cost, as Treelet_1): nat_R grp_R grp_T pg_T pp_T. End to end arms (everything on the GPU inside the timed span: keys, sort, gather, launches, scatter of the results to the natural order):
//   e_mo4 e_mo5  Morton sort of the ray origin (4 or 5 bits per axis), root start, normal launch
//   e_ogR e_ogT e_opT  oracle treelet key (free), sorted, root start grouped / treelet start grouped / treelet start pinned (persistent, queue per TPC)
//   e_igT e_ipT  inspector: nearest treelet box along the ray (custom primitive GAS of the treelet boxes) as the key, sorted, treelet start grouped / pinned,
//                then a verification trace (any OTHER box with entry distance up to the treelet hit, or the whole ray if it missed) and a root start trace in the full GAS for the rays that fail it
// OUT.json is opened with mode x (never overwritten).
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <nvml.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
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
#include "treelet2_params.h"

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

static std::vector<Tri> gen_scene(unsigned n_tri, unsigned seed)   // same generator as setup3.cu soup:N and morton.cu (edge 3 / cbrt(N)); seed 1234 reproduces Treelet_1
{
    const float e = 3.f / std::cbrt((float)n_tri); std::mt19937 g(seed); std::uniform_real_distribution<float> U(-1.f, 1.f);
    std::vector<Tri> v; v.reserve(n_tri);
    for (unsigned i = 0; i < n_tri; ++i) { const float3 c{U(g), U(g), U(g)}; auto pt = [&]() { return float3{c.x + 0.5f * e * U(g), c.y + 0.5f * e * U(g), c.z + 0.5f * e * U(g)}; }; v.push_back({pt(), pt(), pt()}); }
    return v;
}
static unsigned morton3(unsigned x, unsigned y, unsigned z) { unsigned c = 0; for (int b = 0; b < 10; ++b) c |= (((x >> b) & 1u) << (3 * b)) | (((y >> b) & 1u) << (3 * b + 1)) | (((z >> b) & 1u) << (3 * b + 2)); return c; }


static OptixTraversableHandle build_boxes(const std::vector<float> &bx, CUdeviceptr *dbox_out)
{
    const unsigned C = (unsigned)(bx.size() / 6); CUdeviceptr dab; CK(cudaMalloc((void **)&dab, bx.size() * 4)); CK(cudaMemcpy((void *)dab, bx.data(), bx.size() * 4, cudaMemcpyHostToDevice)); *dbox_out = dab;
    OptixBuildInput bi{}; bi.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES; unsigned fl = OPTIX_GEOMETRY_FLAG_NONE;
    bi.customPrimitiveArray.aabbBuffers = &dab; bi.customPrimitiveArray.numPrimitives = C; bi.customPrimitiveArray.strideInBytes = 24; bi.customPrimitiveArray.flags = &fl; bi.customPrimitiveArray.numSbtRecords = 1;
    OptixAccelBuildOptions ao{}; ao.buildFlags = OPTIX_BUILD_FLAG_NONE; ao.operation = OPTIX_BUILD_OPERATION_BUILD; OptixAccelBufferSizes bs{}; OX(optixAccelComputeMemoryUsage(ctx, &ao, &bi, 1, &bs));
    CUdeviceptr tmp, out; CK(cudaMalloc((void **)&tmp, bs.tempSizeInBytes)); CK(cudaMalloc((void **)&out, bs.outputSizeInBytes)); OptixTraversableHandle h;
    OX(optixAccelBuild(ctx, 0, &ao, &bi, 1, tmp, bs.tempSizeInBytes, out, bs.outputSizeInBytes, &h, nullptr, 0)); CK(cudaDeviceSynchronize()); CK(cudaFree((void *)tmp)); return h;
}

// ---- pipeline: raygens direct, persist, coarse; miss; hitgroup triangles; hitgroup boxes ----
struct Pipe { OptixPipeline pipe; OptixShaderBindingTable sbt[3]; CUdeviceptr drec; };
static Pipe make_pipe(const char *ptxfile)
{
    OptixModuleCompileOptions mo{}; OptixPipelineCompileOptions po{};
    po.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS; po.numPayloadValues = 7; po.numAttributeValues = 2;
    po.pipelineLaunchParamsVariableName = "params"; po.usesPrimitiveTypeFlags = (unsigned)(OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE | OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM);
    std::string ptx = read_file(ptxfile); char log[4096]; size_t ls = sizeof log; OptixModule m;
    { OptixResult mr = optixModuleCreate(ctx, &mo, &po, ptx.c_str(), ptx.size(), log, &ls, &m); if (mr != OPTIX_SUCCESS) { fprintf(stderr, "optixModuleCreate %d: %s\n", (int)mr, log); exit(2); } }
    OptixProgramGroupOptions go{}; OptixProgramGroupDesc d[6] = {};
    const char *rg[3] = {"__raygen__direct", "__raygen__persist", "__raygen__coarse"};
    for (int k = 0; k < 3; ++k) { d[k].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN; d[k].raygen.module = m; d[k].raygen.entryFunctionName = rg[k]; }
    d[3].kind = OPTIX_PROGRAM_GROUP_KIND_MISS; d[3].miss.module = m; d[3].miss.entryFunctionName = "__miss__ms";
    d[4].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP; d[4].hitgroup.moduleCH = m; d[4].hitgroup.entryFunctionNameCH = "__closesthit__ch";
    d[5].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP; d[5].hitgroup.moduleCH = m; d[5].hitgroup.entryFunctionNameCH = "__closesthit__box"; d[5].hitgroup.moduleIS = m; d[5].hitgroup.entryFunctionNameIS = "__intersection__box";
    OptixProgramGroup pg[6]; ls = sizeof log; OX(optixProgramGroupCreate(ctx, d, 6, &go, log, &ls, pg));
    OptixPipelineLinkOptions lo{}; lo.maxTraceDepth = 1; Pipe p; ls = sizeof log; OX(optixPipelineCreate(ctx, &po, &lo, pg, 6, log, &ls, &p.pipe));
    struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) Rec { char h[OPTIX_SBT_RECORD_HEADER_SIZE]; };
    Rec rec[6]; for (int k = 0; k < 6; ++k) OX(optixSbtRecordPackHeader(pg[k], &rec[k]));
    CK(cudaMalloc((void **)&p.drec, sizeof rec)); CK(cudaMemcpy((void *)p.drec, rec, sizeof rec, cudaMemcpyHostToDevice));
    for (int k = 0; k < 3; ++k) { p.sbt[k] = {}; p.sbt[k].raygenRecord = p.drec + k * sizeof(Rec); p.sbt[k].missRecordBase = p.drec + 3 * sizeof(Rec); p.sbt[k].missRecordStrideInBytes = sizeof(Rec); p.sbt[k].missRecordCount = 1;
        p.sbt[k].hitgroupRecordBase = p.drec + 4 * sizeof(Rec); p.sbt[k].hitgroupRecordStrideInBytes = sizeof(Rec); p.sbt[k].hitgroupRecordCount = 2; }
    return p;
}

static CUdeviceptr to_dev(const Params &p) { CUdeviceptr d; CK(cudaMalloc((void **)&d, sizeof p)); CK(cudaMemcpy((void *)d, &p, sizeof p, cudaMemcpyHostToDevice)); return d; }

static std::vector<RefOut> trace_full(Pipe &pp, OptixTraversableHandle h, const std::vector<Ray8> &rays)
{
    Params prm{}; prm.handle = h; prm.use_tl = 0; unsigned n = (unsigned)rays.size(); CUdeviceptr dn; CK(cudaMalloc((void **)&dn, 4)); CK(cudaMemcpy((void *)dn, &n, 4, cudaMemcpyHostToDevice)); prm.n_dev = (const unsigned *)dn;
    CK(cudaMalloc((void **)&prm.rays, rays.size() * sizeof(Ray8))); CK(cudaMemcpy((void *)prm.rays, rays.data(), rays.size() * sizeof(Ray8), cudaMemcpyHostToDevice)); CK(cudaMalloc((void **)&prm.out, rays.size() * sizeof(RefOut)));
    CUdeviceptr dprm = to_dev(prm); OX(optixLaunch(pp.pipe, 0, dprm, sizeof prm, &pp.sbt[0], (rays.size() + 31) / 32 * 32, 1, 1)); CK(cudaDeviceSynchronize());
    std::vector<RefOut> o(rays.size()); CK(cudaMemcpy(o.data(), prm.out, o.size() * sizeof(RefOut), cudaMemcpyDeviceToHost));
    CK(cudaFree((void *)prm.rays)); CK(cudaFree(prm.out)); CK(cudaFree((void *)dprm)); CK(cudaFree((void *)dn)); return o;
}

static Ray8 bounce_ray(const Tri &t, const Ray8 &r, float tt, std::mt19937 &g, float eps)
{
    std::uniform_real_distribution<float> U(0.f, 1.f);
    float ux = t.b.x - t.a.x, uy = t.b.y - t.a.y, uz = t.b.z - t.a.z, vx = t.c.x - t.a.x, vy = t.c.y - t.a.y, vz = t.c.z - t.a.z;
    float nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx; const float il = 1.f / std::sqrt(nx * nx + ny * ny + nz * nz + 1e-30f); nx *= il; ny *= il; nz *= il;
    if (nx * r.d[0] + ny * r.d[1] + nz * r.d[2] > 0.f) { nx = -nx; ny = -ny; nz = -nz; }
    const float u1 = U(g), u2 = U(g), rr = std::sqrt(u1), ph = 6.2831853f * u2, x = rr * std::cos(ph), y = rr * std::sin(ph), z = std::sqrt(std::max(0.f, 1.f - u1));
    const float ax = std::fabs(nx) > 0.9f ? 0.f : 1.f, ay = std::fabs(nx) > 0.9f ? 1.f : 0.f; float tx = ay * nz, ty = -ax * nz, tz = ax * ny - ay * nx; const float it = 1.f / std::sqrt(tx * tx + ty * ty + tz * tz + 1e-30f); tx *= it; ty *= it; tz *= it;
    const float bx = ny * tz - nz * ty, by = nz * tx - nx * tz, bz = nx * ty - ny * tx;
    Ray8 o; o.o[0] = r.o[0] + tt * r.d[0] + eps * nx; o.o[1] = r.o[1] + tt * r.d[1] + eps * ny; o.o[2] = r.o[2] + tt * r.d[2] + eps * nz;
    o.tmin = 0.f; o.d[0] = x * tx + y * bx + z * nx; o.d[1] = x * ty + y * by + z * ny; o.d[2] = x * tz + y * bz + z * nz; o.tmax = 1e16f; return o;
}

// ---- device helper kernels (outside OptiX) ----
__global__ void k_key_morton(const Ray8 *r, unsigned *key, unsigned n, float3 lo, float3 inv, int bits)
{
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return; const Ray8 x = r[i]; unsigned q[3]; const float o[3] = {x.o[0], x.o[1], x.o[2]}, l[3] = {lo.x, lo.y, lo.z}, iv[3] = {inv.x, inv.y, inv.z};
    for (int a = 0; a < 3; ++a) { int k = (int)((o[a] - l[a]) * iv[a] * (float)(1 << bits)); q[a] = (unsigned)(k < 0 ? 0 : (k > (1 << bits) - 1 ? (1 << bits) - 1 : k)); }
    unsigned c = 0; for (int b = 0; b < bits; ++b) for (int a = 0; a < 3; ++a) c |= ((q[a] >> b) & 1u) << (3 * b + a); key[i] = c;
}
__global__ void k_key_hit(const RefOut *c, unsigned *key, unsigned n, unsigned C) { const unsigned i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) key[i] = c[i].hit ? c[i].prim : C; }
__global__ void k_gather(const Ray8 *src, const unsigned *perm, Ray8 *dst, unsigned n) { const unsigned i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) dst[i] = src[perm[i]]; }
__global__ void k_cuts(const unsigned *key, unsigned n, unsigned nq, unsigned *qend, unsigned *head)
{
    const unsigned q = threadIdx.x; if (q >= nq) return;
    auto cut = [&](unsigned k) { if (k == 0) return 0u; if (k >= nq) return n; const unsigned i = (unsigned)((unsigned long long)k * n / nq); if (i == 0) return 0u;
        const unsigned v = key[i - 1]; unsigned lo = i, hi = n; while (lo < hi) { const unsigned mid = (lo + hi) / 2; if (key[mid] > v) hi = mid; else lo = mid + 1; } return lo; };   // first index at or after i whose key differs from key[i - 1] (keys are sorted)
    head[q] = cut(q); qend[q] = cut(q + 1);
}
__global__ void k_vrays(const Ray8 *rs, const RefOut *o, Ray8 *vr, unsigned n) { const unsigned i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return; Ray8 r = rs[i]; r.tmax = o[i].hit ? o[i].t : 1e16f; vr[i] = r; }
__global__ void k_flag(const RefOut *v, const unsigned *key, unsigned *flag, unsigned n, unsigned C) { const unsigned i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) flag[i] = (v[i].hit && key[i] < C) ? 1u : 0u; }
__global__ void k_fbgather(const Ray8 *rs, const unsigned *flag, const unsigned *rank, Ray8 *fb, unsigned n) { const unsigned i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n && flag[i]) fb[rank[i]] = rs[i]; }
__global__ void k_nfb(const unsigned *flag, const unsigned *rank, unsigned n, unsigned *nfb) { if (threadIdx.x == 0) nfb[0] = rank[n - 1] + flag[n - 1]; }
__global__ void k_final_map(unsigned n, const RefOut *src, const unsigned *perm, const unsigned *tl, unsigned C, const unsigned *toff, const unsigned *tri_sorted, RefOut *dst)
{
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return; RefOut o = src[i];
    if (tl && o.hit && tl[i] < C) o.prim = tri_sorted[toff[tl[i]] + o.prim];
    dst[perm ? perm[i] : i] = o;
}
__global__ void k_final_ins(unsigned n, const RefOut *out_a, const unsigned *tl, const unsigned *flag, const unsigned *rank, const RefOut *out_fb, const unsigned *perm, unsigned C, const unsigned *toff, const unsigned *tri_sorted, RefOut *dst)
{
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return; RefOut o;
    if (flag[i]) o = out_fb[rank[i]]; else { o = out_a[i]; if (o.hit && tl[i] < C) o.prim = tri_sorted[toff[tl[i]] + o.prim]; }
    dst[perm[i]] = o;
}
__global__ void k_keyacc(const unsigned *key, const unsigned *perm, const unsigned *orc, unsigned n, unsigned *cnt) { const unsigned i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n && key[i] == orc[perm[i]]) atomicAdd(cnt, 1u); }

struct Arm { std::string name; int kind; };   // kind: 0 presorted direct, 1 presorted persistent, 2 end to end
enum { A_PRE_D = 0, A_PRE_P = 1, A_E = 2 };

static void parse_part(const std::string &part, const std::vector<Tri> &tris, std::vector<unsigned> &tl_of_tri, std::vector<unsigned> &tri_sorted, std::vector<unsigned> &toff, unsigned &C)
{
    const unsigned N = (unsigned)tris.size(); tl_of_tri.assign(N, 0); tri_sorted.assign(N, 0);
    if (part.rfind("grid:", 0) == 0) {   // soup in [-1,1]^3: K^3 cells by triangle centroid, treelet id = Morton rank of the cell (as Treelet_1)
        const unsigned K = (unsigned)atoi(part.c_str() + 5); C = K * K * K; std::vector<unsigned> cell_of(N), cellcount(C, 0), code(C), rank_of_cell(C);
        for (unsigned i = 0; i < N; ++i) { const Tri &t = tris[i]; const float cx = (t.a.x + t.b.x + t.c.x) / 3.f, cy = (t.a.y + t.b.y + t.c.y) / 3.f, cz = (t.a.z + t.b.z + t.c.z) / 3.f;
            auto q = [&](float v) { int k = (int)std::floor((v + 1.f) * 0.5f * K); return (unsigned)std::min<int>(std::max<int>(k, 0), K - 1); }; cell_of[i] = (q(cz) * K + q(cy)) * K + q(cx); ++cellcount[cell_of[i]]; }
        std::vector<unsigned> ord(C); std::iota(ord.begin(), ord.end(), 0); for (unsigned c = 0; c < C; ++c) code[c] = morton3(c % K, (c / K) % K, c / (K * K));
        std::stable_sort(ord.begin(), ord.end(), [&](unsigned a, unsigned b) { return code[a] < code[b]; }); for (unsigned r = 0; r < C; ++r) rank_of_cell[ord[r]] = r;
        toff.assign(C + 1, 0); for (unsigned r = 0; r < C; ++r) toff[r + 1] = toff[r] + cellcount[ord[r]];
        std::vector<unsigned> fillp(toff.begin(), toff.end() - 1); for (unsigned i = 0; i < N; ++i) { const unsigned t = rank_of_cell[cell_of[i]]; tl_of_tri[i] = t; tri_sorted[fillp[t]++] = i; }
    } else if (part.rfind("chunk:", 0) == 0) {   // any scene: triangles in Morton order of the centroid (10 bits per axis over the bounding box), T consecutive triangles per treelet
        const unsigned T = (unsigned)atoi(part.c_str() + 6); C = (N + T - 1) / T; float lo[3] = {1e30f, 1e30f, 1e30f}, hi[3] = {-1e30f, -1e30f, -1e30f};
        std::vector<float> cen(3ull * N); for (unsigned i = 0; i < N; ++i) { const Tri &t = tris[i]; const float c[3] = {(t.a.x + t.b.x + t.c.x) / 3.f, (t.a.y + t.b.y + t.c.y) / 3.f, (t.a.z + t.b.z + t.c.z) / 3.f}; for (int a = 0; a < 3; ++a) { cen[3ull * i + a] = c[a]; lo[a] = std::min(lo[a], c[a]); hi[a] = std::max(hi[a], c[a]); } }
        std::vector<unsigned> code(N); for (unsigned i = 0; i < N; ++i) { unsigned q[3]; for (int a = 0; a < 3; ++a) { int k = (int)((cen[3ull * i + a] - lo[a]) / std::max(hi[a] - lo[a], 1e-20f) * 1024.f); q[a] = (unsigned)std::min(std::max(k, 0), 1023); } code[i] = morton3(q[0], q[1], q[2]); }
        std::iota(tri_sorted.begin(), tri_sorted.end(), 0u); std::stable_sort(tri_sorted.begin(), tri_sorted.end(), [&](unsigned a, unsigned b) { return code[a] < code[b]; });
        toff.assign(C + 1, 0); for (unsigned t = 0; t <= C; ++t) toff[t] = std::min(N, t * T); for (unsigned r = 0; r < N; ++r) tl_of_tri[tri_sorted[r]] = r / T;
    } else { fprintf(stderr, "bad PART %s\n", part.c_str()); exit(2); }
}

int main(int argc, char **argv)
{
    if (argc < 7) { fprintf(stderr, "usage: see the top of treelet2.cu\n"); return 2; }
    const std::string mode = argv[1], work = argv[3], scene = argv[4]; const unsigned seed = (unsigned)atoi(argv[5]); const std::string part = argv[6];
    CK(cudaFree(0)); OX(optixInit()); OptixDeviceContextOptions co{}; OX(optixDeviceContextCreate(0, &co, &ctx));
    cudaDeviceProp dp; CK(cudaGetDeviceProperties(&dp, 0)); const unsigned nsm = dp.multiProcessorCount, nq = nsm / 2;
    std::vector<Tri> tris;
    if (scene.rfind("soup:", 0) == 0) tris = gen_scene((unsigned)atoi(scene.c_str() + 5), seed); else if (scene.rfind("file:", 0) == 0) tris = read_bin<Tri>(scene.substr(5)); else { fprintf(stderr, "bad SCENE\n"); return 2; }
    const unsigned N = (unsigned)tris.size(); float slo[3] = {1e30f, 1e30f, 1e30f}, shi[3] = {-1e30f, -1e30f, -1e30f};
    for (const Tri &t : tris) for (const float3 &p : {t.a, t.b, t.c}) { const float v[3] = {p.x, p.y, p.z}; for (int a = 0; a < 3; ++a) { slo[a] = std::min(slo[a], v[a]); shi[a] = std::max(shi[a], v[a]); } }
    const float diag = std::sqrt((shi[0] - slo[0]) * (shi[0] - slo[0]) + (shi[1] - slo[1]) * (shi[1] - slo[1]) + (shi[2] - slo[2]) * (shi[2] - slo[2]));
    Pipe pp = make_pipe(argv[2]);
    std::vector<unsigned> tl_of_tri, tri_sorted, toff; unsigned C = 0; parse_part(part, tris, tl_of_tri, tri_sorted, toff, C);
    std::vector<Tri> sv(N); for (unsigned j = 0; j < N; ++j) sv[j] = tris[tri_sorted[j]];
    CUdeviceptr dsv; CK(cudaMalloc((void **)&dsv, (size_t)N * sizeof(Tri))); CK(cudaMemcpy((void *)dsv, sv.data(), (size_t)N * sizeof(Tri), cudaMemcpyHostToDevice));
    Treelets tl = build_treelets(dsv, toff);
    { std::vector<size_t> b = tl.bytes; std::sort(b.begin(), b.end());
      printf("{\"scene\": \"%s\", \"tris\": %u, \"bbox_diag\": %.3f, \"treelets\": %u, \"tris_per_treelet_mean\": %.1f, \"gas_bytes_min\": %zu, \"gas_bytes_median\": %zu, \"gas_bytes_max\": %zu, \"gas_bytes_total\": %zu, \"sm\": %u, \"tpc_queues\": %u}\n",
             scene.c_str(), N, diag, C, (double)N / C, b.front(), b[b.size() / 2], b.back(), tl.total, nsm, nq); fflush(stdout); }
    if (mode == "tlsize") return 0;
    CUdeviceptr dfull; CK(cudaMalloc((void **)&dfull, (size_t)N * sizeof(Tri))); CK(cudaMemcpy((void *)dfull, tris.data(), (size_t)N * sizeof(Tri), cudaMemcpyHostToDevice));
    Gas full = build_one(dfull, N);

    if (mode == "prep") {
        const std::string tag = argv[7], kind = argv[8]; const unsigned S = (unsigned)atoi(argv[9]); const float eps = 3e-5f * diag;
        printf("{\"full_gas_bytes\": %zu}\n", full.bytes); fflush(stdout);
        std::mt19937 rg(seed + 777); std::uniform_real_distribution<float> U(0.f, 1.f); std::normal_distribution<float> Nn(0.f, 1.f); std::vector<Ray8> r0(S);
        if (kind == "rand" || kind == "b3") {
            for (unsigned i = 0; i < S; ++i) { float dx, dy, dz, l; do { dx = Nn(rg); dy = Nn(rg); dz = Nn(rg); l = std::sqrt(dx * dx + dy * dy + dz * dz); } while (l < 1e-6f);
                r0[i] = Ray8{{slo[0] + U(rg) * (shi[0] - slo[0]), slo[1] + U(rg) * (shi[1] - slo[1]), slo[2] + U(rg) * (shi[2] - slo[2])}, 0.f, {dx / l, dy / l, dz / l}, 1e16f}; }
        } else if (kind.rfind("cam3:", 0) == 0) {
            float c[7]; sscanf(kind.c_str() + 5, "%f,%f,%f,%f,%f,%f,%f", &c[0], &c[1], &c[2], &c[3], &c[4], &c[5], &c[6]);
            float fx = c[3] - c[0], fy = c[4] - c[1], fz = c[5] - c[2]; float il = 1.f / std::sqrt(fx * fx + fy * fy + fz * fz); fx *= il; fy *= il; fz *= il;
            float rx = fy * 0.f - fz * 1.f, ry = fz * 0.f - fx * 0.f, rz = fx * 1.f - fy * 0.f; rx = fy * 0.f - fz * 1.f; // right = f x up, up = (0,1,0)
            rx = -fz; ry = 0.f; rz = fx; il = 1.f / std::sqrt(rx * rx + rz * rz); rx *= il; rz *= il;
            const float ux = ry * fz - rz * fy, uy = rz * fx - rx * fz, uz = rx * fy - ry * fx; const unsigned H = (unsigned)std::sqrt(S / 2.0), W = 2 * H; const float th = std::tan(0.5f * c[6] * 3.14159265f / 180.f); r0.resize((size_t)W * H);
            for (unsigned y = 0; y < H; ++y) for (unsigned x = 0; x < W; ++x) { const float px = ((x + 0.5f) / W * 2.f - 1.f) * th * 2.f, py = (1.f - (y + 0.5f) / H * 2.f) * th; float dx = fx + px * rx + py * ux, dy = fy + px * ry + py * uy, dz = fz + px * rz + py * uz;
                const float l = std::sqrt(dx * dx + dy * dy + dz * dz); r0[(size_t)y * W + x] = Ray8{{c[0], c[1], c[2]}, 0.f, {dx / l, dy / l, dz / l}, 1e16f}; }
            printf("{\"camera\": \"%s\", \"width\": %u, \"height\": %u}\n", kind.c_str(), W, H);
        } else { fprintf(stderr, "bad KIND\n"); return 2; }
        auto save = [&](const std::string &t, const std::vector<Ray8> &r) {
            std::vector<RefOut> o = trace_full(pp, full.h, r); std::vector<Ray8> kr; std::vector<RefOut> ko; for (size_t i = 0; i < r.size(); ++i) if (o[i].hit) { kr.push_back(r[i]); ko.push_back(o[i]); }
            kr.resize(kr.size() / 32 * 32); ko.resize(kr.size()); write_bin(work + "/" + t + "_rays.bin", kr); write_bin(work + "/" + t + "_ref.bin", ko);
            printf("{\"set\": \"%s\", \"generated\": %zu, \"hits_kept\": %zu}\n", t.c_str(), r.size(), kr.size()); fflush(stdout); };
        if (kind == "rand") { save(tag, r0); return 0; }
        std::mt19937 rb(seed + 4242); std::vector<Ray8> cur = r0; int depth = kind == "b3" ? 3 : 3;
        for (int d = 0; d < depth; ++d) { std::vector<RefOut> o = trace_full(pp, full.h, cur); std::vector<Ray8> nx; for (size_t i = 0; i < cur.size(); ++i) if (o[i].hit) nx.push_back(bounce_ray(tris[o[i].prim], cur[i], o[i].t, rb, eps));
            printf("{\"bounce\": %d, \"alive_in\": %zu, \"alive_out\": %zu}\n", d + 1, cur.size(), nx.size()); cur.swap(nx); }
        save(tag, cur); return 0;
    }

    // ================= run =================
    const std::string tag = argv[7]; const unsigned WPS = (unsigned)atoi(argv[8]); const int ROUNDS = atoi(argv[9]); const char *outjson = argv[10];
    std::vector<Arm> arms; for (int a = 11; a < argc; ++a) { const std::string s = argv[a]; int k = A_E; if (s == "nat_R" || s == "grp_R" || s == "grp_T") k = A_PRE_D; else if (s == "pg_T" || s == "pp_T") k = A_PRE_P; else if (s.rfind("e_", 0) != 0) { fprintf(stderr, "bad arm %s\n", s.c_str()); return 2; } arms.push_back({s, k}); }
    std::vector<Ray8> rays = read_bin<Ray8>(work + "/" + tag + "_rays.bin"); std::vector<RefOut> ref = read_bin<RefOut>(work + "/" + tag + "_ref.bin"); const unsigned n = (unsigned)rays.size();
    std::vector<unsigned> orc(n); for (unsigned i = 0; i < n; ++i) orc[i] = tl_of_tri[ref[i].prim];
    unsigned kb = 1; while ((1u << kb) < C + 1) ++kb;
    // presorted layouts (as Treelet_1): 0 natural, 1 grouped by treelet
    std::vector<unsigned> src[2], tlv[2]; std::vector<Ray8> ry[2]; src[0].resize(n); std::iota(src[0].begin(), src[0].end(), 0u); src[1] = src[0];
    std::stable_sort(src[1].begin(), src[1].end(), [&](unsigned a, unsigned b) { return orc[a] < orc[b]; });
    for (int l = 0; l < 2; ++l) { ry[l].resize(n); tlv[l].resize(n); for (unsigned i = 0; i < n; ++i) { ry[l][i] = rays[src[l][i]]; tlv[l][i] = orc[src[l][i]]; } }
    std::vector<unsigned> tstart(C + 1, 0); for (unsigned i = 0; i < n; ++i) ++tstart[tlv[1][i] + 1]; for (unsigned t = 0; t < C; ++t) tstart[t + 1] += tstart[t];
    std::vector<unsigned> qbeg(nq), qend(nq);
    for (unsigned q = 0; q < nq; ++q) { auto cut = [&](unsigned k) { if (k == 0) return 0u; if (k == nq) return n; const unsigned tgt = (unsigned)((unsigned long long)k * n / nq); return *std::lower_bound(tstart.begin(), tstart.end(), tgt); }; qbeg[q] = cut(q); qend[q] = cut(q + 1); }
    // treelet boxes (tight, plus a margin) and the box GAS
    std::vector<float> bx(6ull * C); { const float mg = 1e-4f * diag; for (unsigned t = 0; t < C; ++t) { float lo[3] = {1e30f, 1e30f, 1e30f}, hi[3] = {-1e30f, -1e30f, -1e30f};
        for (unsigned j = toff[t]; j < toff[t + 1]; ++j) { const Tri &q = sv[j]; for (const float3 &p : {q.a, q.b, q.c}) { const float v[3] = {p.x, p.y, p.z}; for (int a = 0; a < 3; ++a) { lo[a] = std::min(lo[a], v[a]); hi[a] = std::max(hi[a], v[a]); } } }
        for (int a = 0; a < 3; ++a) { bx[6ull * t + a] = lo[a] - mg; bx[6ull * t + 3 + a] = hi[a] + mg; } } }
    CUdeviceptr dbox; OptixTraversableHandle hbox = build_boxes(bx, &dbox);
    // Morton bounds of the ray origins
    float olo[3] = {1e30f, 1e30f, 1e30f}, ohi[3] = {-1e30f, -1e30f, -1e30f}; for (const Ray8 &r : rays) for (int a = 0; a < 3; ++a) { olo[a] = std::min(olo[a], r.o[a]); ohi[a] = std::max(ohi[a], r.o[a]); }
    const float3 mlo = make_float3(olo[0], olo[1], olo[2]), minv = make_float3(1.f / std::max(ohi[0] - olo[0], 1e-20f), 1.f / std::max(ohi[1] - olo[1], 1e-20f), 1.f / std::max(ohi[2] - olo[2], 1e-20f));
    // device buffers
    auto A = [&](size_t bytes) { CUdeviceptr d; CK(cudaMalloc((void **)&d, bytes)); return d; };
    CUdeviceptr d_ry[2], d_tl[2]; for (int l = 0; l < 2; ++l) { d_ry[l] = A(n * sizeof(Ray8)); CK(cudaMemcpy((void *)d_ry[l], ry[l].data(), n * sizeof(Ray8), cudaMemcpyHostToDevice)); d_tl[l] = A(n * 4); CK(cudaMemcpy((void *)d_tl[l], tlv[l].data(), n * 4, cudaMemcpyHostToDevice)); }
    std::vector<OptixTraversableHandle> hh = tl.h; hh.push_back(full.h); CUdeviceptr d_h = A(8ull * (C + 1)); CK(cudaMemcpy((void *)d_h, hh.data(), 8ull * (C + 1), cudaMemcpyHostToDevice));
    CUdeviceptr d_rn = d_ry[0], d_rs = A(n * sizeof(Ray8)), d_vr = A(n * sizeof(Ray8)), d_fbr = A(n * sizeof(Ray8)); RefOut *d_out = (RefOut *)A(n * sizeof(RefOut));
    CUdeviceptr d_oa = A(n * sizeof(RefOut)), d_on = A(n * sizeof(RefOut)), d_oc = A(n * sizeof(RefOut)), d_ov = A(n * sizeof(RefOut)), d_ofb = A(n * sizeof(RefOut));
    CUdeviceptr d_kin = A(n * 4), d_kout = A(n * 4), d_pin = A(n * 4), d_perm = A(n * 4), d_orc = A(n * 4), d_none = A(n * 4), d_flag = A(n * 4), d_rank = A(n * 4), d_n = A(4), d_nfb = A(4), d_cnt = A(4);
    { std::vector<unsigned> iota(n), none(n, 0xffffffffu); std::iota(iota.begin(), iota.end(), 0u); CK(cudaMemcpy((void *)d_pin, iota.data(), n * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy((void *)d_none, none.data(), n * 4, cudaMemcpyHostToDevice));
      CK(cudaMemcpy((void *)d_orc, orc.data(), n * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy((void *)d_n, &n, 4, cudaMemcpyHostToDevice)); }
    CUdeviceptr d_toff = A(4ull * (C + 1)), d_tris = A(4ull * N); CK(cudaMemcpy((void *)d_toff, toff.data(), 4ull * (C + 1), cudaMemcpyHostToDevice)); CK(cudaMemcpy((void *)d_tris, tri_sorted.data(), 4ull * N, cudaMemcpyHostToDevice));
    CUdeviceptr d_qend_p = A(4ull * nq), d_head_p = A(4ull * nq), d_qend1 = A(4), d_qend_g = A(4ull * nq), d_head_g = A(4ull * nq), d_stats = A(16), d_smhist = A(1024); const unsigned maxwarps = 4096; CUdeviceptr d_warpsm = A(4ull * maxwarps);
    CK(cudaMemcpy((void *)d_qend_p, qend.data(), 4ull * nq, cudaMemcpyHostToDevice)); CK(cudaMemcpy((void *)d_qend1, &n, 4, cudaMemcpyHostToDevice));
    size_t sort_bytes = 0, scan_bytes = 0; cub::DeviceRadixSort::SortPairs(nullptr, sort_bytes, (unsigned *)d_kin, (unsigned *)d_kout, (unsigned *)d_pin, (unsigned *)d_perm, n, 0, 24); cub::DeviceScan::ExclusiveSum(nullptr, scan_bytes, (unsigned *)d_flag, (unsigned *)d_rank, n);
    CUdeviceptr d_stmp = A(std::max(sort_bytes, scan_bytes));
    const unsigned PW = nsm * WPS; if (PW > maxwarps) { fprintf(stderr, "WPS too large\n"); return 2; }
    // parameter blocks (device resident, so the end to end sequences have no host synchronisation)
    auto base = [&]() { Params p{}; p.handle = full.h; p.tl_handle = (const OptixTraversableHandle *)d_h; p.n_dev = (const unsigned *)d_n; p.coarse = hbox; p.boxes = (const float *)dbox; p.excl = (const unsigned *)d_none; p.stats = (unsigned *)d_stats; p.smhist = (unsigned *)d_smhist; p.warp_sm = (unsigned *)d_warpsm; p.qend = (const unsigned *)d_qend1; p.head = (unsigned *)d_head_g; p.nq = 1; return p; };
    Params p_root = base(); p_root.rays = (const Ray8 *)d_rs; p_root.out = (RefOut *)d_oa; CUdeviceptr P_root = to_dev(p_root);
    Params p_t = base(); p_t.rays = (const Ray8 *)d_rs; p_t.out = (RefOut *)d_oa; p_t.tl = (const unsigned *)d_kout; p_t.use_tl = 1; CUdeviceptr P_t = to_dev(p_t);
    Params p_tp = p_t; p_tp.pinned = 1; p_tp.nq = nq; p_tp.qend = (const unsigned *)d_qend_g; p_tp.head = (unsigned *)d_head_g; CUdeviceptr P_tp = to_dev(p_tp);
    Params p_c1 = base(); p_c1.rays = (const Ray8 *)d_rn; p_c1.out = (RefOut *)d_oc; CUdeviceptr P_c1 = to_dev(p_c1);
    Params p_v = base(); p_v.rays = (const Ray8 *)d_vr; p_v.out = (RefOut *)d_ov; p_v.excl = (const unsigned *)d_kout; CUdeviceptr P_v = to_dev(p_v);
    Params p_fb = base(); p_fb.rays = (const Ray8 *)d_fbr; p_fb.out = (RefOut *)d_ofb; p_fb.n_dev = (const unsigned *)d_nfb; CUdeviceptr P_fb = to_dev(p_fb);
    const unsigned grd = (n + 255) / 256, NT = (n + 31) / 32 * 32;
    cudaEvent_t ev[8]; for (int k = 0; k < 8; ++k) CK(cudaEventCreate(&ev[k]));
    nvmlDevice_t nd; char bus[32]; CK(cudaDeviceGetPCIBusId(bus, sizeof bus, 0)); bool nv_ok = nvmlInit_v2() == NVML_SUCCESS && nvmlDeviceGetHandleByPciBusId_v2(bus, &nd) == NVML_SUCCESS;
    struct Res { std::vector<double> ms; std::vector<unsigned> clk; unsigned stats[4]; unsigned long bad = 0, tbad = 0, ties = 0; std::vector<std::vector<double>> st; double flagged = -1, keyacc = -1; unsigned smmin = 0, smmax = 0; };
    std::vector<Res> res(arms.size());
    auto sortk = [&](CUdeviceptr kin, int bits) { size_t b = sort_bytes; cub::DeviceRadixSort::SortPairs((void *)d_stmp, b, (unsigned *)kin, (unsigned *)d_kout, (unsigned *)d_pin, (unsigned *)d_perm, n, 0, bits); };
    auto launch = [&](CUdeviceptr P, int sbt, unsigned threads) { OX(optixLaunch(pp.pipe, 0, P, sizeof(Params), &pp.sbt[sbt], threads, 1, 1)); };
    auto reset_stats = [&]() { CK(cudaMemsetAsync((void *)d_stats, 0, 16)); CK(cudaMemsetAsync((void *)d_smhist, 0, 1024)); CK(cudaMemsetAsync((void *)d_warpsm, 0xff, 4ull * maxwarps)); };
    // ---- end to end sequences (all asynchronous on stream 0) ----
    auto exec_e = [&](const std::string &nm, bool stages) {
        const bool ins = nm == "e_igT" || nm == "e_ipT", pin = nm == "e_opT" || nm == "e_ipT"; int k = 0; auto mark = [&]() { if (stages) CK(cudaEventRecord(ev[k++])); };
        mark();
        if (nm == "e_mo4" || nm == "e_mo5") {
            const int b = nm == "e_mo4" ? 4 : 5; k_key_morton<<<grd, 256>>>((const Ray8 *)d_rn, (unsigned *)d_kin, n, mlo, minv, b); sortk(d_kin, 3 * b); k_gather<<<grd, 256>>>((const Ray8 *)d_rn, (const unsigned *)d_perm, (Ray8 *)d_rs, n);
            launch(P_root, 0, NT); k_final_map<<<grd, 256>>>(n, (const RefOut *)d_oa, (const unsigned *)d_perm, nullptr, C, (const unsigned *)d_toff, (const unsigned *)d_tris, (RefOut *)d_on);
        } else if (nm == "e_ogR") {
            sortk(d_orc, kb); k_gather<<<grd, 256>>>((const Ray8 *)d_rn, (const unsigned *)d_perm, (Ray8 *)d_rs, n); launch(P_root, 0, NT);
            k_final_map<<<grd, 256>>>(n, (const RefOut *)d_oa, (const unsigned *)d_perm, nullptr, C, (const unsigned *)d_toff, (const unsigned *)d_tris, (RefOut *)d_on);
        } else if (nm == "e_ogT" || nm == "e_opT") {
            sortk(d_orc, kb); k_gather<<<grd, 256>>>((const Ray8 *)d_rn, (const unsigned *)d_perm, (Ray8 *)d_rs, n);
            if (pin) { k_cuts<<<1, 128>>>((const unsigned *)d_kout, n, nq, (unsigned *)d_qend_g, (unsigned *)d_head_g); reset_stats(); launch(P_tp, 1, PW * 32); } else launch(P_t, 0, NT);
            k_final_map<<<grd, 256>>>(n, (const RefOut *)d_oa, (const unsigned *)d_perm, (const unsigned *)d_kout, C, (const unsigned *)d_toff, (const unsigned *)d_tris, (RefOut *)d_on);
        } else if (ins) {
            launch(P_c1, 2, NT); mark();                                                                                       // 1 inspector: nearest treelet box
            k_key_hit<<<grd, 256>>>((const RefOut *)d_oc, (unsigned *)d_kin, n, C); sortk(d_kin, kb); k_gather<<<grd, 256>>>((const Ray8 *)d_rn, (const unsigned *)d_perm, (Ray8 *)d_rs, n);
            if (pin) k_cuts<<<1, 128>>>((const unsigned *)d_kout, n, nq, (unsigned *)d_qend_g, (unsigned *)d_head_g); mark();   // 2 key, sort, gather, queue cuts
            if (pin) { reset_stats(); launch(P_tp, 1, PW * 32); } else launch(P_t, 0, NT); mark();                            // 3 treelet start trace
            k_vrays<<<grd, 256>>>((const Ray8 *)d_rs, (const RefOut *)d_oa, (Ray8 *)d_vr, n); launch(P_v, 2, NT);
            k_flag<<<grd, 256>>>((const RefOut *)d_ov, (const unsigned *)d_kout, (unsigned *)d_flag, n, C); mark();            // 4 verification trace
            { size_t b = scan_bytes; cub::DeviceScan::ExclusiveSum((void *)d_stmp, b, (unsigned *)d_flag, (unsigned *)d_rank, n); }
            k_fbgather<<<grd, 256>>>((const Ray8 *)d_rs, (const unsigned *)d_flag, (const unsigned *)d_rank, (Ray8 *)d_fbr, n); k_nfb<<<1, 32>>>((const unsigned *)d_flag, (const unsigned *)d_rank, n, (unsigned *)d_nfb);
            launch(P_fb, 0, NT); mark();                                                                                       // 5 fallback: root start trace of the rays that failed verification
            k_final_ins<<<grd, 256>>>(n, (const RefOut *)d_oa, (const unsigned *)d_kout, (const unsigned *)d_flag, (const unsigned *)d_rank, (const RefOut *)d_ofb, (const unsigned *)d_perm, C, (const unsigned *)d_toff, (const unsigned *)d_tris, (RefOut *)d_on); mark();   // 6 scatter
        } else { fprintf(stderr, "bad e arm\n"); exit(2); }
    };
    auto check_nat = [&](const RefOut *dev, Res &r) {
        std::vector<RefOut> o(n); CK(cudaMemcpy(o.data(), dev, (size_t)n * sizeof(RefOut), cudaMemcpyDeviceToHost)); for (unsigned i = 0; i < n; ++i) { const RefOut &x = o[i], &y = ref[i]; if (x.hit != y.hit) ++r.bad; else if (x.prim != y.prim) { if (std::fabs(x.t - y.t) <= 1e-5f * std::fabs(y.t) + 1e-7f) ++r.ties; else ++r.bad; } else if (x.t != y.t) ++r.tbad; } };
    auto run_arm = [&](size_t j, bool timed, bool verify) {
        const Arm &a = arms[j]; Res &r = res[j]; float ms = 0;
        if (a.kind == A_E) {
            const bool ins = a.name == "e_igT" || a.name == "e_ipT"; CK(cudaMemset((void *)d_on, 0xff, (size_t)n * sizeof(RefOut))); CK(cudaDeviceSynchronize());
            CK(cudaEventRecord(ev[7])); exec_e(a.name, ins); cudaEvent_t e1; CK(cudaEventCreate(&e1)); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); CK(cudaEventElapsedTime(&ms, ev[7], e1));
            if (ins) { if (r.st.empty()) r.st.resize(6); for (int s = 0; s < 6; ++s) { float t; CK(cudaEventElapsedTime(&t, ev[s], ev[s + 1])); r.st[s].push_back(t); } }
            CK(cudaEventDestroy(e1));
            if (verify) {
                check_nat((const RefOut *)d_on, r);
                if (ins) { unsigned nf; CK(cudaMemcpy(&nf, (void *)d_nfb, 4, cudaMemcpyDeviceToHost)); r.flagged = (double)nf / n; CK(cudaMemset((void *)d_cnt, 0, 4)); k_keyacc<<<grd, 256>>>((const unsigned *)d_kout, (const unsigned *)d_perm, (const unsigned *)d_orc, n, (unsigned *)d_cnt); unsigned c; CK(cudaMemcpy(&c, (void *)d_cnt, 4, cudaMemcpyDeviceToHost)); r.keyacc = (double)c / n; }
                if (a.name == "e_opT" || a.name == "e_ipT") { CK(cudaMemcpy(r.stats, (void *)d_stats, 16, cudaMemcpyDeviceToHost)); }
            }
        } else {   // presorted, as Treelet_1 (rays and keys already in the arm's order, timing = the launch)
            const int layout = a.name == "nat_R" ? 0 : 1; const bool tl_start = a.name.back() == 'T', persistent = a.kind == A_PRE_P, pinned = a.name == "pp_T";
            Params p = base(); p.rays = (const Ray8 *)d_ry[layout]; p.out = d_out; p.tl = (const unsigned *)d_tl[layout]; p.use_tl = tl_start; p.pinned = pinned; p.nq = pinned ? nq : 1; p.qend = (const unsigned *)(pinned ? d_qend_p : d_qend1);
            std::vector<unsigned> hb(nq, 0); if (pinned) hb = qbeg; CK(cudaMemcpy((void *)d_head_p, hb.data(), 4ull * nq, cudaMemcpyHostToDevice)); p.head = (unsigned *)d_head_p;
            CK(cudaMemset((void *)d_stats, 0, 16)); CK(cudaMemset((void *)d_smhist, 0, 1024)); CK(cudaMemset((void *)d_warpsm, 0xff, 4ull * maxwarps)); CK(cudaMemset(d_out, 0xff, (size_t)n * sizeof(RefOut)));
            CUdeviceptr dp2 = to_dev(p); CK(cudaDeviceSynchronize()); CK(cudaEventRecord(ev[0])); if (persistent) launch(dp2, 1, PW * 32); else launch(dp2, 0, NT); CK(cudaEventRecord(ev[1])); CK(cudaEventSynchronize(ev[1])); CK(cudaEventElapsedTime(&ms, ev[0], ev[1])); CK(cudaFree((void *)dp2));
            if (persistent) { CK(cudaMemcpy(r.stats, (void *)d_stats, 16, cudaMemcpyDeviceToHost)); }
            if (verify) { std::vector<RefOut> o(n); CK(cudaMemcpy(o.data(), d_out, (size_t)n * sizeof(RefOut), cudaMemcpyDeviceToHost));
                for (unsigned i = 0; i < n; ++i) { const RefOut &x = o[i], &y = ref[src[layout][i]]; const unsigned gp = (x.hit && tl_start) ? tri_sorted[toff[tlv[layout][i]] + x.prim] : x.prim; if (x.hit != y.hit) ++r.bad; else if (gp != y.prim) { if (std::fabs(x.t - y.t) <= 1e-5f * std::fabs(y.t) + 1e-7f) ++r.ties; else ++r.bad; } else if (x.t != y.t) ++r.tbad; } }
        }
        unsigned c = 0; if (nv_ok) nvmlDeviceGetClockInfo(nd, NVML_CLOCK_SM, &c); if (timed) { r.ms.push_back(ms); r.clk.push_back(c); }
    };
    FILE *jf = fopen(outjson, "wx"); if (!jf) { fprintf(stderr, "refusing to overwrite or cannot write %s\n", outjson); return 2; }
    fprintf(jf, "{\"scene\": \"%s\", \"seed\": %u, \"part\": \"%s\", \"tag\": \"%s\", \"rays\": %u, \"tris\": %u, \"treelets\": %u, \"sm\": %u, \"tpc_queues\": %u, \"wps\": %u, \"rounds\": %d, \"full_gas_bytes\": %zu, \"treelet_gas_total\": %zu, \"treelet_gas_median\": %zu,\n \"arms\": [\n",
            scene.c_str(), seed, part.c_str(), tag.c_str(), n, N, C, nsm, nq, WPS, ROUNDS, full.bytes, tl.total, [&] { auto b = tl.bytes; std::sort(b.begin(), b.end()); return b[b.size() / 2]; }());
    for (size_t j = 0; j < arms.size(); ++j) run_arm(j, false, true);                                    // untimed first launch of every arm, checked
    for (int r = 0; r < ROUNDS; ++r) for (size_t k = 0; k < arms.size(); ++k) { const size_t j = (k + r) % arms.size(); run_arm(j, true, r == ROUNDS - 1); }
    { size_t touched = 0; std::vector<char> seen(C, 0); for (unsigned i = 0; i < n; ++i) seen[orc[i]] = 1; unsigned tc = 0; for (unsigned t = 0; t < C; ++t) if (seen[t]) { touched += tl.bytes[t]; ++tc; }
      fprintf(jf, "  {\"info\": \"oracle touched treelets\", \"treelets_touched\": %u, \"bytes_touched\": %zu},\n", tc, touched); }
    for (size_t j = 0; j < arms.size(); ++j) {
        Res &r = res[j]; std::vector<double> m = r.ms; std::sort(m.begin(), m.end()); const bool pers = arms[j].kind == A_PRE_P || arms[j].name == "e_opT" || arms[j].name == "e_ipT";
        fprintf(jf, "  {\"arm\": \"%s\", \"ms\": [", arms[j].name.c_str()); for (size_t i = 0; i < r.ms.size(); ++i) fprintf(jf, "%s%.4f", i ? "," : "", r.ms[i]);
        fprintf(jf, "], \"clk_mhz\": ["); for (size_t i = 0; i < r.clk.size(); ++i) fprintf(jf, "%s%u", i ? "," : "", r.clk[i]);
        fprintf(jf, "], \"median_ms\": %.4f, \"min_ms\": %.4f, \"bad\": %lu, \"tbad\": %lu, \"ties\": %lu", m.empty() ? 0.0 : m[m.size() / 2], m.empty() ? 0.0 : m[0], r.bad, r.tbad, r.ties);
        if (pers) fprintf(jf, ", \"own_chunks\": %u, \"stolen_chunks\": %u, \"lane_mismatch\": %u, \"chunks\": %u", r.stats[0], r.stats[1], r.stats[2], r.stats[3]);
        if (!r.st.empty()) { fprintf(jf, ", \"stage_median_ms\": ["); for (size_t s = 0; s < r.st.size(); ++s) { std::vector<double> v = r.st[s]; std::sort(v.begin(), v.end()); fprintf(jf, "%s%.4f", s ? "," : "", v[v.size() / 2]); } fprintf(jf, "], \"flagged_frac\": %.4f, \"key_accuracy\": %.4f", r.flagged, r.keyacc); }
        fprintf(jf, "}%s\n", j + 1 < arms.size() ? "," : "");
    }
    fprintf(jf, " ]}\n"); fclose(jf); return 0;
}
