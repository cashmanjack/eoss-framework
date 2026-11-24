// Treelet_3: the most aggressive treelet design we can build with OptiX, measured end to end (no TTU code anywhere).
//   treelet3 tlsize PTXOLD PTXNEW WORK SCENE SEED PART
//   treelet3 prep   PTXOLD PTXNEW WORK SCENE SEED PART TAG KIND NRAYS      (as Treelet_2)
//   treelet3 run    PTXOLD PTXNEW WORK SCENE SEED PART TAG WPS ROUNDS OUT.json ARM...
// ARMS (old pipeline, as Treelet_2): nat_R (natural order, full GAS), e_mo4, e_mo5 (Morton sort of the origin, end to end), pp_T, e_opT (oracle key, pinned persistent reference)
//   new pipeline: ias_nat (natural order through an IAS whose instances are the real treelet GASes), f_pin, f_nopin (the fused design, end to end), f_rep:R (fused first pass, grouping reused R-1 times), mo_rep:R, nat_rep:R (R traces of the baselines)
// f_*: [A1 inspector: IAS of treelet shells, first treelet of every ray + histogram] [scan, counting enqueue into per treelet queues] [persistent warps: drain the TPC's treelets, trace, next treelet by the IAS step, requeue]
// OUT.json is opened with mode x (never overwritten).
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <nvml.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/block/block_scan.cuh>
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
#include <thread>
#include <chrono>
#include "treelet3_params.h"

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


// ---- pipelines ----
struct Pipe { OptixPipeline pipe; OptixShaderBindingTable sbt[8]; CUdeviceptr drec; };
struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) Rec { char h[OPTIX_SBT_RECORD_HEADER_SIZE]; };
static Pipe finish_pipe(OptixPipeline pipe, OptixProgramGroup *pg, int nrg, int imiss, int ihit, int nhit)
{
    Pipe p; p.pipe = pipe; std::vector<Rec> rec(nrg + 1 + nhit); for (size_t k = 0; k < rec.size(); ++k) OX(optixSbtRecordPackHeader(pg[k], &rec[k]));
    CK(cudaMalloc((void **)&p.drec, rec.size() * sizeof(Rec))); CK(cudaMemcpy((void *)p.drec, rec.data(), rec.size() * sizeof(Rec), cudaMemcpyHostToDevice));
    for (int k = 0; k < 8; ++k) { p.sbt[k] = {}; p.sbt[k].raygenRecord = p.drec + (k < nrg ? k : 0) * sizeof(Rec); p.sbt[k].missRecordBase = p.drec + imiss * sizeof(Rec); p.sbt[k].missRecordStrideInBytes = sizeof(Rec); p.sbt[k].missRecordCount = 1;
        p.sbt[k].hitgroupRecordBase = p.drec + ihit * sizeof(Rec); p.sbt[k].hitgroupRecordStrideInBytes = sizeof(Rec); p.sbt[k].hitgroupRecordCount = nhit; }
    return p;
}
static OptixModule make_module(const char *ptxfile, const OptixPipelineCompileOptions &po)
{
    OptixModuleCompileOptions mo{}; std::string ptx = read_file(ptxfile); char log[4096]; size_t ls = sizeof log; OptixModule m;
    OptixResult mr = optixModuleCreate(ctx, &mo, &po, ptx.c_str(), ptx.size(), log, &ls, &m); if (mr != OPTIX_SUCCESS) { fprintf(stderr, "optixModuleCreate %d: %s\n", (int)mr, log); exit(2); } return m;
}
static Pipe make_pipe_old(const char *ptxfile)
{
    OptixPipelineCompileOptions po{}; po.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS; po.numPayloadValues = 7; po.numAttributeValues = 2; po.pipelineLaunchParamsVariableName = "params"; po.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
    OptixModule m = make_module(ptxfile, po); OptixProgramGroupOptions go{}; OptixProgramGroupDesc d[4] = {};
    const char *rg[2] = {"__raygen__direct", "__raygen__persist"};
    for (int k = 0; k < 2; ++k) { d[k].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN; d[k].raygen.module = m; d[k].raygen.entryFunctionName = rg[k]; }
    d[2].kind = OPTIX_PROGRAM_GROUP_KIND_MISS; d[2].miss.module = m; d[2].miss.entryFunctionName = "__miss__ms";
    d[3].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP; d[3].hitgroup.moduleCH = m; d[3].hitgroup.entryFunctionNameCH = "__closesthit__ch";
    OptixProgramGroup pg[4]; char log[4096]; size_t ls = sizeof log; OX(optixProgramGroupCreate(ctx, d, 4, &go, log, &ls, pg));
    OptixPipelineLinkOptions lo{}; lo.maxTraceDepth = 1; OptixPipeline pipe; ls = sizeof log; OX(optixPipelineCreate(ctx, &po, &lo, pg, 4, log, &ls, &pipe));
    return finish_pipe(pipe, pg, 2, 2, 3, 1);
}
// raygens: 0 first, 1 iasnat, 2 fused, 3 replay, 4 shelltest; miss 5; hit groups 6 (triangles), 7 (shell: any hit + closest hit)
static Pipe make_pipe_new(const char *ptxfile)
{
    OptixPipelineCompileOptions po{}; po.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS | OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING; po.numPayloadValues = 7; po.numAttributeValues = 2; po.pipelineLaunchParamsVariableName = "params"; po.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
    OptixModule m = make_module(ptxfile, po); OptixProgramGroupOptions go{}; OptixProgramGroupDesc d[9] = {};
    const char *rg[6] = {"__raygen__first", "__raygen__iasnat", "__raygen__fused", "__raygen__replay", "__raygen__shelltest", "__raygen__tltest"};
    for (int k = 0; k < 6; ++k) { d[k].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN; d[k].raygen.module = m; d[k].raygen.entryFunctionName = rg[k]; }
    d[6].kind = OPTIX_PROGRAM_GROUP_KIND_MISS; d[6].miss.module = m; d[6].miss.entryFunctionName = "__miss__ms";
    d[7].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP; d[7].hitgroup.moduleCH = m; d[7].hitgroup.entryFunctionNameCH = "__closesthit__ch";
    d[8].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP; d[8].hitgroup.moduleCH = m; d[8].hitgroup.entryFunctionNameCH = "__closesthit__shell"; d[8].hitgroup.moduleAH = m; d[8].hitgroup.entryFunctionNameAH = "__anyhit__shell";
    OptixProgramGroup pg[9]; char log[4096]; size_t ls = sizeof log; OX(optixProgramGroupCreate(ctx, d, 9, &go, log, &ls, pg));
    OptixPipelineLinkOptions lo{}; lo.maxTraceDepth = 1; OptixPipeline pipe; ls = sizeof log; OptixResult r = optixPipelineCreate(ctx, &po, &lo, pg, 9, log, &ls, &pipe); if (r != OPTIX_SUCCESS) { fprintf(stderr, "pipeline: %s\n", log); exit(2); }
    return finish_pipe(pipe, pg, 6, 6, 7, 2);
}

// ---- the IAS inspector: one instance per treelet; children are a shared unit cube GAS (shell) scaled to the treelet box, or the real treelet GASes ----
static Gas build_shell(bool flip)
{
    static const float V[12][3][3] = {   // outward facing, counter clockwise seen from outside
        {{0,0,0},{0,1,0},{1,1,0}}, {{0,0,0},{1,1,0},{1,0,0}}, {{0,0,1},{1,0,1},{1,1,1}}, {{0,0,1},{1,1,1},{0,1,1}}, {{0,0,0},{1,0,0},{1,0,1}}, {{0,0,0},{1,0,1},{0,0,1}},
        {{0,1,0},{0,1,1},{1,1,1}}, {{0,1,0},{1,1,1},{1,1,0}}, {{0,0,0},{0,0,1},{0,1,1}}, {{0,0,0},{0,1,1},{0,1,0}}, {{1,0,0},{1,1,0},{1,1,1}}, {{1,0,0},{1,1,1},{1,0,1}}};
    std::vector<float> v; for (int t = 0; t < 12; ++t) { int o[3] = {0, 1, 2}; if (flip) std::swap(o[1], o[2]); for (int k = 0; k < 3; ++k) for (int a = 0; a < 3; ++a) v.push_back(V[t][o[k]][a]); }
    CUdeviceptr dv; CK(cudaMalloc((void **)&dv, v.size() * 4)); CK(cudaMemcpy((void *)dv, v.data(), v.size() * 4, cudaMemcpyHostToDevice));
    OptixBuildInput bi{}; bi.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES; unsigned fl = OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL; bi.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3; bi.triangleArray.numVertices = 36; bi.triangleArray.vertexBuffers = &dv; bi.triangleArray.flags = &fl; bi.triangleArray.numSbtRecords = 1;
    OptixAccelBuildOptions ao{}; ao.buildFlags = OPTIX_BUILD_FLAG_NONE; ao.operation = OPTIX_BUILD_OPERATION_BUILD; OptixAccelBufferSizes bs{}; OX(optixAccelComputeMemoryUsage(ctx, &ao, &bi, 1, &bs));
    CUdeviceptr tmp, out; CK(cudaMalloc((void **)&tmp, bs.tempSizeInBytes)); CK(cudaMalloc((void **)&out, bs.outputSizeInBytes)); Gas g; g.buf = out; g.bytes = bs.outputSizeInBytes;
    OX(optixAccelBuild(ctx, 0, &ao, &bi, 1, tmp, bs.tempSizeInBytes, out, bs.outputSizeInBytes, &g.h, nullptr, 0)); CK(cudaDeviceSynchronize()); CK(cudaFree((void *)tmp)); return g;
}
static Gas build_ias(const std::vector<OptixInstance> &inst)
{
    CUdeviceptr di; CK(cudaMalloc((void **)&di, inst.size() * sizeof(OptixInstance))); CK(cudaMemcpy((void *)di, inst.data(), inst.size() * sizeof(OptixInstance), cudaMemcpyHostToDevice));
    OptixBuildInput bi{}; bi.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES; bi.instanceArray.instances = di; bi.instanceArray.numInstances = (unsigned)inst.size();
    OptixAccelBuildOptions ao{}; ao.buildFlags = OPTIX_BUILD_FLAG_NONE; ao.operation = OPTIX_BUILD_OPERATION_BUILD; OptixAccelBufferSizes bs{}; OX(optixAccelComputeMemoryUsage(ctx, &ao, &bi, 1, &bs));
    CUdeviceptr tmp, out; CK(cudaMalloc((void **)&tmp, bs.tempSizeInBytes)); CK(cudaMalloc((void **)&out, bs.outputSizeInBytes)); Gas g; g.buf = out; g.bytes = bs.outputSizeInBytes;
    OX(optixAccelBuild(ctx, 0, &ao, &bi, 1, tmp, bs.tempSizeInBytes, out, bs.outputSizeInBytes, &g.h, nullptr, 0)); CK(cudaDeviceSynchronize()); CK(cudaFree((void *)tmp)); return g;
}
static OptixInstance mk_inst(unsigned id, OptixTraversableHandle h, const float *lo, const float *hi)
{
    OptixInstance in{}; const float sx = hi ? hi[0] - lo[0] : 1.f, sy = hi ? hi[1] - lo[1] : 1.f, sz = hi ? hi[2] - lo[2] : 1.f;
    in.transform[0] = sx; in.transform[3] = lo ? lo[0] : 0.f; in.transform[5] = sy; in.transform[7] = lo ? lo[1] : 0.f; in.transform[10] = sz; in.transform[11] = lo ? lo[2] : 0.f;
    in.instanceId = id; in.sbtOffset = 0; in.visibilityMask = 255; in.flags = OPTIX_INSTANCE_FLAG_NONE; in.traversableHandle = h; return in;
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
__global__ void k_gather(const Ray8 *src, const unsigned *perm, Ray8 *dst, unsigned n) { const unsigned i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) dst[i] = src[perm[i]]; }
__global__ void k_cuts(const unsigned *key, unsigned n, unsigned nq, unsigned *qend, unsigned *head)
{
    const unsigned q = threadIdx.x; if (q >= nq) return;
    auto cut = [&](unsigned k) { if (k == 0) return 0u; if (k >= nq) return n; const unsigned i = (unsigned)((unsigned long long)k * n / nq); if (i == 0) return 0u;
        const unsigned v = key[i - 1]; unsigned lo = i, hi = n; while (lo < hi) { const unsigned mid = (lo + hi) / 2; if (key[mid] > v) hi = mid; else lo = mid + 1; } return lo; };   // first index at or after i whose key differs from key[i - 1] (keys are sorted)
    head[q] = cut(q); qend[q] = cut(q + 1);
}
__global__ void k_final_map(unsigned n, const RefOut *src, const unsigned *perm, const unsigned *tl, unsigned C, const unsigned *toff, const unsigned *tri_sorted, RefOut *dst)
{
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return; RefOut o = src[i];
    if (tl && o.hit && tl[i] < C) o.prim = tri_sorted[toff[tl[i]] + o.prim];
    dst[perm ? perm[i] : i] = o;
}

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

// ---- device helper kernels for the new arms ----
__global__ void k_reset(unsigned *hist, unsigned C1, unsigned *qhead, unsigned *qtail, unsigned C, unsigned *cnt, unsigned long long *cyc)
{
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < C1) hist[i] = 0; if (i < C) { qhead[i] = 0; qtail[i] = 0; } if (i < 8) { cnt[i] = 0; cyc[i] = 0; }
}
// one block: exclusive prefix of the first visit counts (balances the TPC ranges by rays), queue bases (capacity 3 x count + 1024), TPC ranges [tlo, thi), start pointers, rays outstanding
__global__ void k_scan(const unsigned *hist, unsigned C, unsigned nq, unsigned *qbase, unsigned *qcap, unsigned *tlo, unsigned *thi, unsigned *tcur, unsigned *outst)
{
    typedef cub::BlockScan<unsigned, 1024> BS; __shared__ typename BS::TempStorage tmp; __shared__ unsigned pf[4097];
    unsigned h[4], cp[4], a[4], b[4];
    for (int j = 0; j < 4; ++j) { const unsigned i = threadIdx.x * 4 + j; h[j] = i < C ? hist[i] : 0; cp[j] = i < C ? 3 * h[j] + 1024 : 0; }
    unsigned total, tot2; BS(tmp).ExclusiveSum(h, a, total); __syncthreads(); BS(tmp).ExclusiveSum(cp, b, tot2); __syncthreads();
    for (int j = 0; j < 4; ++j) { const unsigned i = threadIdx.x * 4 + j; if (i < C) { pf[i] = a[j]; qbase[i] = b[j]; qcap[i] = cp[j]; } }
    if (threadIdx.x == 0) { pf[C] = total; outst[0] = total; }
    __syncthreads();
    auto lower = [&](unsigned q) { if (q >= nq) return C; const unsigned target = (unsigned)((unsigned long long)q * total / nq); unsigned lo = 0, hi = C; while (lo < hi) { const unsigned m = (lo + hi) / 2; if (pf[m] >= target) hi = m; else lo = m + 1; } return lo; };
    if (threadIdx.x < nq) { const unsigned q = threadIdx.x; tlo[q] = lower(q); thi[q] = lower(q + 1); tcur[q] = tlo[q]; }
}
__global__ void k_enq(const Cur *cur, unsigned n, unsigned C, unsigned *qtail, const unsigned *qbase, unsigned *qbuf)
{
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    const unsigned id = cur[i].id; if (id >= C) return; const unsigned s = atomicAdd(&qtail[id], 1u); qbuf[(size_t)qbase[id] + s] = i;
}
__global__ void k_lvl_key(const Cur *cur, const unsigned short *vis, unsigned n, unsigned k, unsigned C, unsigned *key) { const unsigned i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) key[i] = cur[i].k > k ? (unsigned)vis[(size_t)i * KMAX + k] : C; }
__global__ void k_count_lt(const unsigned *key, unsigned n, unsigned C, unsigned *cnt) { const unsigned i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n && key[i] < C) atomicAdd(cnt, 1u); }
__global__ void k_lvl_gather(unsigned m, const unsigned *perm, const unsigned *kout, const Ray8 *rays, unsigned *lidx, Ray8 *lray, unsigned *lkey) { const unsigned i = blockIdx.x * blockDim.x + threadIdx.x; if (i < m) { lidx[i] = perm[i]; lray[i] = rays[perm[i]]; lkey[i] = kout[i]; } }
__global__ void k_map_gi(unsigned n, RefOut *o, const unsigned *toff, const unsigned *tri_sorted) { const unsigned i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n && o[i].hit) o[i].prim = tri_sorted[toff[o[i].gas_index] + o[i].prim]; }

struct Arm { std::string name; int R; };
static const char *jnum(char *b, size_t n, double v) { snprintf(b, n, "%.4f", v); return b; }

int main(int argc, char **argv)
{
    if (argc < 8) { fprintf(stderr, "usage: see the top of treelet3.cu\n"); return 2; }
    const std::string mode = argv[1], work = argv[4], scene = argv[5]; const unsigned seed = (unsigned)atoi(argv[6]); const std::string part = argv[7];
    CK(cudaFree(0)); OX(optixInit()); OptixDeviceContextOptions co{}; OX(optixDeviceContextCreate(0, &co, &ctx));
    cudaDeviceProp dp; CK(cudaGetDeviceProperties(&dp, 0)); const unsigned nsm = dp.multiProcessorCount, nq = nsm / 2;
    std::vector<Tri> tris;
    if (scene.rfind("soup:", 0) == 0) tris = gen_scene((unsigned)atoi(scene.c_str() + 5), seed); else if (scene.rfind("file:", 0) == 0) tris = read_bin<Tri>(scene.substr(5)); else { fprintf(stderr, "bad SCENE\n"); return 2; }
    const unsigned N = (unsigned)tris.size(); float slo[3] = {1e30f, 1e30f, 1e30f}, shi[3] = {-1e30f, -1e30f, -1e30f};
    for (const Tri &t : tris) for (const float3 &p : {t.a, t.b, t.c}) { const float v[3] = {p.x, p.y, p.z}; for (int a = 0; a < 3; ++a) { slo[a] = std::min(slo[a], v[a]); shi[a] = std::max(shi[a], v[a]); } }
    const float diag = std::sqrt((shi[0] - slo[0]) * (shi[0] - slo[0]) + (shi[1] - slo[1]) * (shi[1] - slo[1]) + (shi[2] - slo[2]) * (shi[2] - slo[2]));
    Pipe pp = make_pipe_old(argv[2]);
    std::vector<unsigned> tl_of_tri, tri_sorted, toff; unsigned C = 0; parse_part(part, tris, tl_of_tri, tri_sorted, toff, C);
    if (C + 1 > 4096) { fprintf(stderr, "too many treelets for k_scan\n"); return 2; }
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
        const std::string tag = argv[8], kind = argv[9]; const unsigned S = (unsigned)atoi(argv[10]); const float eps = 3e-5f * diag;
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
    const std::string tag = argv[8]; const unsigned WPS = (unsigned)atoi(argv[9]); const int ROUNDS = atoi(argv[10]); const char *outjson = argv[11];
    std::vector<Arm> arms; for (int a = 12; a < argc; ++a) { std::string s = argv[a]; int R = 1; const size_t c = s.find(':'); if (c != std::string::npos) { R = atoi(s.c_str() + c + 1); s = s.substr(0, c); }
        const char *ok[] = {"nat_R", "e_mo4", "e_mo5", "e_opT", "tl_test", "ias_nat", "f_pin", "f_nopin", "f_rep", "mo_rep", "nat_rep"}; bool f = false; for (const char *o : ok) f |= s == o; if (!f) { fprintf(stderr, "bad arm %s\n", s.c_str()); return 2; } arms.push_back({s, R}); }
    std::vector<Ray8> rays = read_bin<Ray8>(work + "/" + tag + "_rays.bin"); std::vector<RefOut> ref = read_bin<RefOut>(work + "/" + tag + "_ref.bin"); const unsigned n = (unsigned)rays.size();
    std::vector<unsigned> orc(n); for (unsigned i = 0; i < n; ++i) orc[i] = tl_of_tri[ref[i].prim];
    unsigned kb = 1; while ((1u << kb) < C + 1) ++kb;
    // treelet boxes (tight plus a margin), shell IAS, real IAS
    std::vector<float> bx(6ull * C); float maxdiag = 0.f, sumdiag = 0.f; { const float mg = 1e-4f * diag; for (unsigned t = 0; t < C; ++t) { float lo[3] = {1e30f, 1e30f, 1e30f}, hi[3] = {-1e30f, -1e30f, -1e30f};
        for (unsigned j = toff[t]; j < toff[t + 1]; ++j) { const Tri &q = sv[j]; for (const float3 &p : {q.a, q.b, q.c}) { const float v[3] = {p.x, p.y, p.z}; for (int a = 0; a < 3; ++a) { lo[a] = std::min(lo[a], v[a]); hi[a] = std::max(hi[a], v[a]); } } }
        for (int a = 0; a < 3; ++a) { bx[6ull * t + a] = lo[a] - mg; bx[6ull * t + 3 + a] = hi[a] + mg; }
        const float dg = std::sqrt((hi[0] - lo[0]) * (hi[0] - lo[0]) + (hi[1] - lo[1]) * (hi[1] - lo[1]) + (hi[2] - lo[2]) * (hi[2] - lo[2])) + 2.f * mg * 1.7321f; maxdiag = std::max(maxdiag, dg); sumdiag += dg; } }
    const float pull = maxdiag * 1.002f; { std::vector<float> dg(C); for (unsigned t = 0; t < C; ++t) dg[t] = std::sqrt((bx[6ull*t+3]-bx[6ull*t])*(bx[6ull*t+3]-bx[6ull*t]) + (bx[6ull*t+4]-bx[6ull*t+1])*(bx[6ull*t+4]-bx[6ull*t+1]) + (bx[6ull*t+5]-bx[6ull*t+2])*(bx[6ull*t+5]-bx[6ull*t+2])); std::sort(dg.begin(), dg.end());
      printf("{\"box_diag_median\": %.4f, \"box_diag_p99\": %.4f, \"box_diag_max\": %.4f, \"pull\": %.4f}\n", dg[dg.size() / 2], dg[(size_t)(dg.size() * 0.99)], dg.back(), pull); fflush(stdout); }
    CUdeviceptr dbox; CK(cudaMalloc((void **)&dbox, bx.size() * 4)); CK(cudaMemcpy((void *)dbox, bx.data(), bx.size() * 4, cudaMemcpyHostToDevice));
    Pipe pn = make_pipe_new(argv[3]);
    // winding self test: an entering ray must hit the shell with back faces culled, a ray from inside must not
    bool flip = false;
    for (int attempt = 0; attempt < 2; ++attempt) {
        Gas sh = build_shell(flip); const float lo[3] = {0, 0, 0}, hi[3] = {1, 1, 1}; Gas ia = build_ias({mk_inst(0, sh.h, lo, hi)});
        const float ub[6] = {0, 0, 0, 1, 1, 1}; CUdeviceptr dub; CK(cudaMalloc((void **)&dub, 24)); CK(cudaMemcpy((void *)dub, ub, 24, cudaMemcpyHostToDevice));
        const Ray8 rs[2] = {{{-1.f, .5f, .5f}, 0.f, {1.f, 0.f, 0.f}, 1e16f}, {{.5f, .5f, .5f}, 0.f, {1.f, 0.f, 0.f}, 1e16f}}; RefOut res2[2]; bool good = true;
        for (int k = 0; k < 2; ++k) { Params p{}; p.ias_shell = ia.h; p.boxes = (const float *)dub; p.pull = 0.f; CUdeviceptr dr; CK(cudaMalloc((void **)&dr, 32)); CK(cudaMemcpy((void *)dr, &rs[k], 32, cudaMemcpyHostToDevice)); p.rays = (const Ray8 *)dr;
            CUdeviceptr dout; CK(cudaMalloc((void **)&dout, 32)); p.out = (RefOut *)dout; CUdeviceptr dpp = to_dev(p); OX(optixLaunch(pn.pipe, 0, dpp, sizeof p, &pn.sbt[4], 32, 1, 1)); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(&res2[k], (void *)dout, 32, cudaMemcpyDeviceToHost)); }
        good = res2[0].hit == 1 && std::fabs(res2[0].t - 1.f) < 1e-4f && res2[1].hit == 0;
        printf("{\"shell_winding_test\": {\"flip\": %d, \"enter_found\": %u, \"enter_t\": %.5f, \"inside_found\": %u, \"ok\": %d}}\n", (int)flip, res2[0].hit, res2[0].t, res2[1].hit, (int)good); fflush(stdout);
        if (good) break; flip = !flip; if (attempt == 1) { fprintf(stderr, "shell winding test failed both ways\n"); return 2; }
    }
    Gas shell = build_shell(flip); std::vector<OptixInstance> ins_s(C), ins_r(C);
    for (unsigned t = 0; t < C; ++t) { ins_s[t] = mk_inst(t, shell.h, &bx[6ull * t], &bx[6ull * t + 3]); ins_r[t] = mk_inst(t, tl.h[t], nullptr, nullptr); }
    Gas ias_s = build_ias(ins_s), ias_r = build_ias(ins_r);
    printf("{\"ias_shell_bytes\": %zu, \"ias_real_bytes\": %zu}\n", ias_s.bytes, ias_r.bytes); fflush(stdout);
    // Morton bounds of the ray origins
    float olo[3] = {1e30f, 1e30f, 1e30f}, ohi[3] = {-1e30f, -1e30f, -1e30f}; for (const Ray8 &r : rays) for (int a = 0; a < 3; ++a) { olo[a] = std::min(olo[a], r.o[a]); ohi[a] = std::max(ohi[a], r.o[a]); }
    const float3 mlo = make_float3(olo[0], olo[1], olo[2]), minv = make_float3(1.f / std::max(ohi[0] - olo[0], 1e-20f), 1.f / std::max(ohi[1] - olo[1], 1e-20f), 1.f / std::max(ohi[2] - olo[2], 1e-20f));
    // device buffers
    auto A = [&](size_t bytes) { CUdeviceptr d; CK(cudaMalloc((void **)&d, bytes)); return d; };
    CUdeviceptr d_rn = A(n * sizeof(Ray8)); CK(cudaMemcpy((void *)d_rn, rays.data(), n * sizeof(Ray8), cudaMemcpyHostToDevice));
    std::vector<OptixTraversableHandle> hh = tl.h; hh.push_back(full.h); CUdeviceptr d_h = A(8ull * (C + 1)); CK(cudaMemcpy((void *)d_h, hh.data(), 8ull * (C + 1), cudaMemcpyHostToDevice));
    CUdeviceptr d_rs = A(n * sizeof(Ray8)); RefOut *d_of = (RefOut *)A(n * sizeof(RefOut));
    CUdeviceptr d_oa = A(n * sizeof(RefOut)), d_on = A(n * sizeof(RefOut)), d_oc = A(n * sizeof(RefOut));
    CUdeviceptr d_kin = A(n * 4), d_kout = A(n * 4), d_pin = A(n * 4), d_perm = A(n * 4), d_orc = A(n * 4), d_n = A(4), d_c1 = A(4);
    { std::vector<unsigned> iota(n); std::iota(iota.begin(), iota.end(), 0u); CK(cudaMemcpy((void *)d_pin, iota.data(), n * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy((void *)d_orc, orc.data(), n * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy((void *)d_n, &n, 4, cudaMemcpyHostToDevice)); }
    CUdeviceptr d_toff = A(4ull * (C + 1)), d_tris = A(4ull * N); CK(cudaMemcpy((void *)d_toff, toff.data(), 4ull * (C + 1), cudaMemcpyHostToDevice)); CK(cudaMemcpy((void *)d_tris, tri_sorted.data(), 4ull * N, cudaMemcpyHostToDevice));
    CUdeviceptr d_qend_g = A(4ull * nq), d_head_g = A(4ull * nq), d_stats = A(16), d_smhist = A(1024), d_qend1 = A(4); const unsigned maxwarps = 4096; CUdeviceptr d_warpsm = A(4ull * maxwarps); CK(cudaMemcpy((void *)d_qend1, &n, 4, cudaMemcpyHostToDevice));
    size_t sort_bytes = 0; cub::DeviceRadixSort::SortPairs(nullptr, sort_bytes, (unsigned *)d_kin, (unsigned *)d_kout, (unsigned *)d_pin, (unsigned *)d_perm, n, 0, 24); CUdeviceptr d_stmp = A(sort_bytes);
    // fused design buffers
    const size_t qentries = 3ull * n + 1024ull * (C + 1) + 64; CUdeviceptr d_cur = A(n * sizeof(Cur)), d_vis = A((size_t)n * KMAX * 2), d_hist = A(4ull * (C + 1)), d_qbase = A(4ull * C), d_qcap = A(4ull * C), d_qhead = A(4ull * C), d_qtail = A(4ull * C), d_qbuf = A(qentries * 4);
    CUdeviceptr d_tcur = A(4ull * nq), d_tlo = A(4ull * nq), d_thi = A(4ull * nq), d_outst = A(4), d_cyc = A(64), d_cnt = A(32);
    unsigned *h_hb = nullptr; CUdeviceptr d_hbdev = 0; if (getenv("T3_HB")) { CK(cudaHostAlloc((void **)&h_hb, 4ull * 32768, cudaHostAllocMapped)); memset(h_hb, 0, 4ull * 32768); CK(cudaHostGetDevicePointer((void **)&d_hbdev, h_hb, 0)); }
    const unsigned PW = nsm * WPS; if (PW > maxwarps) { fprintf(stderr, "WPS too large\n"); return 2; }
    auto base = [&]() { Params p{}; p.handle = full.h; p.tl_handle = (const OptixTraversableHandle *)d_h; p.n_dev = (const unsigned *)d_n; p.stats = (unsigned *)d_stats; p.smhist = (unsigned *)d_smhist; p.warp_sm = (unsigned *)(getenv("T3_HB") ? d_hbdev : d_warpsm); p.qend = (const unsigned *)d_qend1; p.head = (unsigned *)d_head_g; p.nq = 1;
        p.ias_shell = ias_s.h; p.ias_real = ias_r.h; p.boxes = (const float *)dbox; p.pull = pull; p.C = C; p.record = 0; p.steal_tries = 3; p.cur = (Cur *)d_cur; p.qhead = (unsigned *)d_qhead; p.qtail = (unsigned *)d_qtail; p.qbase = (const unsigned *)d_qbase; p.qcap = (const unsigned *)d_qcap; p.qbuf = (unsigned *)d_qbuf;
        p.tcur = (unsigned *)d_tcur; p.tlo = (const unsigned *)d_tlo; p.thi = (const unsigned *)d_thi; p.outstanding = (unsigned *)d_outst; p.hist = (unsigned *)d_hist; p.toff = (const unsigned *)d_toff; p.tri_sorted = (const unsigned *)d_tris; p.vis = (unsigned short *)d_vis; p.cyc = (unsigned long long *)d_cyc; p.cnt = (unsigned *)d_cnt; return p; };
    Params p_root = base(); p_root.rays = (const Ray8 *)d_rs; p_root.out = (RefOut *)d_oa; CUdeviceptr P_root = to_dev(p_root);
    Params p_nat = base(); p_nat.rays = (const Ray8 *)d_rn; p_nat.out = (RefOut *)d_oc; CUdeviceptr P_nat = to_dev(p_nat);
    Params p_tp = base(); p_tp.rays = (const Ray8 *)d_rs; p_tp.out = (RefOut *)d_oa; p_tp.tl = (const unsigned *)d_kout; p_tp.use_tl = 1; p_tp.pinned = 1; p_tp.nq = nq; p_tp.qend = (const unsigned *)d_qend_g; p_tp.head = (unsigned *)d_head_g; CUdeviceptr P_tp = to_dev(p_tp);
    Params p_ias = base(); p_ias.rays = (const Ray8 *)d_rn; p_ias.out = (RefOut *)d_of; CUdeviceptr P_ias = to_dev(p_ias);
    Params p_f = base(); p_f.rays = (const Ray8 *)d_rn; p_f.out = (RefOut *)d_of; p_f.nq = nq; p_f.pinned = 1; CUdeviceptr P_first = to_dev(p_f), P_fp = P_first; Params p_fr = p_f; p_fr.record = 1; CUdeviceptr P_fpr = to_dev(p_fr); Params p_fn = p_f; p_fn.pinned = 0; CUdeviceptr P_fn = to_dev(p_fn);
    const unsigned grd = (n + 255) / 256, NT = (n + 31) / 32 * 32;
    cudaEvent_t ev[8]; for (int k = 0; k < 8; ++k) CK(cudaEventCreate(&ev[k]));
    nvmlDevice_t nd; char bus[32]; CK(cudaDeviceGetPCIBusId(bus, sizeof bus, 0)); bool nv_ok = nvmlInit_v2() == NVML_SUCCESS && nvmlDeviceGetHandleByPciBusId_v2(bus, &nd) == NVML_SUCCESS;
    struct Res { std::vector<double> ms; std::vector<unsigned> clk; unsigned long bad = 0, tbad = 0, ties = 0; std::vector<std::vector<double>> st; unsigned cnt[8] = {0}; unsigned long long cyc[8] = {0}; bool has_f = false; unsigned K = 0, V = 0; std::vector<unsigned> lvlcnt; };
    std::vector<Res> res(arms.size());
    auto sortk = [&](CUdeviceptr kin, int bits) { size_t b = sort_bytes; cub::DeviceRadixSort::SortPairs((void *)d_stmp, b, (unsigned *)kin, (unsigned *)d_kout, (unsigned *)d_pin, (unsigned *)d_perm, n, 0, bits); };
    auto launch_o = [&](CUdeviceptr P, int sbt, unsigned threads) { OX(optixLaunch(pp.pipe, 0, P, sizeof(Params), &pp.sbt[sbt], threads, 1, 1)); };
    auto launch_n = [&](CUdeviceptr P, int sbt, unsigned threads) { OX(optixLaunch(pn.pipe, 0, P, sizeof(Params), &pn.sbt[sbt], threads, 1, 1)); };
    auto reset_stats = [&]() { CK(cudaMemsetAsync((void *)d_stats, 0, 16)); CK(cudaMemsetAsync((void *)d_smhist, 0, 1024)); CK(cudaMemsetAsync((void *)d_warpsm, 0xff, 4ull * maxwarps)); };
    // ---- the fused pass: A1 inspector, A2 scan + counting enqueue, B persistent drain ----
    auto fused_pass = [&](bool pin, bool record, int &k, bool mk) {
        auto mark = [&]() { if (mk) CK(cudaEventRecord(ev[k++])); };
        mark();
        CK(cudaMemsetAsync((void *)d_qbuf, 0xff, qentries * 4)); k_reset<<<(C + 1 + 255) / 256, 256>>>((unsigned *)d_hist, C + 1, (unsigned *)d_qhead, (unsigned *)d_qtail, C, (unsigned *)d_cnt, (unsigned long long *)d_cyc);
        launch_n(P_first, 0, NT); mark();                                                                                             // 1: reset + inspector step of every ray (first treelet, histogram)
        k_scan<<<1, 1024>>>((const unsigned *)d_hist, C, nq, (unsigned *)d_qbase, (unsigned *)d_qcap, (unsigned *)d_tlo, (unsigned *)d_thi, (unsigned *)d_tcur, (unsigned *)d_outst);
        k_enq<<<grd, 256>>>((const Cur *)d_cur, n, C, (unsigned *)d_qtail, (const unsigned *)d_qbase, (unsigned *)d_qbuf); mark();    // 2: scan + enqueue
        launch_n(record ? P_fpr : (pin ? P_fp : P_fn), 2, PW * 32); mark();                                                           // 3: persistent drain, trace, step, requeue
    };
    // ---- grouping reuse: levels sorted by treelet, built from the visit records of one fused pass ----
    struct Lvl { unsigned off, cnt; }; std::vector<Lvl> lv; std::vector<CUdeviceptr> P_lvl; CUdeviceptr d_lidx = 0, d_lray = 0, d_lkey = 0, d_lhead = 0, d_lend = 0; size_t lcap = 0;
    auto build_levels = [&](Res &r) {
        unsigned cn[8]; CK(cudaMemcpy(cn, (void *)d_cnt, 32, cudaMemcpyDeviceToHost)); const unsigned K = cn[6], V = cn[3]; r.K = K; r.V = V;
        if (K > KMAX || cn[4] || cn[5]) { fprintf(stderr, "invalid pass for levels: K %u overflow %u timeouts %u\n", K, cn[4], cn[5]); exit(3); }
        const size_t need = (size_t)V + 4096; if (need > lcap) { if (lcap) { CK(cudaFree((void *)d_lidx)); CK(cudaFree((void *)d_lray)); CK(cudaFree((void *)d_lkey)); } lcap = need + need / 20; d_lidx = A(lcap * 4); d_lray = A(lcap * sizeof(Ray8)); d_lkey = A(lcap * 4); if (!d_lhead) { d_lhead = A(4ull * nq * KMAX); d_lend = A(4ull * nq * KMAX); } }
        lv.clear(); unsigned off = 0;
        for (unsigned k = 0; k < K; ++k) {
            k_lvl_key<<<grd, 256>>>((const Cur *)d_cur, (const unsigned short *)d_vis, n, k, C, (unsigned *)d_kin); sortk(d_kin, kb);
            CK(cudaMemsetAsync((void *)d_c1, 0, 4)); k_count_lt<<<grd, 256>>>((const unsigned *)d_kout, n, C, (unsigned *)d_c1); unsigned m; CK(cudaMemcpy(&m, (void *)d_c1, 4, cudaMemcpyDeviceToHost));
            if ((size_t)off + m > lcap) { fprintf(stderr, "level capacity\n"); exit(3); }
            k_lvl_gather<<<(m + 255) / 256 + 1, 256>>>(m, (const unsigned *)d_perm, (const unsigned *)d_kout, (const Ray8 *)d_rn, (unsigned *)d_lidx + off, (Ray8 *)d_lray + off, (unsigned *)d_lkey + off);
            lv.push_back({off, m}); off += m;
        }
        CK(cudaDeviceSynchronize()); r.lvlcnt.clear(); for (const Lvl &l : lv) r.lvlcnt.push_back(l.cnt);
        for (CUdeviceptr d : P_lvl) CK(cudaFree((void *)d)); P_lvl.clear();
        for (unsigned k = 0; k < K; ++k) { Params p = base(); p.rays = (const Ray8 *)d_rn; p.out = (RefOut *)d_of; p.nq = nq; p.lvl_idx = (const unsigned *)d_lidx + lv[k].off; p.lvl_ray = (const Ray8 *)d_lray + lv[k].off; p.lvl_key = (const unsigned *)d_lkey + lv[k].off; p.lvl_level = k; p.lvl_head = (unsigned *)d_lhead + (size_t)k * nq; p.lvl_end = (const unsigned *)d_lend + (size_t)k * nq; P_lvl.push_back(to_dev(p)); }
    };
    auto replay_once = [&]() { for (size_t k = 0; k < lv.size(); ++k) { k_cuts<<<1, 128>>>((const unsigned *)d_lkey + lv[k].off, lv[k].cnt, nq, (unsigned *)d_lend + k * nq, (unsigned *)d_lhead + k * nq); launch_n(P_lvl[k], 3, PW * 32); } };
    auto check_nat = [&](const RefOut *dev, Res &r) {
        std::vector<RefOut> o(n); CK(cudaMemcpy(o.data(), dev, (size_t)n * sizeof(RefOut), cudaMemcpyDeviceToHost)); for (unsigned i = 0; i < n; ++i) { const RefOut &x = o[i], &y = ref[i]; if (getenv("T3_DEBUG") && i < 6) fprintf(stderr, "DEBUG ray %u got hit %u t %g prim %u gi %u | ref hit %u t %g prim %u\n", i, x.hit, x.t, x.prim, x.gas_index, y.hit, y.t, y.prim); if (x.hit != y.hit) ++r.bad; else if (x.prim != y.prim) { if (std::fabs(x.t - y.t) <= 1e-5f * std::fabs(y.t) + 1e-7f) ++r.ties; else ++r.bad; } else if (x.t != y.t) ++r.tbad; } };
    auto run_arm = [&](size_t j, bool timed, bool verify) {
        const Arm &a = arms[j]; Res &r = res[j]; float ms = 0; const std::string &nm = a.name; const RefOut *chk = nullptr;
        CK(cudaMemset((void *)d_on, 0xff, (size_t)n * sizeof(RefOut))); CK(cudaMemset((void *)d_of, 0xff, (size_t)n * sizeof(RefOut))); CK(cudaMemset((void *)d_oc, 0xff, (size_t)n * sizeof(RefOut))); CK(cudaDeviceSynchronize());
        cudaEvent_t e1; CK(cudaEventCreate(&e1)); int k = 0;
        if (nm == "nat_R" || nm == "nat_rep") { CK(cudaEventRecord(ev[7])); for (int q = 0; q < a.R; ++q) launch_o(P_nat, 0, NT); chk = (const RefOut *)d_oc; }
        else if (nm == "e_mo4" || nm == "e_mo5" || nm == "mo_rep") {
            const int b = nm == "e_mo4" ? 4 : 5; CK(cudaEventRecord(ev[7]));
            k_key_morton<<<grd, 256>>>((const Ray8 *)d_rn, (unsigned *)d_kin, n, mlo, minv, b); sortk(d_kin, 3 * b); k_gather<<<grd, 256>>>((const Ray8 *)d_rn, (const unsigned *)d_perm, (Ray8 *)d_rs, n);
            for (int q = 0; q < a.R; ++q) { launch_o(P_root, 0, NT); k_final_map<<<grd, 256>>>(n, (const RefOut *)d_oa, (const unsigned *)d_perm, nullptr, C, (const unsigned *)d_toff, (const unsigned *)d_tris, (RefOut *)d_on); }
            chk = (const RefOut *)d_on;
        } else if (nm == "e_opT") {
            CK(cudaEventRecord(ev[7])); sortk(d_orc, kb); k_gather<<<grd, 256>>>((const Ray8 *)d_rn, (const unsigned *)d_perm, (Ray8 *)d_rs, n);
            k_cuts<<<1, 128>>>((const unsigned *)d_kout, n, nq, (unsigned *)d_qend_g, (unsigned *)d_head_g); reset_stats(); launch_o(P_tp, 1, PW * 32);
            k_final_map<<<grd, 256>>>(n, (const RefOut *)d_oa, (const unsigned *)d_perm, (const unsigned *)d_kout, C, (const unsigned *)d_toff, (const unsigned *)d_tris, (RefOut *)d_on); chk = (const RefOut *)d_on;
        } else if (nm == "tl_test") {
            CK(cudaEventRecord(ev[7])); { Params p = base(); p.rays = (const Ray8 *)d_rn; p.out = (RefOut *)d_of; p.tl = (const unsigned *)d_orc; CUdeviceptr dpp = to_dev(p); launch_n(dpp, 5, NT); } chk = d_of;
        } else if (nm == "ias_nat") {
            CK(cudaEventRecord(ev[7])); launch_n(P_ias, 1, NT); k_map_gi<<<grd, 256>>>(n, d_of, (const unsigned *)d_toff, (const unsigned *)d_tris); chk = d_of;
        } else if (nm == "f_pin" || nm == "f_nopin") {
            CK(cudaEventRecord(ev[7])); fused_pass(nm == "f_pin", false, k, true); chk = d_of;
        } else if (nm == "f_rep") {
            CK(cudaEventRecord(ev[7])); fused_pass(true, true, k, true); CK(cudaEventRecord(ev[k++]));
            if (getenv("T3_DEBUG")) { CK(cudaDeviceSynchronize()); std::vector<Cur> hc(n); std::vector<unsigned short> hv((size_t)n * KMAX); CK(cudaMemcpy(hc.data(), (void *)d_cur, n * sizeof(Cur), cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hv.data(), (void *)d_vis, (size_t)n * KMAX * 2, cudaMemcpyDeviceToHost)); unsigned has = 0, first = 0; double av = 0; for (unsigned i = 0; i < n; ++i) { bool f = false; for (unsigned q = 0; q < hc[i].k && q < KMAX; ++q) f |= hv[(size_t)i * KMAX + q] == orc[i]; has += f; first += hv[(size_t)i * KMAX] == orc[i]; av += hc[i].k; } fprintf(stderr, "DEBUG oracle treelet visited %u of %u, first visit is oracle %u, mean visits %.2f\n", has, n, first, av / n); }
            build_levels(r); CK(cudaEventRecord(ev[k++])); for (int q = 1; q < a.R; ++q) replay_once(); CK(cudaEventRecord(ev[k++])); chk = d_of;
        } else { fprintf(stderr, "bad arm\n"); exit(2); }
        CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); CK(cudaEventElapsedTime(&ms, ev[7], e1));
        if (nm == "f_pin" || nm == "f_nopin" || nm == "f_rep") {
            const int ns = nm == "f_rep" ? 5 : 3; if (r.st.empty()) r.st.resize(ns);
            for (int s = 0; s < ns; ++s) { float t; CK(cudaEventElapsedTime(&t, ev[s], ev[s + 1])); r.st[s].push_back(t); }
        }
        CK(cudaEventDestroy(e1));
        if (verify) {
            check_nat(chk, r);
            if (nm == "f_pin" || nm == "f_nopin" || nm == "f_rep") { r.has_f = true; if (getenv("T3_DEBUG")) { unsigned os; std::vector<unsigned> hd(C), tlv(C), cp(C), lo(nq), hi(nq), hs(C + 1); CK(cudaMemcpy(&os, (void *)d_outst, 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hd.data(), (void *)d_qhead, 4ull * C, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(tlv.data(), (void *)d_qtail, 4ull * C, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(cp.data(), (void *)d_qcap, 4ull * C, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(lo.data(), (void *)d_tlo, 4ull * nq, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hi.data(), (void *)d_thi, 4ull * nq, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hs.data(), (void *)d_hist, 4ull * (C + 1), cudaMemcpyDeviceToHost)); unsigned long long un = 0, tot = 0, hsum = 0; for (unsigned t = 0; t < C; ++t) { un += tlv[t] - hd[t]; tot += tlv[t]; hsum += hs[t]; } { unsigned long long cy[8]; CK(cudaMemcpy(cy, (void *)d_cyc, 64, cudaMemcpyDeviceToHost));  } fprintf(stderr, "DEBUG outstanding %u unclaimed %llu tails %llu hist_sum %llu hist_none %u; tpc ranges:", os, un, tot, hsum, hs[C]); for (unsigned q = 0; q < 8; ++q) fprintf(stderr, " [%u,%u)", lo[q], hi[q]); fprintf(stderr, "\n"); } CK(cudaMemcpy(r.cnt, (void *)d_cnt, 32, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(r.cyc, (void *)d_cyc, 64, cudaMemcpyDeviceToHost)); }
        }
        unsigned c = 0; if (nv_ok) nvmlDeviceGetClockInfo(nd, NVML_CLOCK_SM, &c); if (timed) { r.ms.push_back(ms); r.clk.push_back(c); }
    };
    FILE *jf = fopen(outjson, "wx"); if (!jf) { fprintf(stderr, "refusing to overwrite or cannot write %s\n", outjson); return 2; }
    fprintf(jf, "{\"scene\": \"%s\", \"seed\": %u, \"part\": \"%s\", \"tag\": \"%s\", \"rays\": %u, \"tris\": %u, \"treelets\": %u, \"sm\": %u, \"tpc_queues\": %u, \"wps\": %u, \"rounds\": %d, \"full_gas_bytes\": %zu, \"treelet_gas_total\": %zu, \"treelet_gas_median\": %zu, \"ias_shell_bytes\": %zu, \"ias_real_bytes\": %zu, \"pull\": %.4f,\n \"arms\": [\n",
            scene.c_str(), seed, part.c_str(), tag.c_str(), n, N, C, nsm, nq, WPS, ROUNDS, full.bytes, tl.total, [&] { auto b = tl.bytes; std::sort(b.begin(), b.end()); return b[b.size() / 2]; }(), ias_s.bytes, ias_r.bytes, pull);
    std::thread wd; if (h_hb) wd = std::thread([&]() { for (int sec = 0; sec < 12; ++sec) { std::this_thread::sleep_for(std::chrono::seconds(1)); } unsigned hist2[2][16] = {{0}}; for (unsigned w = 0; w < PW; ++w) { hist2[0][h_hb[8 * w + 1] & 15]++; hist2[1][h_hb[8 * w + 5] & 15]++; } fprintf(stderr, "HB lane0 / lane31 stage histogram:"); for (int l = 0; l < 2; ++l) { for (int i = 0; i < 8; ++i) fprintf(stderr, " %d:%u", i, hist2[l][i]); fprintf(stderr, " |"); } fprintf(stderr, "\n"); int shown = 0; for (unsigned w = 0; w < PW && shown < 8; ++w) if (h_hb[8 * w + 1] != h_hb[8 * w + 5]) { ++shown; fprintf(stderr, "warp %u lane0 stage %u iter %u | lane31 stage %u iter %u\n", w, h_hb[8 * w + 1], h_hb[8 * w], h_hb[8 * w + 5], h_hb[8 * w + 4]); } });
    for (size_t j = 0; j < arms.size(); ++j) run_arm(j, false, true);                                    // untimed first run of every arm, checked
    for (int r = 0; r < ROUNDS; ++r) for (size_t k = 0; k < arms.size(); ++k) { const size_t j = (k + r) % arms.size(); run_arm(j, true, r == ROUNDS - 1); }
    for (size_t j = 0; j < arms.size(); ++j) {
        Res &r = res[j]; std::vector<double> m = r.ms; std::sort(m.begin(), m.end());
        fprintf(jf, "  {\"arm\": \"%s\", \"R\": %d, \"ms\": [", arms[j].name.c_str(), arms[j].R); for (size_t i = 0; i < r.ms.size(); ++i) fprintf(jf, "%s%.4f", i ? "," : "", r.ms[i]);
        fprintf(jf, "], \"clk_mhz\": ["); for (size_t i = 0; i < r.clk.size(); ++i) fprintf(jf, "%s%u", i ? "," : "", r.clk[i]);
        fprintf(jf, "], \"median_ms\": %.4f, \"min_ms\": %.4f, \"bad\": %lu, \"tbad\": %lu, \"ties\": %lu", m.empty() ? 0.0 : m[m.size() / 2], m.empty() ? 0.0 : m[0], r.bad, r.tbad, r.ties);
        if (!r.st.empty()) { fprintf(jf, ", \"stage_median_ms\": ["); for (size_t s = 0; s < r.st.size(); ++s) { std::vector<double> v = r.st[s]; std::sort(v.begin(), v.end()); fprintf(jf, "%s%.4f", s ? "," : "", v[v.size() / 2]); } fprintf(jf, "]"); }
        if (r.has_f) { fprintf(jf, ", \"cnt\": ["); for (int i = 0; i < 8; ++i) fprintf(jf, "%s%u", i ? "," : "", r.cnt[i]); fprintf(jf, "], \"cyc\": ["); for (int i = 0; i < 8; ++i) fprintf(jf, "%s%llu", i ? "," : "", r.cyc[i]); fprintf(jf, "]"); }
        if (r.K) { fprintf(jf, ", \"levels\": %u, \"visits\": %u, \"level_counts\": [", r.K, r.V); for (size_t i = 0; i < r.lvlcnt.size(); ++i) fprintf(jf, "%s%u", i ? "," : "", r.lvlcnt[i]); fprintf(jf, "]"); }
        fprintf(jf, "}%s\n", j + 1 < arms.size() ? "," : "");
    }
    if (wd.joinable()) wd.join(); fprintf(jf, " ]}\n"); fclose(jf); return 0;
}
