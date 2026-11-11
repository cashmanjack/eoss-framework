// Morton / SER test host: OptiX only, no TTU code. Closest hit of N rays (file order = launch order), timed with CUDA events.
//   morton PTX SOUP_N RAYS.bin OUT_PREFIX REPEAT [--dump-gas F --dump-tris F] MODE...
//   MODE: plain | serpost | ser:HINTS.bin:BITS     (hints: one uint32 per ray, BITS low bits used)
// Per mode: REPEAT timed launches (after one untimed), results of the last launch to OUT_PREFIX.<k>.bin (RefOut per ray), JSON line on stdout.
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
#include "morton_params.h"
#include <cub/device/device_radix_sort.cuh>
// ---- GPU sort path (net Morton speedup): key generation, CUB radix sort, ray gather, result scatter ----
__device__ static unsigned spread3(unsigned x[3], int k) { unsigned c = 0; for (int b = 0; b < k; ++b) for (int a = 0; a < 3; ++a) c |= ((x[a] >> b) & 1u) << (3 * b + a); return c; }
__device__ static unsigned quant(float v, float lo, float hi, int k) { if (!(hi > lo) || k == 0) return 0; float f = (v - lo) / (hi - lo) * (float)(1 << k); int q = (int)f; return (unsigned)(q < 0 ? 0 : (q > (1 << k) - 1 ? (1 << k) - 1 : q)); }
struct Bounds { float olo[3], ohi[3], dlo[3], dhi[3]; };
__global__ void keygen(const Ray8 *r, unsigned *key, unsigned *idx, unsigned n, Bounds b, int ko, int kd, int dmajor)
{
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    Ray8 x = r[i]; unsigned qo[3], qd[3];
    for (int a = 0; a < 3; ++a) { qo[a] = quant(x.o[a], b.olo[a], b.ohi[a], ko); qd[a] = quant(x.d[a], b.dlo[a], b.dhi[a], kd); }
    unsigned co = spread3(qo, ko), cd = spread3(qd, kd);
    key[i] = dmajor ? ((cd << (3 * ko)) | co) : ((co << (3 * kd)) | cd); idx[i] = i;
}
__global__ void gather(const Ray8 *src, const unsigned *idx, Ray8 *dst, unsigned n) { unsigned i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) dst[i] = src[idx[i]]; }
__global__ void scatter(const RefOut *src, const unsigned *idx, RefOut *dst, unsigned n) { unsigned i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) dst[idx[i]] = src[i]; }
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(2); } } while (0)
#define OX(x) do { OptixResult r_ = (x); if (r_ != OPTIX_SUCCESS) { fprintf(stderr, "%s:%d %s: %d\n", __FILE__, __LINE__, #x, (int)r_); exit(2); } } while (0)
struct Tri { float3 a, b, c; };
static std::string read_file(const char *p) { std::ifstream f(p); if (!f) { fprintf(stderr, "cannot read %s\n", p); exit(2); } std::stringstream s; s << f.rdbuf(); return s.str(); }
template <class T> static std::vector<T> read_bin(const char *p) { FILE *f = fopen(p, "rb"); if (!f) { fprintf(stderr, "cannot read %s\n", p); exit(2); } std::vector<T> v; T x; while (fread(&x, sizeof x, 1, f) == 1) v.push_back(x); fclose(f); return v; }

int main(int argc, char **argv)
{
    if (argc < 7) { fprintf(stderr, "usage\n"); return 2; }
    const char *dump_gas = nullptr, *dump_tris = nullptr; std::vector<std::string> modes;
    const int n_tri = atoi(argv[2]); const std::string outp = argv[4]; const int repeat = atoi(argv[5]);
    for (int a = 6; a < argc; ++a) {
        if (!strcmp(argv[a], "--dump-gas") && a + 1 < argc) dump_gas = argv[++a];
        else if (!strcmp(argv[a], "--dump-tris") && a + 1 < argc) dump_tris = argv[++a];
        else modes.push_back(argv[a]);
    }
    auto rays = read_bin<Ray8>(argv[3]); const unsigned NR = (unsigned)rays.size();
    CK(cudaFree(0)); OX(optixInit());
    OptixDeviceContextOptions co{}; OptixDeviceContext ctx; OX(optixDeviceContextCreate(0, &co, &ctx));
    // scene: same generator as setup3.cu soup:N (edge 3 / cbrt(N), seed 1234)
    const float e = 3.f / std::cbrt((float)n_tri);
    std::mt19937 g(1234); std::uniform_real_distribution<float> U(-1.f, 1.f);
    std::vector<Tri> v; v.reserve(n_tri);
    for (int i = 0; i < n_tri; ++i) {
        const float3 c{U(g), U(g), U(g)};
        auto pt = [&]() { return float3{c.x + 0.5f * e * U(g), c.y + 0.5f * e * U(g), c.z + 0.5f * e * U(g)}; };
        v.push_back({pt(), pt(), pt()});
    }
    if (dump_tris) { FILE *f = fopen(dump_tris, "wb"); fwrite(v.data(), sizeof(Tri), v.size(), f); fclose(f); }
    CUdeviceptr dv; CK(cudaMalloc((void **)&dv, v.size() * sizeof(Tri))); CK(cudaMemcpy((void *)dv, v.data(), v.size() * sizeof(Tri), cudaMemcpyHostToDevice));
    unsigned gflags = OPTIX_GEOMETRY_FLAG_NONE; OptixBuildInput bi{}; bi.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    bi.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3; bi.triangleArray.numVertices = (unsigned)v.size() * 3; bi.triangleArray.vertexBuffers = &dv;
    bi.triangleArray.flags = &gflags; bi.triangleArray.numSbtRecords = 1;
    OptixAccelBuildOptions ao{}; ao.buildFlags = OPTIX_BUILD_FLAG_NONE; ao.operation = OPTIX_BUILD_OPERATION_BUILD;
    OptixAccelBufferSizes bs{}; OX(optixAccelComputeMemoryUsage(ctx, &ao, &bi, 1, &bs));
    CUdeviceptr tmp, gas; CK(cudaMalloc((void **)&tmp, bs.tempSizeInBytes)); CK(cudaMalloc((void **)&gas, bs.outputSizeInBytes));
    OptixTraversableHandle h; OX(optixAccelBuild(ctx, 0, &ao, &bi, 1, tmp, bs.tempSizeInBytes, gas, bs.outputSizeInBytes, &h, nullptr, 0));
    CK(cudaDeviceSynchronize()); CK(cudaFree((void *)tmp));
    if (dump_gas) { std::vector<char> b(bs.outputSizeInBytes); CK(cudaMemcpy(b.data(), (void *)gas, b.size(), cudaMemcpyDeviceToHost)); FILE *f = fopen(dump_gas, "wb"); fwrite(b.data(), 1, b.size(), f); fclose(f); }
    // pipeline with three raygens
    OptixModuleCompileOptions mo{}; OptixPipelineCompileOptions po{};
    po.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS; po.numPayloadValues = 7; po.numAttributeValues = 2;
    po.pipelineLaunchParamsVariableName = "params"; po.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
    std::string ptx = read_file(argv[1]); char log[4096]; size_t ls = sizeof log; OptixModule m;
    OX(optixModuleCreate(ctx, &mo, &po, ptx.c_str(), ptx.size(), log, &ls, &m));
    OptixProgramGroupOptions go{}; OptixProgramGroupDesc d[5] = {};
    const char *rgn[3] = {"__raygen__plain", "__raygen__ser", "__raygen__serpost"};
    for (int k = 0; k < 3; ++k) { d[k].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN; d[k].raygen.module = m; d[k].raygen.entryFunctionName = rgn[k]; }
    d[3].kind = OPTIX_PROGRAM_GROUP_KIND_MISS; d[3].miss.module = m; d[3].miss.entryFunctionName = "__miss__ms";
    d[4].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP; d[4].hitgroup.moduleCH = m; d[4].hitgroup.entryFunctionNameCH = "__closesthit__ch";
    OptixProgramGroup pg[5]; ls = sizeof log; OX(optixProgramGroupCreate(ctx, d, 5, &go, log, &ls, pg));
    OptixPipelineLinkOptions lo{}; lo.maxTraceDepth = 1; OptixPipeline pipe; ls = sizeof log; OX(optixPipelineCreate(ctx, &po, &lo, pg, 5, log, &ls, &pipe));
    struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) Rec { char h[OPTIX_SBT_RECORD_HEADER_SIZE]; };
    Rec rec[5]; for (int k = 0; k < 5; ++k) OX(optixSbtRecordPackHeader(pg[k], &rec[k]));
    CUdeviceptr drec; CK(cudaMalloc((void **)&drec, sizeof rec)); CK(cudaMemcpy((void *)drec, rec, sizeof rec, cudaMemcpyHostToDevice));
    Params prm{}; prm.handle = h; prm.n = NR;
    CK(cudaMalloc((void **)&prm.rays, NR * sizeof(Ray8))); CK(cudaMemcpy((void *)prm.rays, rays.data(), NR * sizeof(Ray8), cudaMemcpyHostToDevice));
    CK(cudaMalloc((void **)&prm.out, NR * sizeof(RefOut))); CK(cudaMalloc((void **)&prm.hints, NR * 4));
    CUdeviceptr dprm; CK(cudaMalloc((void **)&dprm, sizeof prm));
    cudaEvent_t a0, a1; CK(cudaEventCreate(&a0)); CK(cudaEventCreate(&a1));
    for (size_t mi = 0; mi < modes.size(); ++mi) {
        const std::string &md = modes[mi]; int rg = 0;
        if (md.rfind("gsort:", 0) == 0) {   // gsort:KO:KD:o|d  (key = origin code major / direction code major)
            int ko, kd; char mj; if (sscanf(md.c_str(), "gsort:%d:%d:%c", &ko, &kd, &mj) != 3 || 3 * (ko + kd) > 32) { fprintf(stderr, "gsort mode\n"); return 2; }
            Bounds bd; for (int a = 0; a < 3; ++a) { bd.olo[a] = bd.dlo[a] = 1e30f; bd.ohi[a] = bd.dhi[a] = -1e30f; }
            for (auto &r : rays) for (int a = 0; a < 3; ++a) { bd.olo[a] = fminf(bd.olo[a], r.o[a]); bd.ohi[a] = fmaxf(bd.ohi[a], r.o[a]); bd.dlo[a] = fminf(bd.dlo[a], r.d[a]); bd.dhi[a] = fmaxf(bd.dhi[a], r.d[a]); }
            const int bits = 3 * (ko + kd); const unsigned thr = 256, blk = (NR + thr - 1) / thr;
            Ray8 *rs; RefOut *os; unsigned *k0, *k1, *i0, *i1; void *tmpb = nullptr; size_t tb = 0;
            CK(cudaMalloc((void **)&rs, NR * sizeof(Ray8))); CK(cudaMalloc((void **)&os, NR * sizeof(RefOut)));
            CK(cudaMalloc((void **)&k0, NR * 4)); CK(cudaMalloc((void **)&k1, NR * 4)); CK(cudaMalloc((void **)&i0, NR * 4)); CK(cudaMalloc((void **)&i1, NR * 4));
            cub::DeviceRadixSort::SortPairs(nullptr, tb, k0, k1, i0, i1, (int)NR, 0, bits); CK(cudaMalloc(&tmpb, tb));
            const Ray8 *orig = prm.rays; RefOut *origout = prm.out;
            Params pg = prm; pg.rays = rs; pg.out = os;
            CK(cudaMemcpy((void *)dprm, &pg, sizeof pg, cudaMemcpyHostToDevice));
            OptixShaderBindingTable sbt{}; sbt.raygenRecord = drec;
            sbt.missRecordBase = drec + 3 * sizeof(Rec); sbt.missRecordStrideInBytes = sizeof(Rec); sbt.missRecordCount = 1;
            sbt.hitgroupRecordBase = drec + 4 * sizeof(Rec); sbt.hitgroupRecordStrideInBytes = sizeof(Rec); sbt.hitgroupRecordCount = 1;
            cudaEvent_t e[6]; for (auto &x : e) CK(cudaEventCreate(&x));
            std::vector<float> tk, ts, tg, tt, tc;
            for (int k = -1; k < repeat; ++k) {
                CK(cudaEventRecord(e[0]));
                keygen<<<blk, thr>>>(orig, k0, i0, NR, bd, ko, kd, mj == 'd'); CK(cudaEventRecord(e[1]));
                cub::DeviceRadixSort::SortPairs(tmpb, tb, k0, k1, i0, i1, (int)NR, 0, bits); CK(cudaEventRecord(e[2]));
                gather<<<blk, thr>>>(orig, i1, rs, NR); CK(cudaEventRecord(e[3]));
                OX(optixLaunch(pipe, 0, dprm, sizeof pg, &sbt, NR, 1, 1)); CK(cudaEventRecord(e[4]));
                scatter<<<blk, thr>>>(os, i1, origout, NR); CK(cudaEventRecord(e[5])); CK(cudaEventSynchronize(e[5]));
                if (k < 0) continue; float x;
                CK(cudaEventElapsedTime(&x, e[0], e[1])); tk.push_back(x); CK(cudaEventElapsedTime(&x, e[1], e[2])); ts.push_back(x);
                CK(cudaEventElapsedTime(&x, e[2], e[3])); tg.push_back(x); CK(cudaEventElapsedTime(&x, e[3], e[4])); tt.push_back(x); CK(cudaEventElapsedTime(&x, e[4], e[5])); tc.push_back(x);
            }
            std::vector<RefOut> out(NR); CK(cudaMemcpy(out.data(), origout, NR * sizeof(RefOut), cudaMemcpyDeviceToHost));
            FILE *f = fopen((outp + "." + std::to_string(mi) + ".bin").c_str(), "wb"); fwrite(out.data(), sizeof(RefOut), NR, f); fclose(f);
            auto pr = [](const char *n, std::vector<float> &v) { printf("\"%s\":[", n); for (size_t q = 0; q < v.size(); ++q) printf("%s%.5f", q ? "," : "", v[q]); printf("]"); };
            printf("{\"mode\":\"%s\",\"rays\":%u,", md.c_str(), NR); pr("keygen", tk); printf(","); pr("sort", ts); printf(","); pr("gather", tg); printf(","); pr("ms", tt); printf(","); pr("scatter", tc); printf("}\n"); fflush(stdout);
            for (auto &x : e) cudaEventDestroy(x); cudaFree(rs); cudaFree(os); cudaFree(k0); cudaFree(k1); cudaFree(i0); cudaFree(i1); cudaFree(tmpb);
            continue;
        }
        if (md == "plain") rg = 0; else if (md == "serpost") rg = 2;
        else if (md.rfind("ser:", 0) == 0) {
            rg = 1; const size_t c = md.rfind(':'); const std::string hf = md.substr(4, c - 4); prm.bits = (unsigned)atoi(md.c_str() + c + 1);
            auto hv = read_bin<unsigned>(hf.c_str()); if (hv.size() != NR) { fprintf(stderr, "hints size\n"); return 2; }
            CK(cudaMemcpy((void *)prm.hints, hv.data(), NR * 4, cudaMemcpyHostToDevice));
        } else { fprintf(stderr, "mode %s\n", md.c_str()); return 2; }
        CK(cudaMemcpy((void *)dprm, &prm, sizeof prm, cudaMemcpyHostToDevice));
        OptixShaderBindingTable sbt{}; sbt.raygenRecord = drec + rg * sizeof(Rec);
        sbt.missRecordBase = drec + 3 * sizeof(Rec); sbt.missRecordStrideInBytes = sizeof(Rec); sbt.missRecordCount = 1;
        sbt.hitgroupRecordBase = drec + 4 * sizeof(Rec); sbt.hitgroupRecordStrideInBytes = sizeof(Rec); sbt.hitgroupRecordCount = 1;
        OX(optixLaunch(pipe, 0, dprm, sizeof prm, &sbt, NR, 1, 1)); CK(cudaDeviceSynchronize());
        std::vector<float> ms;
        for (int k = 0; k < repeat; ++k) { CK(cudaEventRecord(a0)); OX(optixLaunch(pipe, 0, dprm, sizeof prm, &sbt, NR, 1, 1)); CK(cudaEventRecord(a1)); CK(cudaEventSynchronize(a1)); float x; CK(cudaEventElapsedTime(&x, a0, a1)); ms.push_back(x); }
        std::vector<RefOut> out(NR); CK(cudaMemcpy(out.data(), prm.out, NR * sizeof(RefOut), cudaMemcpyDeviceToHost));
        FILE *f = fopen((outp + "." + std::to_string(mi) + ".bin").c_str(), "wb"); fwrite(out.data(), sizeof(RefOut), NR, f); fclose(f);
        printf("{\"mode\":\"%s\",\"rays\":%u,\"ms\":[", md.c_str(), NR); for (size_t k = 0; k < ms.size(); ++k) printf("%s%.5f", k ? "," : "", ms[k]); printf("]}\n"); fflush(stdout);
    }
    return 0;
}
