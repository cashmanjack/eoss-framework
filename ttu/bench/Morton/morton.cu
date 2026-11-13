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
#include <nvml.h>
#include <chrono>
#include <thread>
static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
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
__global__ void shade_kernel(const Ray8 *in, const RefOut *h, const float *tris, Ray8 *nb, Ray8 *sh, float4 *acc, unsigned n)
{
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    const RefOut o = h[i]; const Ray8 r = in[i]; Ray8 a{}, b{}; float4 c = make_float4(0, 0, 0, 0);
    if (o.hit) {
        const float *t = tris + 9ull * o.prim; const float ux = t[3] - t[0], uy = t[4] - t[1], uz = t[5] - t[2], vx = t[6] - t[0], vy = t[7] - t[1], vz = t[8] - t[2];
        float nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx; const float il = rsqrtf(nx * nx + ny * ny + nz * nz + 1e-30f); nx *= il; ny *= il; nz *= il;
        if (nx * r.d[0] + ny * r.d[1] + nz * r.d[2] > 0.f) { nx = -nx; ny = -ny; nz = -nz; }
        unsigned s = i * 747796405u + 2891336453u; s = ((s >> ((s >> 28) + 4)) ^ s) * 277803737u; const float u1 = (s >> 8) * (1.f / 16777216.f); s = s * 747796405u + 2891336453u; s = ((s >> ((s >> 28) + 4)) ^ s) * 277803737u; const float u2 = (s >> 8) * (1.f / 16777216.f);
        const float rr = sqrtf(u1), ph = 6.2831853f * u2, x = rr * cosf(ph), y = rr * sinf(ph), z = sqrtf(fmaxf(0.f, 1.f - u1));
        const float ax = fabsf(nx) > 0.9f ? 0.f : 1.f, ay = fabsf(nx) > 0.9f ? 1.f : 0.f; float tx = ay * nz, ty = -ax * nz, tz = ax * ny - ay * nx; const float it = rsqrtf(tx * tx + ty * ty + tz * tz + 1e-30f); tx *= it; ty *= it; tz *= it;
        const float bx = ny * tz - nz * ty, by = nz * tx - nx * tz, bz = nx * ty - ny * tx;
        const float p0 = r.o[0] + o.t * r.d[0], p1 = r.o[1] + o.t * r.d[1], p2 = r.o[2] + o.t * r.d[2];
        a.o[0] = p0 + 1e-4f * nx; a.o[1] = p1 + 1e-4f * ny; a.o[2] = p2 + 1e-4f * nz; a.d[0] = x * tx + y * bx + z * nx; a.d[1] = x * ty + y * by + z * ny; a.d[2] = x * tz + y * bz + z * nz; a.tmax = 1e16f;
        b = a; b.d[0] = 0.3f; b.d[1] = 0.8f; b.d[2] = 0.5f;
        c = make_float4(0.5f * (nx + 1.f), 0.5f * (ny + 1.f), 0.5f * (nz + 1.f), 1.f);
    }
    nb[i] = a; sh[i] = b; acc[i] = c;
}
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(2); } } while (0)
#define OX(x) do { OptixResult r_ = (x); if (r_ != OPTIX_SUCCESS) { fprintf(stderr, "%s:%d %s: %d\n", __FILE__, __LINE__, #x, (int)r_); exit(2); } } while (0)

// ---- energy windows (NVML total energy counter, 100 ms updates): the loop runs short launches back to back, synchronizing every 20 ----
struct Egy { nvmlDevice_t nd; bool ok = false; unsigned long long e0 = 0; double t0 = 0; unsigned long long clk_sum = 0; int clk_n = 0; long long nl = 0;
    void begin() { char bus[32]; CK(cudaDeviceGetPCIBusId(bus, sizeof bus, 0));
        if (nvmlInit_v2() != NVML_SUCCESS || nvmlDeviceGetHandleByPciBusId_v2(bus, &nd) != NVML_SUCCESS) { fprintf(stderr, "NVML init failed\n"); exit(3); }
        CK(cudaDeviceSynchronize()); if (nvmlDeviceGetTotalEnergyConsumption(nd, &e0) != NVML_SUCCESS) { fprintf(stderr, "NVML energy unavailable\n"); exit(3); } t0 = now_s(); clk_sum = 0; clk_n = 0; nl = 0; }
    void sample() { unsigned c = 0; nvmlDeviceGetClockInfo(nd, NVML_CLOCK_SM, &c); clk_sum += c; ++clk_n; }
    void end(const char *name, unsigned rays) { CK(cudaDeviceSynchronize()); const double t1 = now_s(); unsigned long long e1 = 0; unsigned smc = 0, mc = 0, pw = 0;
        nvmlDeviceGetTotalEnergyConsumption(nd, &e1); nvmlDeviceGetClockInfo(nd, NVML_CLOCK_SM, &smc); nvmlDeviceGetClockInfo(nd, NVML_CLOCK_MEM, &mc); nvmlDeviceGetPowerUsage(nd, &pw);
        printf("{\"mode\":\"%s\",\"rays\":%u,\"mJ\":%llu,\"s\":%.6f,\"launches\":%lld,\"sm_mhz_mean\":%.1f,\"sm_mhz_end\":%u,\"mem_mhz\":%u,\"power_mW_end\":%u}\n", name, rays, e1 - e0, t1 - t0, nl, clk_n ? (double)clk_sum / clk_n : 0.0, smc, mc, pw); fflush(stdout); nvmlShutdown(); }
};
struct Tri { float3 a, b, c; };
static std::string read_file(const char *p) { std::ifstream f(p); if (!f) { fprintf(stderr, "cannot read %s\n", p); exit(2); } std::stringstream s; s << f.rdbuf(); return s.str(); }
template <class T> static std::vector<T> read_bin(const char *p) { FILE *f = fopen(p, "rb"); if (!f) { fprintf(stderr, "cannot read %s\n", p); exit(2); } std::vector<T> v; T x; while (fread(&x, sizeof x, 1, f) == 1) v.push_back(x); fclose(f); return v; }

int main(int argc, char **argv)
{
    if (argc < 7) { fprintf(stderr, "usage\n"); return 2; }
    const char *dump_gas = nullptr, *dump_tris = nullptr; std::vector<std::string> modes;
    const char *scene_file = (!strncmp(argv[2], "file:", 5)) ? argv[2] + 5 : nullptr; const char *info_file = nullptr;
    const int n_tri = scene_file ? 0 : atoi(argv[2]); const std::string outp = argv[4]; const int repeat = atoi(argv[5]);
    for (int a = 6; a < argc; ++a) {
        if (!strcmp(argv[a], "--dump-gas") && a + 1 < argc) dump_gas = argv[++a];
        else if (!strcmp(argv[a], "--dump-tris") && a + 1 < argc) dump_tris = argv[++a];
        else if (!strcmp(argv[a], "--info") && a + 1 < argc) info_file = argv[++a];
        else modes.push_back(argv[a]);
    }
    auto rays = read_bin<Ray8>(argv[3]); const unsigned NR = (unsigned)rays.size();
    CK(cudaFree(0)); OX(optixInit());
    OptixDeviceContextOptions co{}; OptixDeviceContext ctx; OX(optixDeviceContextCreate(0, &co, &ctx));
    // scene: same generator as setup3.cu soup:N (edge 3 / cbrt(N), seed 1234)
    const float e = 3.f / std::cbrt((float)n_tri);
    std::mt19937 g(1234); std::uniform_real_distribution<float> U(-1.f, 1.f);
    std::vector<Tri> v; v.reserve(n_tri);
    if (scene_file) { v = read_bin<Tri>(scene_file); fprintf(stderr, "scene file %s: %zu triangles\n", scene_file, v.size()); }
    for (int i = 0; i < (scene_file ? 0 : n_tri); ++i) {
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
    if (info_file) { FILE *f = fopen(info_file, "w"); fprintf(f, "{\"triangles\": %zu, \"gas_bytes\": %zu, \"temp_bytes\": %zu}\n", v.size(), bs.outputSizeInBytes, bs.tempSizeInBytes); fclose(f); }
    if (dump_gas) { std::vector<char> b(bs.outputSizeInBytes); CK(cudaMemcpy(b.data(), (void *)gas, b.size(), cudaMemcpyDeviceToHost)); FILE *f = fopen(dump_gas, "wb"); fwrite(b.data(), 1, b.size(), f); fclose(f); }
    // pipeline with three raygens
    OptixModuleCompileOptions mo{}; OptixPipelineCompileOptions po{};
    po.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS; po.numPayloadValues = 7; po.numAttributeValues = 2;
    po.pipelineLaunchParamsVariableName = "params"; po.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
    std::string ptx = read_file(argv[1]); char log[4096]; size_t ls = sizeof log; OptixModule m;
    OX(optixModuleCreate(ctx, &mo, &po, ptx.c_str(), ptx.size(), log, &ls, &m));
    OptixProgramGroupOptions go{}; OptixProgramGroupDesc d[6] = {};
    const char *rgn[3] = {"__raygen__plain", "__raygen__ser", "__raygen__serpost"};
    for (int k = 0; k < 3; ++k) { d[k].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN; d[k].raygen.module = m; d[k].raygen.entryFunctionName = rgn[k]; }
    d[3].kind = OPTIX_PROGRAM_GROUP_KIND_MISS; d[3].miss.module = m; d[3].miss.entryFunctionName = "__miss__ms";
    d[4].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP; d[4].hitgroup.moduleCH = m; d[4].hitgroup.entryFunctionNameCH = "__closesthit__ch";
    d[5].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN; d[5].raygen.module = m; d[5].raygen.entryFunctionName = "__raygen__any";
    OptixProgramGroup pg[6]; ls = sizeof log; OX(optixProgramGroupCreate(ctx, d, 6, &go, log, &ls, pg));
    OptixPipelineLinkOptions lo{}; lo.maxTraceDepth = 1; OptixPipeline pipe; ls = sizeof log; OX(optixPipelineCreate(ctx, &po, &lo, pg, 6, log, &ls, &pipe));
    struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) Rec { char h[OPTIX_SBT_RECORD_HEADER_SIZE]; };
    Rec rec[6]; for (int k = 0; k < 6; ++k) OX(optixSbtRecordPackHeader(pg[k], &rec[k]));
    CUdeviceptr drec; CK(cudaMalloc((void **)&drec, sizeof rec)); CK(cudaMemcpy((void *)drec, rec, sizeof rec, cudaMemcpyHostToDevice));
    Params prm{}; prm.handle = h; prm.n = NR;
    CK(cudaMalloc((void **)&prm.rays, NR * sizeof(Ray8))); CK(cudaMemcpy((void *)prm.rays, rays.data(), NR * sizeof(Ray8), cudaMemcpyHostToDevice));
    CK(cudaMalloc((void **)&prm.out, NR * sizeof(RefOut))); CK(cudaMalloc((void **)&prm.hints, NR * 4));
    CUdeviceptr dprm; CK(cudaMalloc((void **)&dprm, sizeof prm));
    cudaEvent_t a0, a1; CK(cudaEventCreate(&a0)); CK(cudaEventCreate(&a1));
    for (size_t mi = 0; mi < modes.size(); ++mi) {
        const std::string &md = modes[mi]; int rg = 0;
        if (md.rfind("egy", 0) == 0 && md.rfind("egysort", 0) != 0) {   // egyidle:S (nothing runs) | egy:S (plain trace of the loaded rays, S seconds; results of the last launch to OUT.k.bin)
            double S = atof(md.c_str() + md.find(':') + 1); Egy ey; ey.begin();
            if (md.rfind("egyidle", 0) == 0) { while (now_s() - ey.t0 < S) { std::this_thread::sleep_for(std::chrono::milliseconds(100)); if (++ey.nl % 10 == 0) ey.sample(); } ey.nl = 0; ey.end("egyidle", 0); continue; }
            CK(cudaMemcpy((void *)dprm, &prm, sizeof prm, cudaMemcpyHostToDevice));
            OptixShaderBindingTable sbt{}; sbt.raygenRecord = drec;
            sbt.missRecordBase = drec + 3 * sizeof(Rec); sbt.missRecordStrideInBytes = sizeof(Rec); sbt.missRecordCount = 1;
            sbt.hitgroupRecordBase = drec + 4 * sizeof(Rec); sbt.hitgroupRecordStrideInBytes = sizeof(Rec); sbt.hitgroupRecordCount = 1;
            OX(optixLaunch(pipe, 0, dprm, sizeof prm, &sbt, NR, 1, 1)); CK(cudaDeviceSynchronize()); ey.begin();
            while (now_s() - ey.t0 < S) { for (int q = 0; q < 20; ++q) { OX(optixLaunch(pipe, 0, dprm, sizeof prm, &sbt, NR, 1, 1)); ++ey.nl; } CK(cudaDeviceSynchronize()); if (ey.nl % 200 == 0) ey.sample(); }
            ey.end("egy", NR);
            std::vector<RefOut> out(NR); CK(cudaMemcpy(out.data(), prm.out, NR * sizeof(RefOut), cudaMemcpyDeviceToHost));
            FILE *f = fopen((outp + "." + std::to_string(mi) + ".bin").c_str(), "wb"); fwrite(out.data(), sizeof(RefOut), NR, f); fclose(f); continue;
        }
        double egy_s = 0; std::string mdg = md;
        if (md.rfind("egysort:", 0) == 0) { egy_s = atof(md.c_str() + 8); mdg = "gsort:" + md.substr(md.find(':', 8) + 1); }   // egysort:S:KO:KD:o  = energy window over the whole GPU sort pipeline
        if (mdg.rfind("gsort:", 0) == 0) {   // gsort:KO:KD:o|d  (key = origin code major / direction code minor)
            int ko, kd; char mj; if (sscanf(mdg.c_str(), "gsort:%d:%d:%c", &ko, &kd, &mj) != 3 || 3 * (ko + kd) > 32) { fprintf(stderr, "gsort mode\n"); return 2; }
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
            if (egy_s > 0) {   // energy window: keygen, sort, gather, trace, scatter back to back
                auto pipe_once = [&]() { keygen<<<blk, thr>>>(orig, k0, i0, NR, bd, ko, kd, mj == 'd'); cub::DeviceRadixSort::SortPairs(tmpb, tb, k0, k1, i0, i1, (int)NR, 0, bits); gather<<<blk, thr>>>(orig, i1, rs, NR);
                    OX(optixLaunch(pipe, 0, dprm, sizeof pg, &sbt, NR, 1, 1)); scatter<<<blk, thr>>>(os, i1, origout, NR); };
                pipe_once(); CK(cudaDeviceSynchronize()); Egy ey; ey.begin();
                while (now_s() - ey.t0 < egy_s) { for (int q = 0; q < 20; ++q) { pipe_once(); ++ey.nl; } CK(cudaDeviceSynchronize()); if (ey.nl % 200 == 0) ey.sample(); }
                ey.end(md.c_str(), NR);
                std::vector<RefOut> out(NR); CK(cudaMemcpy(out.data(), origout, NR * sizeof(RefOut), cudaMemcpyDeviceToHost));
                FILE *f = fopen((outp + "." + std::to_string(mi) + ".bin").c_str(), "wb"); fwrite(out.data(), sizeof(RefOut), NR, f); fclose(f);
                for (auto &x : e) cudaEventDestroy(x); cudaFree(rs); cudaFree(os); cudaFree(k0); cudaFree(k1); cudaFree(i0); cudaFree(i1); cudaFree(tmpb); continue;
            }
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
        if (md == "shade") {   // minimal shading stand in on the hits of the last plain/any mode in prm.out: tri fetch, normal, cosine bounce + shadow ray + accumulator write; timed only
            Ray8 *nb, *sh; float4 *acc; CK(cudaMalloc((void **)&nb, NR * sizeof(Ray8))); CK(cudaMalloc((void **)&sh, NR * sizeof(Ray8))); CK(cudaMalloc((void **)&acc, NR * sizeof(float4)));
            const unsigned thr = 256, blk = (NR + thr - 1) / thr; std::vector<float> ms;
            for (int k = -1; k < repeat; ++k) { CK(cudaEventRecord(a0)); shade_kernel<<<blk, thr>>>(prm.rays, prm.out, (const float *)dv, nb, sh, acc, NR); CK(cudaEventRecord(a1)); CK(cudaEventSynchronize(a1)); float x; CK(cudaEventElapsedTime(&x, a0, a1)); if (k >= 0) ms.push_back(x); }
            printf("{\"mode\":\"shade\",\"rays\":%u,\"ms\":[", NR); for (size_t k = 0; k < ms.size(); ++k) printf("%s%.5f", k ? "," : "", ms[k]); printf("]}\n"); fflush(stdout);
            cudaFree(nb); cudaFree(sh); cudaFree(acc); continue;
        }
        if (md == "plain") rg = 0; else if (md == "serpost") rg = 2; else if (md == "any") rg = 5;
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
