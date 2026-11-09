// memdump: run a one triangle OptiX launch, then read ONE window of device
// memory (meant for code addresses) and save it. One window per process, so a
// bad address only kills this process's CUDA context.
//
//   memdump launch [--bvh FILE]                 OptiX launch only (run under ncu to learn addresses)
//   memdump read --method M --addr A --len L --out FILE [--self] [--no-optix]
//       M = attr   : host side pointer queries only, reads nothing
//           copy   : cuMemcpyDtoH from A
//           kernel : a CUDA kernel does plain 32 bit loads from A
//       --self     : A is ignored; use the address of our own device function probe_fn
//                    (a self test: compare against cuobjdump of this binary)
//       --no-optix : skip the OptiX launch
//       --fnrel OFF: A = fnaddr code address + OFF (signed)
//       --fn       : A = code address of our kernel fnaddr, read from CUfunction+0x68 (host memory)
//       --rel OFF  : A = (address of the GAS buffer) + OFF. GPU addresses are randomized per
//                    process, but code sits at a fixed distance from the GAS (seen with ncu)
//
// Never writes device code and never runs TTU instructions of its own. The
// OptiX launch runs NVIDIA's own traversal, same as any OptiX sample.
#include <cuda.h>
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include "memdump_params.h"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_)); exit(2); } } while (0)
#define CU(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char *s_ = 0; cuGetErrorName(r_, &s_); \
    fprintf(stderr, "CU %s at %s:%d: %s\n", #x, __FILE__, __LINE__, s_ ? s_ : "?"); exit(2); } } while (0)
#define OX(x) do { OptixResult r_ = (x); if (r_ != OPTIX_SUCCESS) { \
    fprintf(stderr, "OptiX %s at %s:%d: %s\n", #x, __FILE__, __LINE__, optixGetErrorName(r_)); exit(2); } } while (0)

static const size_t MAX_LEN = 64 * 1024;   // hard cap per process: small windows only

// ---- self test anchor: our own device function -----------------------------
__device__ __noinline__ int probe_fn(int x) { return x * 3 + 0x1234; }
__global__ void fnaddr(unsigned long long *o, int *sink)
{
    int (*f)(int) = probe_fn;
    *o = (unsigned long long)f;
    *sink = f(threadIdx.x);
}
__global__ void readk(const volatile unsigned int *src, unsigned int *dst, size_t n)
{
    for (size_t i = threadIdx.x; i < n; i += blockDim.x) dst[i] = src[i];
}

static unsigned long long probe_addr()
{
    unsigned long long *d, h; int *s;
    CK(cudaMalloc(&d, 8)); CK(cudaMalloc(&s, 4));
    fnaddr<<<1, 1>>>(d, s); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(&h, d, 8, cudaMemcpyDeviceToHost));
    cudaFree(d); cudaFree(s);
    return h;
}

// ---- OptiX: one triangle, one launch ---------------------------------------
struct Optix { OptixDeviceContext ctx; OptixPipeline pipe; CUdeviceptr gas; size_t gas_size; };

template <typename T> struct Rec { __align__(OPTIX_SBT_RECORD_ALIGNMENT) char h[OPTIX_SBT_RECORD_HEADER_SIZE]; T d; };
struct Empty {};

static std::string read_file(const std::string &p)
{
    std::ifstream f(p); if (!f) { fprintf(stderr, "cannot open %s\n", p.c_str()); exit(2); }
    std::stringstream s; s << f.rdbuf(); return s.str();
}

static Optix run_optix(const std::string &ptx_path, const char *bvh_out)
{
    Optix o{};
    CK(cudaFree(0));
    OX(optixInit());
    OptixDeviceContextOptions co{};
    OX(optixDeviceContextCreate(0, &co, &o.ctx));

    // geometry: one triangle in z = 0 covering the middle of the view
    float3 v[3] = {{-0.5f, -0.5f, 0.f}, {0.5f, -0.5f, 0.f}, {0.f, 0.5f, 0.f}};
    CUdeviceptr dv; CK(cudaMalloc((void **)&dv, sizeof v));
    CK(cudaMemcpy((void *)dv, v, sizeof v, cudaMemcpyHostToDevice));
    unsigned int flags = OPTIX_GEOMETRY_FLAG_NONE;
    OptixBuildInput bi{};
    bi.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    bi.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    bi.triangleArray.numVertices = 3;
    bi.triangleArray.vertexBuffers = &dv;
    bi.triangleArray.flags = &flags;
    bi.triangleArray.numSbtRecords = 1;
    OptixAccelBuildOptions ao{}; ao.buildFlags = OPTIX_BUILD_FLAG_NONE; ao.operation = OPTIX_BUILD_OPERATION_BUILD;
    OptixAccelBufferSizes bs{};
    OX(optixAccelComputeMemoryUsage(o.ctx, &ao, &bi, 1, &bs));
    CUdeviceptr tmp; CK(cudaMalloc((void **)&tmp, bs.tempSizeInBytes));
    CK(cudaMalloc((void **)&o.gas, bs.outputSizeInBytes)); o.gas_size = bs.outputSizeInBytes;
    OptixTraversableHandle h;
    OX(optixAccelBuild(o.ctx, 0, &ao, &bi, 1, tmp, bs.tempSizeInBytes, o.gas, bs.outputSizeInBytes, &h, nullptr, 0));
    CK(cudaDeviceSynchronize()); cudaFree((void *)tmp);
    printf("gas 0x%llx size %zu handle 0x%llx\n", (unsigned long long)o.gas, o.gas_size, (unsigned long long)h);
    if (bvh_out) {
        std::vector<char> b(o.gas_size);
        CK(cudaMemcpy(b.data(), (void *)o.gas, o.gas_size, cudaMemcpyDeviceToHost));
        FILE *f = fopen(bvh_out, "wb"); fwrite(b.data(), 1, b.size(), f); fclose(f);
        printf("bvh saved to %s\n", bvh_out);
    }

    // module, program groups, pipeline
    OptixModuleCompileOptions mo{};
    OptixPipelineCompileOptions po{};
    po.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    po.numPayloadValues = 1;
    po.numAttributeValues = 2;
    po.pipelineLaunchParamsVariableName = "params";
    po.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
    std::string ptx = read_file(ptx_path);
    char log[4096]; size_t ls = sizeof log;
    OptixModule m;
    OX(optixModuleCreate(o.ctx, &mo, &po, ptx.c_str(), ptx.size(), log, &ls, &m));
    OptixProgramGroupOptions go{};
    OptixProgramGroupDesc d[3] = {};
    d[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN; d[0].raygen.module = m; d[0].raygen.entryFunctionName = "__raygen__rg";
    d[1].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;   d[1].miss.module = m;   d[1].miss.entryFunctionName = "__miss__ms";
    d[2].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP; d[2].hitgroup.moduleCH = m; d[2].hitgroup.entryFunctionNameCH = "__closesthit__ch";
    OptixProgramGroup g[3];
    ls = sizeof log; OX(optixProgramGroupCreate(o.ctx, d, 3, &go, log, &ls, g));
    OptixPipelineLinkOptions lo{}; lo.maxTraceDepth = 1;
    ls = sizeof log; OX(optixPipelineCreate(o.ctx, &po, &lo, g, 3, log, &ls, &o.pipe));

    // SBT
    Rec<Empty> rg, ms, hg;
    OX(optixSbtRecordPackHeader(g[0], &rg)); OX(optixSbtRecordPackHeader(g[1], &ms)); OX(optixSbtRecordPackHeader(g[2], &hg));
    CUdeviceptr drg, dms, dhg;
    CK(cudaMalloc((void **)&drg, sizeof rg)); CK(cudaMemcpy((void *)drg, &rg, sizeof rg, cudaMemcpyHostToDevice));
    CK(cudaMalloc((void **)&dms, sizeof ms)); CK(cudaMemcpy((void *)dms, &ms, sizeof ms, cudaMemcpyHostToDevice));
    CK(cudaMalloc((void **)&dhg, sizeof hg)); CK(cudaMemcpy((void *)dhg, &hg, sizeof hg, cudaMemcpyHostToDevice));
    OptixShaderBindingTable sbt{};
    sbt.raygenRecord = drg;
    sbt.missRecordBase = dms; sbt.missRecordStrideInBytes = sizeof ms; sbt.missRecordCount = 1;
    sbt.hitgroupRecordBase = dhg; sbt.hitgroupRecordStrideInBytes = sizeof hg; sbt.hitgroupRecordCount = 1;

    // launch 64x64, check that the triangle was hit
    const unsigned W = 64, H = 64;
    Params p{}; p.handle = h; p.w = W; p.h = H;
    CK(cudaMalloc((void **)&p.out, W * H * 4));
    CUdeviceptr dp; CK(cudaMalloc((void **)&dp, sizeof p)); CK(cudaMemcpy((void *)dp, &p, sizeof p, cudaMemcpyHostToDevice));
    OX(optixLaunch(o.pipe, 0, dp, sizeof p, &sbt, W, H, 1));
    CK(cudaDeviceSynchronize());
    std::vector<unsigned> out(W * H);
    CK(cudaMemcpy(out.data(), p.out, W * H * 4, cudaMemcpyDeviceToHost));
    size_t hits = 0; for (unsigned x : out) hits += (x == 1);
    printf("launch ok: %zu / %u rays hit the triangle\n", hits, W * H);
    return o;   // pipeline and buffers stay alive so the code stays loaded
}

// ---- readers ----------------------------------------------------------------
static void attr(unsigned long long a)
{
    unsigned int mt = 0; CUresult r = cuPointerGetAttribute(&mt, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, (CUdeviceptr)a);
    const char *s = 0; cuGetErrorName(r, &s);
    printf("cuPointerGetAttribute(MEMORY_TYPE) = %s, type %u\n", s, mt);
    CUdeviceptr base = 0; size_t sz = 0; r = cuMemGetAddressRange(&base, &sz, (CUdeviceptr)a);
    cuGetErrorName(r, &s);
    printf("cuMemGetAddressRange = %s, base 0x%llx size 0x%zx\n", s, (unsigned long long)base, sz);
}

int main(int argc, char **argv)
{
    if (argc < 2) { fprintf(stderr, "usage: see header of memdump.cu\n"); return 1; }
    std::string mode = argv[1], method = "attr", out, bvh;
    unsigned long long addr = 0, rel = 0; size_t len = 256; bool self = false, optix = true, use_rel = false, fn = false; long long fnrel = 0;
    for (int i = 2; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--method") method = argv[++i];
        else if (a == "--addr") addr = strtoull(argv[++i], 0, 0);
        else if (a == "--len") len = strtoull(argv[++i], 0, 0);
        else if (a == "--out") out = argv[++i];
        else if (a == "--bvh") bvh = argv[++i];
        else if (a == "--rel") { rel = strtoull(argv[++i], 0, 0); use_rel = true; }
        else if (a == "--self") self = true;
        else if (a == "--fn") fn = true;
        else if (a == "--fnrel") { fn = true; fnrel = strtoll(argv[++i], 0, 0); }
        else if (a == "--no-optix") optix = false;
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 1; }
    }
    std::string exe = argv[0], dir = exe.substr(0, exe.find_last_of('/') + 1);

    Optix o{};
    if (optix || mode == "launch") o = run_optix(dir + "memdump_dev.ptx", bvh.empty() ? nullptr : bvh.c_str());
    unsigned long long pa = probe_addr();
    printf("probe_fn at 0x%llx\n", pa);   // printed every run: tells us if the layout is the same run to run
    fflush(stdout);
    if (mode == "launch") return 0;
    if (mode == "funcinfo") {   // host side only: dump the driver's record behind fnaddr's CUfunction
        cudaFunction_t f; CK(cudaGetFuncBySymbol(&f, (const void *)fnaddr));
        const unsigned long long *q = (const unsigned long long *)f;
        printf("CUfunction %p\n", (void *)f);
        for (int i = 0; i < 64; i++) printf("  [%3d] +0x%03x %016llx\n", i, i * 8, q[i]);
        return 0;
    }
    if (mode != "read") { fprintf(stderr, "unknown mode\n"); return 1; }

    if (self) addr = pa;
    if (fn) {   // code address of fnaddr, from field +0x68 of the driver's CUfunction record (driver 580, found with ncu)
        cudaFunction_t f; CK(cudaGetFuncBySymbol(&f, (const void *)fnaddr));
        addr = ((const unsigned long long *)f)[13] + fnrel;
        printf("fnaddr code at 0x%llx (CUfunction+0x68)\n", addr);
    }
    if (use_rel) { if (!o.gas) { fprintf(stderr, "--rel needs the OptiX run\n"); return 1; } addr = (unsigned long long)o.gas + rel; }
    if (!addr || len == 0 || len > MAX_LEN || (len & 3) || (addr & 3)) {
        fprintf(stderr, "need --addr (4 byte aligned) and 0 < --len <= %zu, multiple of 4\n", MAX_LEN); return 1;
    }
    printf("reading 0x%zx bytes at 0x%llx by %s\n", len, addr, method.c_str()); fflush(stdout);
    if (method == "attr") { attr(addr); return 0; }

    std::vector<unsigned char> buf(len);
    if (method == "copy") {
        CU(cuMemcpyDtoH(buf.data(), (CUdeviceptr)addr, len));
    } else if (method == "kernel") {
        unsigned int *d; CK(cudaMalloc(&d, len));
        readk<<<1, 256>>>((const volatile unsigned int *)addr, d, len / 4);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(buf.data(), d, len, cudaMemcpyDeviceToHost));
    } else { fprintf(stderr, "unknown method\n"); return 1; }

    if (!out.empty()) { FILE *f = fopen(out.c_str(), "wb"); fwrite(buf.data(), 1, len, f); fclose(f); }
    for (size_t i = 0; i < len && i < 64; i += 16) {
        unsigned long long lo, hi; memcpy(&lo, &buf[i], 8); memcpy(&hi, &buf[i + 8], 8);
        printf("  +%04zx %016llx%016llx\n", i, hi, lo);
    }
    printf("read ok\n");
    return 0;
}
