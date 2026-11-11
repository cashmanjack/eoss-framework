// BVH_4 host (OptiX only, no TTU code): soup:N GAS, an IAS of K instances with real transforms, closest hit of rays through the IAS.
//   ias PTX SOUP_N K MODE RAYS.bin OUT_PREFIX         MODE: id (identity transforms) | trans (translations) | rot (rotation about z + uniform scale + translation)
// Writes OUT_PREFIX.{out,gas,ias,tris,xf,c3}.bin and one JSON line; rays: Ray8 file (32 bytes each).
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
#include "ias_params.h"
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(2); } } while (0)
#define OX(x) do { OptixResult r_ = (x); if (r_ != OPTIX_SUCCESS) { fprintf(stderr, "%s:%d %s: %d\n", __FILE__, __LINE__, #x, (int)r_); exit(2); } } while (0)
struct Tri { float3 a, b, c; };
static std::string read_file(const char *p) { std::ifstream f(p); if (!f) { fprintf(stderr, "cannot read %s\n", p); exit(2); } std::stringstream s; s << f.rdbuf(); return s.str(); }
static void wr(const std::string &p, const void *d, size_t n) { FILE *f = fopen(p.c_str(), "wb"); fwrite(d, 1, n, f); fclose(f); }
int main(int argc, char **argv)
{
    if (argc < 7) { fprintf(stderr, "usage\n"); return 2; }
    const int n_tri = atoi(argv[2]), K = atoi(argv[3]); const std::string mode = argv[4], pre = argv[6];
    std::vector<Ray8> rays; { FILE *f = fopen(argv[5], "rb"); if (!f) { fprintf(stderr, "rays\n"); return 2; } Ray8 y; while (fread(&y, sizeof y, 1, f) == 1) rays.push_back(y); fclose(f); }
    const unsigned NR = (unsigned)rays.size();
    CK(cudaFree(0)); OX(optixInit()); OptixDeviceContextOptions co{}; OptixDeviceContext ctx; OX(optixDeviceContextCreate(0, &co, &ctx));
    const float e = 3.f / std::cbrt((float)n_tri); std::mt19937 g(1234); std::uniform_real_distribution<float> U(-1.f, 1.f);   // same soup as setup3 / morton
    std::vector<Tri> v; for (int i = 0; i < n_tri; ++i) { const float3 c{U(g), U(g), U(g)}; auto pt = [&]() { return float3{c.x + 0.5f * e * U(g), c.y + 0.5f * e * U(g), c.z + 0.5f * e * U(g)}; }; v.push_back({pt(), pt(), pt()}); }
    wr(pre + ".tris.bin", v.data(), v.size() * sizeof(Tri));
    CUdeviceptr dv; CK(cudaMalloc((void **)&dv, v.size() * sizeof(Tri))); CK(cudaMemcpy((void *)dv, v.data(), v.size() * sizeof(Tri), cudaMemcpyHostToDevice));
    unsigned gflags = OPTIX_GEOMETRY_FLAG_NONE; OptixBuildInput bi{}; bi.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    bi.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3; bi.triangleArray.numVertices = (unsigned)v.size() * 3; bi.triangleArray.vertexBuffers = &dv; bi.triangleArray.flags = &gflags; bi.triangleArray.numSbtRecords = 1;
    OptixAccelBuildOptions ao{}; ao.buildFlags = OPTIX_BUILD_FLAG_NONE; ao.operation = OPTIX_BUILD_OPERATION_BUILD;
    OptixAccelBufferSizes bs{}; OX(optixAccelComputeMemoryUsage(ctx, &ao, &bi, 1, &bs));
    CUdeviceptr tmp, gas; CK(cudaMalloc((void **)&tmp, bs.tempSizeInBytes)); CK(cudaMalloc((void **)&gas, bs.outputSizeInBytes));
    OptixTraversableHandle hg; OX(optixAccelBuild(ctx, 0, &ao, &bi, 1, tmp, bs.tempSizeInBytes, gas, bs.outputSizeInBytes, &hg, nullptr, 0)); CK(cudaDeviceSynchronize()); CK(cudaFree((void *)tmp));
    // instances: transform = 3x4 row major, applied object -> world
    std::vector<OptixInstance> inst(K); std::vector<float> xf(12 * K);
    for (int k = 0; k < K; ++k) {
        float s = 1.f, ang = 0.f, tx = 0.f, ty = 0.f, tz = 0.f;
        if (mode == "trans") { tx = 2.5f * k; }
        else if (mode == "rot") { s = 1.f / (1.f + 0.15f * k); ang = 0.35f * k; tx = 0.9f * k - 0.45f * (K - 1); ty = 0.3f * k; tz = 0.1f * k; }
        else if (mode != "id") { fprintf(stderr, "mode\n"); return 2; }
        const float c = cosf(ang) * s, sn = sinf(ang) * s; const float m[12] = {c, -sn, 0, tx, sn, c, 0, ty, 0, 0, s, tz};
        memcpy(&xf[12 * k], m, 48); OptixInstance &I = inst[k]; I = {}; memcpy(I.transform, m, 48); I.instanceId = 100 + k; I.sbtOffset = 0; I.visibilityMask = 255; I.flags = OPTIX_INSTANCE_FLAG_NONE; I.traversableHandle = hg;
    }
    wr(pre + ".xf.bin", xf.data(), xf.size() * 4);
    CUdeviceptr di; CK(cudaMalloc((void **)&di, K * sizeof(OptixInstance))); CK(cudaMemcpy((void *)di, inst.data(), K * sizeof(OptixInstance), cudaMemcpyHostToDevice));
    OptixBuildInput ib{}; ib.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES; ib.instanceArray.instances = di; ib.instanceArray.numInstances = K;
    OptixAccelBufferSizes is{}; OX(optixAccelComputeMemoryUsage(ctx, &ao, &ib, 1, &is));
    CUdeviceptr itmp, ias; CK(cudaMalloc((void **)&itmp, is.tempSizeInBytes)); CK(cudaMalloc((void **)&ias, is.outputSizeInBytes));
    OptixTraversableHandle hi; OX(optixAccelBuild(ctx, 0, &ao, &ib, 1, itmp, is.tempSizeInBytes, ias, is.outputSizeInBytes, &hi, nullptr, 0)); CK(cudaDeviceSynchronize());
    { std::vector<char> b(bs.outputSizeInBytes); CK(cudaMemcpy(b.data(), (void *)gas, b.size(), cudaMemcpyDeviceToHost)); wr(pre + ".gas.bin", b.data(), b.size()); }
    { std::vector<char> b(is.outputSizeInBytes); CK(cudaMemcpy(b.data(), (void *)ias, b.size(), cudaMemcpyDeviceToHost)); wr(pre + ".ias.bin", b.data(), b.size()); }
    OptixModuleCompileOptions mo{}; OptixPipelineCompileOptions po{};
    po.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING; po.numPayloadValues = 8; po.numAttributeValues = 2;
    po.pipelineLaunchParamsVariableName = "params"; po.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
    std::string ptx = read_file(argv[1]); char log[4096]; size_t ls = sizeof log; OptixModule m; OX(optixModuleCreate(ctx, &mo, &po, ptx.c_str(), ptx.size(), log, &ls, &m));
    OptixProgramGroupOptions go{}; OptixProgramGroupDesc d[3] = {};
    d[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN; d[0].raygen.module = m; d[0].raygen.entryFunctionName = "__raygen__rg";
    d[1].kind = OPTIX_PROGRAM_GROUP_KIND_MISS; d[1].miss.module = m; d[1].miss.entryFunctionName = "__miss__ms";
    d[2].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP; d[2].hitgroup.moduleCH = m; d[2].hitgroup.entryFunctionNameCH = "__closesthit__ch";
    OptixProgramGroup pg[3]; ls = sizeof log; OX(optixProgramGroupCreate(ctx, d, 3, &go, log, &ls, pg));
    OptixPipelineLinkOptions lo{}; lo.maxTraceDepth = 1; OptixPipeline pipe; ls = sizeof log; OX(optixPipelineCreate(ctx, &po, &lo, pg, 3, log, &ls, &pipe));
    struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) Rec { char h[OPTIX_SBT_RECORD_HEADER_SIZE]; };
    Rec rec[3]; for (int k = 0; k < 3; ++k) OX(optixSbtRecordPackHeader(pg[k], &rec[k]));
    CUdeviceptr drec; CK(cudaMalloc((void **)&drec, sizeof rec)); CK(cudaMemcpy((void *)drec, rec, sizeof rec, cudaMemcpyHostToDevice));
    OptixShaderBindingTable sbt{}; sbt.raygenRecord = drec; sbt.missRecordBase = drec + sizeof(Rec); sbt.missRecordStrideInBytes = sizeof(Rec); sbt.missRecordCount = 1;
    sbt.hitgroupRecordBase = drec + 2 * sizeof(Rec); sbt.hitgroupRecordStrideInBytes = sizeof(Rec); sbt.hitgroupRecordCount = 1;
    Params prm{}; prm.handle = hi; prm.n = NR;
    CK(cudaMalloc((void **)&prm.rays, NR * sizeof(Ray8))); CK(cudaMemcpy((void *)prm.rays, rays.data(), NR * sizeof(Ray8), cudaMemcpyHostToDevice));
    CK(cudaMalloc((void **)&prm.out, NR * sizeof(RefOut))); CK(cudaMalloc((void **)&prm.c3, PARAMS_C3_OFFSET)); CK(cudaMemset(prm.c3, 0, PARAMS_C3_OFFSET));
    CUdeviceptr dprm; CK(cudaMalloc((void **)&dprm, sizeof prm)); CK(cudaMemcpy((void *)dprm, &prm, sizeof prm, cudaMemcpyHostToDevice));
    OX(optixLaunch(pipe, 0, dprm, sizeof prm, &sbt, NR, 1, 1)); CK(cudaDeviceSynchronize());
    std::vector<RefOut> out(NR); CK(cudaMemcpy(out.data(), prm.out, NR * sizeof(RefOut), cudaMemcpyDeviceToHost)); wr(pre + ".out.bin", out.data(), NR * sizeof(RefOut));
    std::vector<unsigned long long> c3(PARAMS_C3_OFFSET / 8); CK(cudaMemcpy(c3.data(), prm.c3, PARAMS_C3_OFFSET, cudaMemcpyDeviceToHost)); wr(pre + ".c3.bin", c3.data(), PARAMS_C3_OFFSET);
    printf("{\"gas\":\"0x%llx\",\"gas_size\":%zu,\"ias\":\"0x%llx\",\"ias_size\":%zu,\"gas_handle\":\"0x%llx\",\"ias_handle\":\"0x%llx\",\"instances\":%d,\"mode\":\"%s\"}\n", (unsigned long long)gas, (size_t)bs.outputSizeInBytes, (unsigned long long)ias, (size_t)is.outputSizeInBytes, (unsigned long long)hg, (unsigned long long)hi, K, mode.c_str());
    return 0;
}
