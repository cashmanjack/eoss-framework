// BVH_3: cost of an SM round trip for custom primitives whose intersection always returns false.
//   bvh3 PTX KIND N MODE COUNT OUT.json
//   KIND tri | aabb; N primitives of the par:N:s|o geometry (setup3.cu): triangles parallel to a +z ray at
//   x = y = 0, 1e-3 off it; aabb uses each triangle's box (it holds the ray's line, so every primitive is a
//   candidate). COUNT 1: the IS program counts its calls (atomic), 0: empty IS. One warp (32 rays, the same
//   ray), 11 launches, launch 0 dropped. Plain OptiX, no patched code.
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(2); } } while (0)
#define OX(x) do { OptixResult r_ = (x); if (r_ != OPTIX_SUCCESS) { fprintf(stderr, "%s:%d %s: %d\n", __FILE__, __LINE__, #x, (int)r_); exit(2); } } while (0)
struct P3 { unsigned long long *cyc; unsigned *hit; unsigned long long *is_calls; OptixTraversableHandle h; int count_is; };
struct V3 { float x, y, z; };

int main(int argc, char **argv)
{
    if (argc < 7) { fprintf(stderr, "usage: bvh3 PTX tri|aabb N s|o COUNT OUT.json\n"); return 2; }
    const bool aabb = !strcmp(argv[2], "aabb"); const int n = atoi(argv[3]); const char mode = argv[4][0]; const int count_is = atoi(argv[5]);
    if (n < 1 || n > (1 << 20) || (mode != 's' && mode != 'o')) { fprintf(stderr, "bad args\n"); return 2; }
    std::vector<V3> v; std::mt19937 g(99); std::uniform_real_distribution<float> U(0.5f, 2.f); const float delta = 1e-3f;
    for (int k = 0; k < n; ++k) {   // same generator as setup3.cu par:N
        const float t = U(g) * (k % 2 ? -1.f : 1.f);
        const float z0 = mode == 's' ? 2.f * k / n - 1.f : -1.f, z1 = mode == 's' ? 2.f * (k + 1) / n - 1.f : 1.f;
        v.push_back({-0.5f, -0.5f * t + delta, z0}); v.push_back({0.5f, 0.5f * t + delta, z0}); v.push_back({0.f, delta, z1});
    }
    CK(cudaFree(0)); OX(optixInit());
    OptixDeviceContextOptions co{}; OptixDeviceContext ctx; OX(optixDeviceContextCreate(0, &co, &ctx));
    OptixBuildInput bi{}; unsigned gflags = OPTIX_GEOMETRY_FLAG_NONE; CUdeviceptr dbuf;
    if (aabb) {
        std::vector<OptixAabb> bb(n);
        for (int k = 0; k < n; ++k) {
            const V3 *t = &v[3 * k]; OptixAabb &b = bb[k];
            b.minX = std::min({t[0].x, t[1].x, t[2].x}); b.maxX = std::max({t[0].x, t[1].x, t[2].x});
            b.minY = std::min({t[0].y, t[1].y, t[2].y}); b.maxY = std::max({t[0].y, t[1].y, t[2].y});
            b.minZ = std::min({t[0].z, t[1].z, t[2].z}); b.maxZ = std::max({t[0].z, t[1].z, t[2].z});
        }
        CK(cudaMalloc((void **)&dbuf, bb.size() * sizeof(OptixAabb))); CK(cudaMemcpy((void *)dbuf, bb.data(), bb.size() * sizeof(OptixAabb), cudaMemcpyHostToDevice));
        bi.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES; bi.customPrimitiveArray.aabbBuffers = &dbuf; bi.customPrimitiveArray.numPrimitives = n;
        bi.customPrimitiveArray.flags = &gflags; bi.customPrimitiveArray.numSbtRecords = 1;
    } else {
        CK(cudaMalloc((void **)&dbuf, v.size() * sizeof(V3))); CK(cudaMemcpy((void *)dbuf, v.data(), v.size() * sizeof(V3), cudaMemcpyHostToDevice));
        bi.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES; bi.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3; bi.triangleArray.numVertices = (unsigned)v.size();
        bi.triangleArray.vertexBuffers = &dbuf; bi.triangleArray.flags = &gflags; bi.triangleArray.numSbtRecords = 1;
    }
    OptixAccelBuildOptions ao{}; ao.buildFlags = OPTIX_BUILD_FLAG_NONE; ao.operation = OPTIX_BUILD_OPERATION_BUILD;
    OptixAccelBufferSizes bs{}; OX(optixAccelComputeMemoryUsage(ctx, &ao, &bi, 1, &bs));
    CUdeviceptr tmp, gas; CK(cudaMalloc((void **)&tmp, bs.tempSizeInBytes)); CK(cudaMalloc((void **)&gas, bs.outputSizeInBytes));
    OptixTraversableHandle h; OX(optixAccelBuild(ctx, 0, &ao, &bi, 1, tmp, bs.tempSizeInBytes, gas, bs.outputSizeInBytes, &h, nullptr, 0));
    CK(cudaDeviceSynchronize());
    std::ifstream f(argv[1]); std::stringstream ss; ss << f.rdbuf(); const std::string ptx = ss.str();
    OptixModuleCompileOptions mo{}; OptixPipelineCompileOptions po{};
    po.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS; po.numPayloadValues = 1; po.numAttributeValues = 2;
    po.pipelineLaunchParamsVariableName = "params";
    po.usesPrimitiveTypeFlags = aabb ? OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM : OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
    char log[4096]; size_t ls = sizeof log; OptixModule m; OX(optixModuleCreate(ctx, &mo, &po, ptx.c_str(), ptx.size(), log, &ls, &m));
    OptixProgramGroupOptions go{}; OptixProgramGroupDesc d[3] = {};
    d[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN; d[0].raygen.module = m; d[0].raygen.entryFunctionName = "__raygen__rg";
    d[1].kind = OPTIX_PROGRAM_GROUP_KIND_MISS; d[1].miss.module = m; d[1].miss.entryFunctionName = "__miss__ms";
    d[2].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP; d[2].hitgroup.moduleCH = m; d[2].hitgroup.entryFunctionNameCH = "__closesthit__ch";
    if (aabb) { d[2].hitgroup.moduleIS = m; d[2].hitgroup.entryFunctionNameIS = "__intersection__is"; }
    OptixProgramGroup pg[3]; ls = sizeof log; OX(optixProgramGroupCreate(ctx, d, 3, &go, log, &ls, pg));
    OptixPipelineLinkOptions lo{}; lo.maxTraceDepth = 1; OptixPipeline pipe; ls = sizeof log; OX(optixPipelineCreate(ctx, &po, &lo, pg, 3, log, &ls, &pipe));
    struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) Rec { char hh[OPTIX_SBT_RECORD_HEADER_SIZE]; };
    Rec rec[3]; for (int k = 0; k < 3; ++k) OX(optixSbtRecordPackHeader(pg[k], &rec[k]));
    CUdeviceptr drec; CK(cudaMalloc((void **)&drec, sizeof rec)); CK(cudaMemcpy((void *)drec, rec, sizeof rec, cudaMemcpyHostToDevice));
    OptixShaderBindingTable sbt{}; sbt.raygenRecord = drec; sbt.missRecordBase = drec + sizeof(Rec); sbt.missRecordStrideInBytes = sizeof(Rec); sbt.missRecordCount = 1;
    sbt.hitgroupRecordBase = drec + 2 * sizeof(Rec); sbt.hitgroupRecordStrideInBytes = sizeof(Rec); sbt.hitgroupRecordCount = 1;
    const int W = 32, R = 11; P3 p{}; p.h = h; p.count_is = count_is;
    CK(cudaMalloc((void **)&p.cyc, W * 8)); CK(cudaMalloc((void **)&p.hit, W * 4)); CK(cudaMalloc((void **)&p.is_calls, 8));
    CUdeviceptr dp; CK(cudaMalloc((void **)&dp, sizeof p)); CK(cudaMemcpy((void *)dp, &p, sizeof p, cudaMemcpyHostToDevice));
    FILE *jf = fopen(argv[6], "w");
    fprintf(jf, "{\"kind\": \"%s\", \"n\": %d, \"mode\": \"%c\", \"count_is\": %d, \"gas_size\": %zu, \"launches\": [", argv[2], n, mode, count_is, (size_t)bs.outputSizeInBytes);
    for (int it = 0; it < R; ++it) {
        CK(cudaMemset(p.is_calls, 0, 8));
        OX(optixLaunch(pipe, 0, dp, sizeof p, &sbt, W, 1, 1)); CK(cudaDeviceSynchronize());
        unsigned long long cyc[W], calls; unsigned hit[W];
        CK(cudaMemcpy(cyc, p.cyc, sizeof cyc, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hit, p.hit, sizeof hit, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(&calls, p.is_calls, 8, cudaMemcpyDeviceToHost));
        unsigned nh = 0; for (int k = 0; k < W; ++k) nh += hit[k];
        std::vector<unsigned long long> c(cyc, cyc + W); std::sort(c.begin(), c.end());
        fprintf(jf, "%s{\"cyc_min\": %llu, \"cyc_med\": %llu, \"cyc_max\": %llu, \"hits\": %u, \"is_calls\": %llu}", it ? ", " : "", c[0], c[W / 2], c[W - 1], nh, calls);
    }
    fprintf(jf, "]}\n"); fclose(jf);
    printf("bvh3 %s N %d %c count %d: done\n", argv[2], n, mode, count_is);
    return 0;
}
