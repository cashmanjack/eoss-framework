// Throughput_10 host (OptiX only, no TTU code): payload size test.
//   payload PTX N_TRI RAYS.bin OUT.bin REPEAT NATTR CUSTOM NPAY      soup of N_TRI triangles (setup3 / morton generator) as triangles or as their tight boxes (custom primitives)
// prints one JSON line: ms per launch (REPEAT launches after one untimed), rays; writes Out per ray to OUT.bin.
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
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(2); } } while (0)
#define OX(x) do { OptixResult r_ = (x); if (r_ != OPTIX_SUCCESS) { fprintf(stderr, "%s:%d %s: %d\n", __FILE__, __LINE__, #x, (int)r_); exit(2); } } while (0)
struct Ray8 { float o[3], tmin, d[3], tmax; };
struct Out { unsigned hit; float t; unsigned prim; unsigned chk; };
struct Params { const Ray8 *rays; Out *out; const float *boxes; OptixTraversableHandle handle; unsigned n; };
struct Tri { float3 a, b, c; };
static std::string read_file(const char *p) { std::ifstream f(p); if (!f) { fprintf(stderr, "cannot read %s\n", p); exit(2); } std::stringstream s; s << f.rdbuf(); return s.str(); }
int main(int argc, char **argv)
{
    if (argc < 9) { fprintf(stderr, "usage\n"); return 2; }
    const int n_tri = atoi(argv[2]), repeat = atoi(argv[5]), nattr = atoi(argv[6]), custom = atoi(argv[7]), npay = atoi(argv[8]);
    std::vector<Ray8> rays; { FILE *f = fopen(argv[3], "rb"); Ray8 y; while (fread(&y, sizeof y, 1, f) == 1) rays.push_back(y); fclose(f); }
    const unsigned NR = (unsigned)rays.size();
    CK(cudaFree(0)); OX(optixInit()); OptixDeviceContextOptions co{}; OptixDeviceContext ctx; OX(optixDeviceContextCreate(0, &co, &ctx));
    const float e = 3.f / std::cbrt((float)n_tri); std::mt19937 g(1234); std::uniform_real_distribution<float> U(-1.f, 1.f);
    std::vector<Tri> v; for (int i = 0; i < n_tri; ++i) { const float3 c{U(g), U(g), U(g)}; auto pt = [&]() { return float3{c.x + 0.5f * e * U(g), c.y + 0.5f * e * U(g), c.z + 0.5f * e * U(g)}; }; v.push_back({pt(), pt(), pt()}); }
    std::vector<float> bx; for (auto &t : v) { bx.push_back(std::min({t.a.x, t.b.x, t.c.x})); bx.push_back(std::min({t.a.y, t.b.y, t.c.y})); bx.push_back(std::min({t.a.z, t.b.z, t.c.z})); bx.push_back(std::max({t.a.x, t.b.x, t.c.x})); bx.push_back(std::max({t.a.y, t.b.y, t.c.y})); bx.push_back(std::max({t.a.z, t.b.z, t.c.z})); }
    CUdeviceptr dv, db; CK(cudaMalloc((void **)&dv, v.size() * sizeof(Tri))); CK(cudaMemcpy((void *)dv, v.data(), v.size() * sizeof(Tri), cudaMemcpyHostToDevice));
    CK(cudaMalloc((void **)&db, bx.size() * 4)); CK(cudaMemcpy((void *)db, bx.data(), bx.size() * 4, cudaMemcpyHostToDevice));
    unsigned fl = OPTIX_GEOMETRY_FLAG_NONE; OptixBuildInput bi{};
    if (!custom) { bi.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES; bi.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3; bi.triangleArray.numVertices = (unsigned)v.size() * 3; bi.triangleArray.vertexBuffers = &dv; bi.triangleArray.flags = &fl; bi.triangleArray.numSbtRecords = 1; }
    else { bi.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES; bi.customPrimitiveArray.aabbBuffers = &db; bi.customPrimitiveArray.numPrimitives = (unsigned)v.size(); bi.customPrimitiveArray.strideInBytes = 24; bi.customPrimitiveArray.flags = &fl; bi.customPrimitiveArray.numSbtRecords = 1; }
    OptixAccelBuildOptions ao{}; ao.buildFlags = OPTIX_BUILD_FLAG_NONE; ao.operation = OPTIX_BUILD_OPERATION_BUILD; OptixAccelBufferSizes bs{}; OX(optixAccelComputeMemoryUsage(ctx, &ao, &bi, 1, &bs));
    CUdeviceptr tmp, gas; CK(cudaMalloc((void **)&tmp, bs.tempSizeInBytes)); CK(cudaMalloc((void **)&gas, bs.outputSizeInBytes)); OptixTraversableHandle h;
    OX(optixAccelBuild(ctx, 0, &ao, &bi, 1, tmp, bs.tempSizeInBytes, gas, bs.outputSizeInBytes, &h, nullptr, 0)); CK(cudaDeviceSynchronize());
    OptixModuleCompileOptions mo{}; OptixPipelineCompileOptions po{}; po.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS; po.numPayloadValues = npay; po.numAttributeValues = custom ? std::max(nattr, 0) : std::max(nattr, 2);
    po.pipelineLaunchParamsVariableName = "params"; po.usesPrimitiveTypeFlags = custom ? OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM : OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
    std::string ptx = read_file(argv[1]); char log[4096]; size_t ls = sizeof log; OptixModule m; { OptixResult rr_ = optixModuleCreate(ctx, &mo, &po, ptx.c_str(), ptx.size(), log, &ls, &m); if (rr_ != OPTIX_SUCCESS) { fprintf(stderr, "optixModuleCreate %d: %s\n", (int)rr_, log); return 2; } }
    OptixProgramGroupOptions go{}; OptixProgramGroupDesc d[3] = {}; d[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN; d[0].raygen.module = m; d[0].raygen.entryFunctionName = "__raygen__rg";
    d[1].kind = OPTIX_PROGRAM_GROUP_KIND_MISS; d[1].miss.module = m; d[1].miss.entryFunctionName = "__miss__ms";
    d[2].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP; d[2].hitgroup.moduleCH = m; d[2].hitgroup.entryFunctionNameCH = "__closesthit__ch"; if (custom) { d[2].hitgroup.moduleIS = m; d[2].hitgroup.entryFunctionNameIS = "__intersection__is"; }
    OptixProgramGroup pg[3]; ls = sizeof log; OX(optixProgramGroupCreate(ctx, d, 3, &go, log, &ls, pg));
    OptixPipelineLinkOptions lo{}; lo.maxTraceDepth = 1; OptixPipeline pipe; ls = sizeof log; OX(optixPipelineCreate(ctx, &po, &lo, pg, 3, log, &ls, &pipe));
    struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) Rec { char h[OPTIX_SBT_RECORD_HEADER_SIZE]; }; Rec rec[3]; for (int k = 0; k < 3; ++k) OX(optixSbtRecordPackHeader(pg[k], &rec[k]));
    CUdeviceptr drec; CK(cudaMalloc((void **)&drec, sizeof rec)); CK(cudaMemcpy((void *)drec, rec, sizeof rec, cudaMemcpyHostToDevice));
    Params prm{}; prm.handle = h; prm.n = NR; prm.boxes = (const float *)db; CK(cudaMalloc((void **)&prm.rays, NR * sizeof(Ray8))); CK(cudaMemcpy((void *)prm.rays, rays.data(), NR * sizeof(Ray8), cudaMemcpyHostToDevice));
    CK(cudaMalloc((void **)&prm.out, NR * sizeof(Out))); CUdeviceptr dprm; CK(cudaMalloc((void **)&dprm, sizeof prm)); CK(cudaMemcpy((void *)dprm, &prm, sizeof prm, cudaMemcpyHostToDevice));
    OptixShaderBindingTable sbt{}; sbt.raygenRecord = drec; sbt.missRecordBase = drec + sizeof(Rec); sbt.missRecordStrideInBytes = sizeof(Rec); sbt.missRecordCount = 1; sbt.hitgroupRecordBase = drec + 2 * sizeof(Rec); sbt.hitgroupRecordStrideInBytes = sizeof(Rec); sbt.hitgroupRecordCount = 1;
    OX(optixLaunch(pipe, 0, dprm, sizeof prm, &sbt, NR, 1, 1)); CK(cudaDeviceSynchronize());
    cudaEvent_t a0, a1; CK(cudaEventCreate(&a0)); CK(cudaEventCreate(&a1)); std::vector<float> ms;
    for (int k = 0; k < repeat; ++k) { CK(cudaEventRecord(a0)); OX(optixLaunch(pipe, 0, dprm, sizeof prm, &sbt, NR, 1, 1)); CK(cudaEventRecord(a1)); CK(cudaEventSynchronize(a1)); float x; CK(cudaEventElapsedTime(&x, a0, a1)); ms.push_back(x); }
    std::vector<Out> out(NR); CK(cudaMemcpy(out.data(), prm.out, NR * sizeof(Out), cudaMemcpyDeviceToHost)); FILE *f = fopen(argv[4], "wb"); fwrite(out.data(), sizeof(Out), NR, f); fclose(f);
    printf("{\"rays\":%u,\"npay\":%d,\"nattr\":%d,\"custom\":%d,\"ms\":[", NR, npay, nattr, custom); for (size_t k = 0; k < ms.size(); ++k) printf("%s%.5f", k ? "," : "", ms[k]); printf("]}\n");
    return 0;
}
