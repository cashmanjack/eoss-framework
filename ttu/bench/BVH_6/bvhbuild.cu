// BVH_6 host (OptiX only, no TTU code, no launch): builds an acceleration structure in our own process and copies our own buffer out with cudaMemcpy.
//   bvhbuild tris  TRIS.bin OUT                 triangle GAS            -> OUT.gas
//   bvhbuild aabb  AABB.bin OUT                 custom primitive GAS (6 floats per box: lo xyz, hi xyz) -> OUT.gas
//   bvhbuild ias   TRIS.bin OUT MASKS [FLAGS]   GAS of the triangles plus an IAS, one instance per mask (comma list, decimal or 0x hex), translated by 4*k in x, FLAGS = comma list of instance flags (optional) -> OUT.gas, OUT.ias
//   bvhbuild ias2  TRIS.bin TRIS2.bin OUT MASKS two different GASes alternating over the instances -> OUT.gas OUT.gas2 OUT.ias
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(2); } } while (0)
#define OX(x) do { OptixResult r_ = (x); if (r_ != OPTIX_SUCCESS) { fprintf(stderr, "%s:%d %s: %d\n", __FILE__, __LINE__, #x, (int)r_); exit(2); } } while (0)
template <class T> static std::vector<T> rd(const char *p) { FILE *f = fopen(p, "rb"); if (!f) { fprintf(stderr, "cannot read %s\n", p); exit(2); } std::vector<T> v; T x; while (fread(&x, sizeof x, 1, f) == 1) v.push_back(x); fclose(f); return v; }
static void wr(const std::string &p, const void *d, size_t n) { FILE *f = fopen(p.c_str(), "wb"); fwrite(d, 1, n, f); fclose(f); }
static std::vector<unsigned> csv(const char *s) { std::vector<unsigned> v; std::stringstream ss(s); std::string t; while (std::getline(ss, t, ',')) v.push_back((unsigned)strtoul(t.c_str(), 0, 0)); return v; }
static OptixDeviceContext ctx;
static OptixTraversableHandle build(const OptixBuildInput &bi, const std::string &dump, size_t *sz)
{
    OptixAccelBuildOptions ao{}; ao.buildFlags = OPTIX_BUILD_FLAG_NONE; ao.operation = OPTIX_BUILD_OPERATION_BUILD;
    OptixAccelBufferSizes bs{}; OX(optixAccelComputeMemoryUsage(ctx, &ao, &bi, 1, &bs));
    CUdeviceptr tmp, out; CK(cudaMalloc((void **)&tmp, bs.tempSizeInBytes)); CK(cudaMalloc((void **)&out, bs.outputSizeInBytes));
    OptixTraversableHandle h; OX(optixAccelBuild(ctx, 0, &ao, &bi, 1, tmp, bs.tempSizeInBytes, out, bs.outputSizeInBytes, &h, nullptr, 0)); CK(cudaDeviceSynchronize());
    std::vector<char> b(bs.outputSizeInBytes); CK(cudaMemcpy(b.data(), (void *)out, b.size(), cudaMemcpyDeviceToHost)); wr(dump, b.data(), b.size()); if (sz) *sz = b.size();
    return h;
}
static OptixTraversableHandle tri_gas(const char *file, const std::string &dump)
{
    auto v = rd<float>(file); CUdeviceptr dv; CK(cudaMalloc((void **)&dv, v.size() * 4)); CK(cudaMemcpy((void *)dv, v.data(), v.size() * 4, cudaMemcpyHostToDevice));
    static unsigned fl = getenv("GEOFLAG") ? (unsigned)strtoul(getenv("GEOFLAG"), 0, 0) : OPTIX_GEOMETRY_FLAG_NONE; OptixBuildInput bi{}; bi.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    bi.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3; bi.triangleArray.numVertices = (unsigned)(v.size() / 3); bi.triangleArray.vertexBuffers = &dv; bi.triangleArray.flags = &fl; bi.triangleArray.numSbtRecords = 1;
    return build(bi, dump, nullptr);
}
int main(int argc, char **argv)
{
    if (argc < 4) { fprintf(stderr, "usage\n"); return 2; }
    const std::string mode = argv[1];
    CK(cudaFree(0)); OX(optixInit()); OptixDeviceContextOptions co{}; OX(optixDeviceContextCreate(0, &co, &ctx));
    if (mode == "tris") { tri_gas(argv[2], std::string(argv[3]) + ".gas"); return 0; }
    if (mode == "aabb") {
        auto v = rd<float>(argv[2]); CUdeviceptr db; CK(cudaMalloc((void **)&db, v.size() * 4)); CK(cudaMemcpy((void *)db, v.data(), v.size() * 4, cudaMemcpyHostToDevice));
        static unsigned fl = getenv("GEOFLAG") ? (unsigned)strtoul(getenv("GEOFLAG"), 0, 0) : OPTIX_GEOMETRY_FLAG_NONE; OptixBuildInput bi{}; bi.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
        bi.customPrimitiveArray.aabbBuffers = &db; bi.customPrimitiveArray.numPrimitives = (unsigned)(v.size() / 6); bi.customPrimitiveArray.strideInBytes = 24; bi.customPrimitiveArray.flags = &fl; bi.customPrimitiveArray.numSbtRecords = 1;
        build(bi, std::string(argv[3]) + ".gas", nullptr); return 0;
    }
    const bool two = mode == "ias2"; if (mode != "ias" && !two) { fprintf(stderr, "mode\n"); return 2; }
    const std::string out = argv[two ? 4 : 3];
    OptixTraversableHandle h1 = tri_gas(argv[2], out + ".gas"), h2 = two ? tri_gas(argv[3], out + ".gas2") : h1;
    auto masks = csv(argv[two ? 5 : 4]); std::vector<unsigned> flags; if (!two && argc > 5) flags = csv(argv[5]);
    std::vector<OptixInstance> inst(masks.size());
    for (size_t k = 0; k < masks.size(); ++k) {
        OptixInstance &I = inst[k]; I = {}; const float m[12] = {1, 0, 0, 4.f * k, 0, 1, 0, 0, 0, 0, 1, 0}; memcpy(I.transform, m, 48);
        I.instanceId = 100 + (unsigned)k; I.sbtOffset = 0; I.visibilityMask = masks[k]; I.flags = flags.empty() ? OPTIX_INSTANCE_FLAG_NONE : flags[k % flags.size()]; I.traversableHandle = (two && (k & 1)) ? h2 : h1;
    }
    CUdeviceptr di; CK(cudaMalloc((void **)&di, inst.size() * sizeof(OptixInstance))); CK(cudaMemcpy((void *)di, inst.data(), inst.size() * sizeof(OptixInstance), cudaMemcpyHostToDevice));
    OptixBuildInput ib{}; ib.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES; ib.instanceArray.instances = di; ib.instanceArray.numInstances = (unsigned)inst.size();
    build(ib, out + ".ias", nullptr);
    return 0;
}
