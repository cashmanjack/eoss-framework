// NNsphere A host: one built in sphere (centre 0, radius 1), a table of rays, many option variants. usage: inside PTX OUT.txt  (OUT opened with mode x)
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(2); } } while (0)
#define OX(x) do { OptixResult r_ = (x); if (r_ != OPTIX_SUCCESS) { fprintf(stderr, "%s:%d %s: %d\n", __FILE__, __LINE__, #x, (int)r_); exit(2); } } while (0)
struct Params { OptixTraversableHandle handle; const float4 *org; const float4 *dir; unsigned rayflags; float tmin; float4 *res; };
int main(int argc, char **argv)
{
    std::ifstream f(argv[1]); std::stringstream ss; ss << f.rdbuf(); std::string ptx = ss.str();
    CK(cudaFree(0)); OX(optixInit()); OptixDeviceContextOptions co{}; OptixDeviceContext ctx; OX(optixDeviceContextCreate(0, &co, &ctx));
    // rays: label, origin, direction
    struct R { const char *name; float o[3], d[3]; }; const float s3 = 0.57735027f;
    std::vector<R> rays = {
        {"control: origin -1.01, dir +x (entry t 0.01)", {-1.01f, 0, 0}, {1, 0, 0}}, {"control: origin -2, dir +x (entry t 1)", {-2, 0, 0}, {1, 0, 0}},
        {"control: origin -2, dir -x (away, miss)", {-2, 0, 0}, {-1, 0, 0}}, {"origin on surface -1, dir +x", {-1, 0, 0}, {1, 0, 0}},
        {"inside: centre, dir +x", {0, 0, 0}, {1, 0, 0}}, {"inside: centre, dir diagonal", {0, 0, 0}, {s3, s3, s3}}, {"inside: 0.5 on x, dir +x", {0.5f, 0, 0}, {1, 0, 0}}, {"inside: 0.5 on x, dir -x", {0.5f, 0, 0}, {-1, 0, 0}},
        {"inside: 0.99 on x, dir +x (0.01 to exit)", {0.99f, 0, 0}, {1, 0, 0}}, {"inside: 0.99 on x, dir -x", {0.99f, 0, 0}, {-1, 0, 0}}, {"inside: 0.999 on x, dir +x", {0.999f, 0, 0}, {1, 0, 0}},
        {"inside: 0.5 on y, dir +z", {0, 0.5f, 0}, {0, 0, 1}}, {"inside: -0.9 on z, dir diagonal", {0, 0, -0.9f}, {s3, s3, s3}},
        {"inside: 0.3 off axis, dir +x", {0, 0.3f, 0.3f}, {1, 0, 0}}};
    const unsigned NR = (unsigned)rays.size(); std::vector<float4> ho(NR), hd(NR); for (unsigned i = 0; i < NR; ++i) { ho[i] = make_float4(rays[i].o[0], rays[i].o[1], rays[i].o[2], 0); hd[i] = make_float4(rays[i].d[0], rays[i].d[1], rays[i].d[2], 0); }
    float4 *dorg, *ddir, *dres; CK(cudaMalloc((void **)&dorg, NR * 16)); CK(cudaMalloc((void **)&ddir, NR * 16)); CK(cudaMalloc((void **)&dres, 2 * 4096 * 16));
    CK(cudaMemcpy(dorg, ho.data(), NR * 16, cudaMemcpyHostToDevice)); CK(cudaMemcpy(ddir, hd.data(), NR * 16, cudaMemcpyHostToDevice));
    float cen[3] = {0, 0, 0}, rad = 1.f; float3 *dc; float *dr; CK(cudaMalloc((void **)&dc, 12)); CK(cudaMalloc((void **)&dr, 4)); CK(cudaMemcpy(dc, cen, 12, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dr, &rad, 4, cudaMemcpyHostToDevice));
    FILE *out = fopen(argv[2], "wx"); if (!out) { fprintf(stderr, "refuse to overwrite %s\n", argv[2]); return 2; }
    struct GF { const char *name; unsigned v; }; const GF gfs[] = {{"geomflag NONE (any hit on)", OPTIX_GEOMETRY_FLAG_NONE}, {"geomflag DISABLE_ANYHIT", OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT}, {"geomflag DISABLE_TRIANGLE_FACE_CULLING", OPTIX_GEOMETRY_FLAG_DISABLE_TRIANGLE_FACE_CULLING}, {"geomflag REQUIRE_SINGLE_ANYHIT_CALL", OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL}};
    struct RF { const char *name; unsigned v; }; const RF rfs[] = {{"rayflags NONE", OPTIX_RAY_FLAG_NONE}, {"CULL_BACK_FACING_TRIANGLES", OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES}, {"CULL_FRONT_FACING_TRIANGLES", OPTIX_RAY_FLAG_CULL_FRONT_FACING_TRIANGLES}, {"CULL_DISABLED_ANYHIT", OPTIX_RAY_FLAG_CULL_DISABLED_ANYHIT}, {"CULL_ENFORCED_ANYHIT", OPTIX_RAY_FLAG_CULL_ENFORCED_ANYHIT}, {"DISABLE_ANYHIT", OPTIX_RAY_FLAG_DISABLE_ANYHIT}, {"ENFORCE_ANYHIT", OPTIX_RAY_FLAG_ENFORCE_ANYHIT}};
    const float tmins[] = {0.f, 1e-4f};
    for (const GF &g : gfs) {
        unsigned gflag = g.v; CUdeviceptr dA = (CUdeviceptr)dc, dB = (CUdeviceptr)dr; OptixBuildInput bi{}; bi.type = OPTIX_BUILD_INPUT_TYPE_SPHERES; bi.sphereArray.vertexBuffers = &dA; bi.sphereArray.numVertices = 1; bi.sphereArray.radiusBuffers = &dB; bi.sphereArray.singleRadius = 1; bi.sphereArray.flags = &gflag; bi.sphereArray.numSbtRecords = 1;
        OptixAccelBuildOptions ao{}; ao.buildFlags = OPTIX_BUILD_FLAG_NONE; ao.operation = OPTIX_BUILD_OPERATION_BUILD; OptixAccelBufferSizes bs{}; OX(optixAccelComputeMemoryUsage(ctx, &ao, &bi, 1, &bs));
        CUdeviceptr tmp, gas; CK(cudaMalloc((void **)&tmp, bs.tempSizeInBytes)); CK(cudaMalloc((void **)&gas, bs.outputSizeInBytes)); OptixTraversableHandle root; OX(optixAccelBuild(ctx, 0, &ao, &bi, 1, tmp, bs.tempSizeInBytes, gas, bs.outputSizeInBytes, &root, nullptr, 0)); CK(cudaDeviceSynchronize());
        OptixModuleCompileOptions mo{}; OptixPipelineCompileOptions po{}; po.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS; po.numPayloadValues = 8; po.numAttributeValues = 2; po.pipelineLaunchParamsVariableName = "params"; po.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE;
        char log[4096]; size_t ls = sizeof log; OptixModule m; OX(optixModuleCreate(ctx, &mo, &po, ptx.c_str(), ptx.size(), log, &ls, &m));
        OptixModule bm; OptixBuiltinISOptions bio{}; bio.builtinISModuleType = OPTIX_PRIMITIVE_TYPE_SPHERE; OX(optixBuiltinISModuleGet(ctx, &mo, &po, &bio, &bm));
        OptixProgramGroupOptions go{}; OptixProgramGroupDesc d[3] = {}; d[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN; d[0].raygen.module = m; d[0].raygen.entryFunctionName = "__raygen__rg";
        d[1].kind = OPTIX_PROGRAM_GROUP_KIND_MISS; d[1].miss.module = m; d[1].miss.entryFunctionName = "__miss__ms";
        d[2].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP; d[2].hitgroup.moduleCH = m; d[2].hitgroup.entryFunctionNameCH = "__closesthit__ch"; d[2].hitgroup.moduleAH = m; d[2].hitgroup.entryFunctionNameAH = "__anyhit__ah"; d[2].hitgroup.moduleIS = bm;
        OptixProgramGroup pg[3]; ls = sizeof log; OX(optixProgramGroupCreate(ctx, d, 3, &go, log, &ls, pg));
        OptixPipelineLinkOptions lo{}; lo.maxTraceDepth = 1; OptixPipeline pipe; ls = sizeof log; OX(optixPipelineCreate(ctx, &po, &lo, pg, 3, log, &ls, &pipe));
        struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) Rec { char h[OPTIX_SBT_RECORD_HEADER_SIZE]; }; Rec rec[3]; for (int k = 0; k < 3; ++k) OX(optixSbtRecordPackHeader(pg[k], &rec[k]));
        CUdeviceptr drec; CK(cudaMalloc((void **)&drec, sizeof rec)); CK(cudaMemcpy((void *)drec, rec, sizeof rec, cudaMemcpyHostToDevice));
        OptixShaderBindingTable sbt{}; sbt.raygenRecord = drec; sbt.missRecordBase = drec + sizeof(Rec); sbt.missRecordStrideInBytes = sizeof(Rec); sbt.missRecordCount = 1; sbt.hitgroupRecordBase = drec + 2 * sizeof(Rec); sbt.hitgroupRecordStrideInBytes = sizeof(Rec); sbt.hitgroupRecordCount = 1;
        CUdeviceptr dprm; CK(cudaMalloc((void **)&dprm, sizeof(Params)));
        for (const RF &rf : rfs) for (float tm : tmins) {
            if (g.v == OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT && rf.v == OPTIX_RAY_FLAG_ENFORCE_ANYHIT) {}   // contradictory combination still sent: OptiX decides
            Params p{}; p.handle = root; p.org = dorg; p.dir = ddir; p.rayflags = rf.v; p.tmin = tm; p.res = dres; CK(cudaMemcpy((void *)dprm, &p, sizeof p, cudaMemcpyHostToDevice));
            CK(cudaMemset(dres, 0, 2 * 4096 * 16)); OptixResult rr = optixLaunch(pipe, 0, dprm, sizeof p, &sbt, NR, 1, 1); cudaError_t ce = cudaDeviceSynchronize();
            fprintf(out, "## %s | %s | tmin %g | launch %d cuda %d\n", g.name, rf.name, tm, (int)rr, (int)ce); if (rr || ce) continue;
            std::vector<float4> h(2 * 4096); CK(cudaMemcpy(h.data(), dres, 2 * 4096 * 16, cudaMemcpyDeviceToHost));
            for (unsigned i = 0; i < NR; ++i) { const float4 a = h[i], b = h[4096 + i]; fprintf(out, "%-48s ch_hit %d ch_t %.6f ch_kind %d ch_back %d | ah_calls %d ah_t %.6f ah_back %d ah_kind %d\n", rays[i].name, (int)a.x, a.x != 0 ? a.y : 0.f, (int)b.x, (int)b.y, (int)a.z, a.z != 0 ? a.w : 0.f, (int)b.z, (int)b.w); }
        }
    }
    fclose(out); printf("done %s\n", argv[2]); return 0;
}
