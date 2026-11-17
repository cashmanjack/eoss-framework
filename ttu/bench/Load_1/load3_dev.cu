// Load_1 program arm device code (OptiX only). PROG 0: trivial intersection program (box load and point test, no report); PROG 1: Mochi's raygen and intersection program (src/s23-3Dparticlesim/deviceCode.cu, OWL removed).
#include <optix.h>
struct Particle { float3 center; float3 velocity; float radius; float mass; };
struct Rec { unsigned long long c0, c1, g0, g1; unsigned sm, calls; };
struct Params { const Particle *particles; const float *boxes; float3 *acc; unsigned *callw; Rec *rec; OptixTraversableHandle handle; unsigned n; float stiffness, restitution; };
extern "C" __constant__ Params params;
__device__ __forceinline__ unsigned long long gt() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ void count_call() { const unsigned m = __activemask(); if ((threadIdx.x & 31) == (unsigned)(__ffs(m) - 1)) params.callw[optixGetLaunchIndex().x >> 5] += __popc(m); }
extern "C" __global__ void __raygen__rg()
{
    const unsigned i = optixGetLaunchIndex().x; if (i >= params.n) return;
    const unsigned long long g0 = gt(), c0 = clock64();
    Particle q = params.particles[i];
#if PROG == 1
    const unsigned long long a = (unsigned long long)&q; unsigned p0 = (unsigned)a, p1 = (unsigned)(a >> 32);
    optixTrace(params.handle, q.center, make_float3(0.f, 0.f, 1.f), 0.f, 1e-16f, 0.f, OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT, 0, 1, 0, p0, p1);
#else
    unsigned p0 = 0, p1 = 0;
    optixTrace(params.handle, q.center, make_float3(0.f, 0.f, 1.f), 0.f, 1e-16f, 0.f, OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT, 0, 1, 0, p0, p1);
#endif
    const unsigned long long c1 = clock64(), g1 = gt();
    if ((threadIdx.x & 31) == 0) { Rec r; r.c0 = c0; r.c1 = c1; r.g0 = g0; r.g1 = g1; r.sm = 0; r.calls = params.callw[i >> 5]; params.rec[i >> 5] = r; }
}
extern "C" __global__ void __miss__ms() {}
extern "C" __global__ void __closesthit__ch() {}
extern "C" __global__ void __intersection__is()
{
    count_call();
    const unsigned primID = optixGetPrimitiveIndex();
#if PROG == 1
    const unsigned queryID = optixGetLaunchIndex().x;
    const Particle hit_prim = params.particles[primID];
    const unsigned long long a = ((unsigned long long)optixGetPayload_1() << 32) | optixGetPayload_0(); const Particle &query_prim = *(const Particle *)a;
    const float3 d = make_float3(hit_prim.center.x - query_prim.center.x, hit_prim.center.y - query_prim.center.y, hit_prim.center.z - query_prim.center.z);
    const float distance = sqrtf(d.x * d.x + d.y * d.y + d.z * d.z), delta = -distance + hit_prim.radius + query_prim.radius;
    const float criteria_other = 2.f * query_prim.radius - distance, current_criteria = 2.f * hit_prim.radius - distance;
    if ((current_criteria > 0) && (delta > 0) && ((queryID < primID) || (criteria_other < 0))) {
        const float3 normal = make_float3(d.x / distance, d.y / distance, d.z / distance); const float3 f1 = make_float3(normal.x * delta * params.stiffness, normal.y * delta * params.stiffness, normal.z * delta * params.stiffness);
        const float cm = (hit_prim.mass * query_prim.mass) / (hit_prim.mass + query_prim.mass), C = params.restitution * sqrtf(params.stiffness * cm);
        const float3 sv = make_float3((hit_prim.velocity.x - query_prim.velocity.x) * normal.x, (hit_prim.velocity.y - query_prim.velocity.y) * normal.y, (hit_prim.velocity.z - query_prim.velocity.z) * normal.z);
        const float3 f2 = make_float3(C * sv.x * normal.x, C * sv.y * normal.y, C * sv.z * normal.z), df = make_float3(f2.x - f1.x, f2.y - f1.y, f2.z - f1.z);
        atomicAdd(&params.acc[queryID].x, df.x); atomicAdd(&params.acc[queryID].y, df.y); atomicAdd(&params.acc[queryID].z, df.z);
        atomicAdd(&params.acc[primID].x, -df.x); atomicAdd(&params.acc[primID].y, -df.y); atomicAdd(&params.acc[primID].z, -df.z);
    }
#else
    const float *b = params.boxes + 6 * primID; const Particle q = params.particles[optixGetLaunchIndex().x]; (void)q;
    const float3 o = optixGetObjectRayOrigin(); volatile bool in = o.x >= b[0] && o.x <= b[3] && o.y >= b[1] && o.y <= b[4] && o.z >= b[2] && o.z <= b[5]; (void)in;
#endif
}
