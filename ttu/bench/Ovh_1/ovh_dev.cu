// Ovh_1 OptiX programs (model/predictions_ovh.md). Compile with -DEMPTY for the raygen that has no optixTrace in its code.
#include <optix.h>
#include "ovh_params.h"
extern "C" __constant__ Params params;

__device__ __forceinline__ unsigned hash32(unsigned x) { x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16; return x; }
__device__ __forceinline__ float u01(unsigned h) { return (float)(h >> 8) * (1.0f / 16777216.0f) * 2.f - 1.f; }

extern "C" __global__ void __raygen__rg()
{
    const unsigned i = optixGetLaunchIndex().x;
    unsigned long long c0 = 0, g0 = 0;
    if (params.cal) { c0 = clock64(); asm volatile("mov.u64 %0, %globaltimer;" : "=l"(g0)); }
    if (i >= params.n) return;
    const unsigned h0 = hash32(i * 4u + 1u), h1 = hash32(i * 4u + 2u), h2 = hash32(i * 4u + 3u), h3 = hash32(i * 4u + 4u);
    float3 o, d;
    if (params.scene == 0) { o = make_float3(0.5f * u01(h0), 0.5f * u01(h1), -1.f); d = make_float3(0.f, 0.f, 1.f); }
    else {
        o = make_float3(u01(h0), u01(h1), u01(h2));
        float3 v = make_float3(u01(h3), u01(hash32(h3)), u01(hash32(h3 + 7u)));
        const float l = rsqrtf(v.x * v.x + v.y * v.y + v.z * v.z + 1e-12f); d = make_float3(v.x * l, v.y * l, v.z * l);
    }
    unsigned acc = __float_as_uint(o.x) ^ __float_as_uint(d.z);
#ifndef EMPTY
    #pragma unroll 1
    for (unsigned k = 0; k < params.nrep; ++k) {
        unsigned p0 = 0, p1 = 0;
        optixTrace(params.handle, o, d, 0.f, 1e16f, 0.f, OptixVisibilityMask(255), 0, 0, 1, 0, p0, p1);
        acc ^= p0 ^ (p1 << 1);
    }
#endif
    if (acc == 0x7fc00001u) params.sink[i >> 5] = acc;   // an impossible value: keeps the work live, no memory traffic in practice
    if (params.cal && (i & 31u) == 0) {
        unsigned long long c1 = clock64(), g1; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(g1));
        unsigned long long *w = params.cal + 4ull * (i >> 5); w[0] = c0; w[1] = c1; w[2] = g0; w[3] = g1;
    }
}

extern "C" __global__ void __miss__ms() { optixSetPayload_0(0u); }

extern "C" __global__ void __closesthit__ch()
{
    optixSetPayload_0(1u);
    optixSetPayload_1(__float_as_uint(optixGetRayTmax()));
}
