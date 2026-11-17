// Struct_1 OptiX programs. Custom primitive arm: the intersection program only counts its calls (never reports), so the count is every candidate the hardware sends to the SM.
#include <optix.h>
#include "struct_params.h"
extern "C" __constant__ Params params;
extern "C" __global__ void __raygen__rg()
{
    const unsigned i = optixGetLaunchIndex().x;
    const unsigned long long ta = clock64();
    if (i >= params.n) return;
    unsigned long long tt[8]; tt[0] = ta; for (int q = 1; q < 8; ++q) tt[q] = 0;
    Out o0{};
    const unsigned nrep = params.clk ? params.nrep : 1u;
    #pragma unroll 1
    for (unsigned k = 0; k < nrep; ++k) {
        const Ray8 r = params.rays[i];
        unsigned p0 = 0, p1 = 0, p2 = 0, p3 = 0;
        unsigned dm; asm volatile("mov.b32 %0, %1;" : "=r"(dm) : "r"(__float_as_uint(r.o[0]) ^ __float_as_uint(r.o[2]) ^ __float_as_uint(r.d[0]) ^ __float_as_uint(r.tmax)));
        const unsigned long long pb = clock64();
        optixTrace(params.handle, make_float3(r.o[0], r.o[1], r.o[2]), make_float3(r.d[0], r.d[1], r.d[2]), r.tmin, r.tmax, 0.f,
                   OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE, 0, 1, 0, p0, p1, p2, p3);
        const unsigned long long pc = clock64();
        if (k == 0) { o0.hit = p0 & 1u; o0.inst = p0 >> 1; o0.tbits = p1; o0.prim = p2; o0.calls = p3; }
        if (k < 3) { tt[1 + 2 * k] = pb; tt[2 + 2 * k] = pc; }
    }
    params.out[i] = o0;
    if (params.clk) { tt[7] = clock64(); for (int q = 0; q < 8; ++q) params.clk[8 * i + q] = tt[q]; }
}
extern "C" __global__ void __miss__ms() { optixSetPayload_0(0u); }
extern "C" __global__ void __closesthit__ch()
{
    optixSetPayload_0(1u | (optixGetInstanceId() << 1));
    optixSetPayload_1(__float_as_uint(optixGetRayTmax()));
    optixSetPayload_2(optixGetPrimitiveIndex());
}
extern "C" __global__ void __intersection__cnt() { optixSetPayload_3(optixGetPayload_3() + 1u); }
