// OptiX programs of the RTNN style radius search reference: one ray per query (origin = the query point, tmax 1e-16, direction +z), custom primitive boxes
// around the data points, an intersection program that tests the exact distance and records the neighbour (never reports an intersection).
#include <optix.h>
#include "rtnn_params.h"

extern "C" __constant__ RParams params;

extern "C" __global__ void __raygen__rg()
{
    const unsigned i = optixGetLaunchIndex().x;
    if (i == 0) {
        const unsigned long long t = ((const volatile unsigned long long *)&params)[(int)(0x58 - PARAMS_C3_OFFSET) / 8];
        params.inst[0] = t - 0x100;
        for (int k = 0; k < 16; ++k) params.inst[1 + k] = ((const volatile unsigned *)(t - 0x100))[k];
        params.inst[17] = ((const volatile unsigned *)&params)[(int)(0x34 - PARAMS_C3_OFFSET) / 4];
    }
    if (i >= params.n) return;
    const float4 q = params.q[i];
    unsigned dummy = 0;
    optixTrace(params.handle, make_float3(q.x, q.y, q.z), make_float3(0.f, 0.f, 1.f), 0.f, 1e-16f, 0.f, OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE, 0, 1, 0, dummy);
}

extern "C" __global__ void __miss__ms() {}

extern "C" __global__ void __intersection__is()
{
    const unsigned i = optixGetLaunchIndex().x, prim = optixGetPrimitiveIndex();
    { const unsigned n = params.ncand[i]; if (n < 256) params.candlog[(size_t)i * 256 + n] = prim; params.ncand[i] = n + 1; }
    const float4 p = params.pts[prim], q = params.q[i];
    const float dx = p.x - q.x, dy = p.y - q.y, dz = p.z - q.z;
    const float d2 = fmaf(dz, dz, fmaf(dy, dy, dx * dx));        // the same operations in the same order as the rehosted kernel's inline test
    if (d2 <= params.r2) {
        const unsigned c = params.nbr[i * 65];
        if (c < 64) params.nbr[i * 65 + 1 + c] = prim;
        params.nbr[i * 65] = c + 1;
    }
}
