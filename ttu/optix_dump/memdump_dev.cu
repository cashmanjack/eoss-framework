// OptiX device programs for memdump: one triangle, one ray per launch index.
// Compiled to PTX by build.sh and loaded by memdump at run time.
#include <optix.h>
#include "memdump_params.h"

extern "C" __constant__ Params params;

extern "C" __global__ void __raygen__rg()
{
    const uint3 idx = optixGetLaunchIndex();
    const float u = (idx.x + 0.5f) / params.w * 2.f - 1.f;
    const float v = (idx.y + 0.5f) / params.h * 2.f - 1.f;
    unsigned int p0 = 0;
    optixTrace(params.handle, make_float3(u, v, -1.f), make_float3(0.f, 0.f, 1.f),
               0.f, 1e16f, 0.f, OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE,
               0, 1, 0, p0);
    params.out[idx.y * params.w + idx.x] = p0;
}

extern "C" __global__ void __miss__ms() { optixSetPayload_0(0u); }

extern "C" __global__ void __closesthit__ch() { optixSetPayload_0(1u + optixGetPrimitiveIndex()); }
