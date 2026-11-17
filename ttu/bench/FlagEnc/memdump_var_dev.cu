// memdump device program with compile time ray constants (flag encoding study). Same as ttu/optix_dump/memdump_dev.cu
// except FLAGS, MASK, TMAX, TMIN come from -D.
#include <optix.h>
#include "memdump_params.h"
#ifndef FLAGS
#define FLAGS OPTIX_RAY_FLAG_NONE
#endif
#ifndef MASK
#define MASK 255
#endif
#ifndef TMAX
#define TMAX 1e16f
#endif
#ifndef TMIN
#define TMIN 0.f
#endif
extern "C" __constant__ Params params;
extern "C" __global__ void __raygen__rg()
{
    const uint3 idx = optixGetLaunchIndex();
    const float u = (idx.x + 0.5f) / params.w * 2.f - 1.f;
    const float v = (idx.y + 0.5f) / params.h * 2.f - 1.f;
    unsigned int p0 = 0;
    optixTrace(params.handle, make_float3(u, v, -1.f), make_float3(0.f, 0.f, 1.f),
               TMIN, TMAX, 0.f, OptixVisibilityMask(MASK), FLAGS, 0, 1, 0, p0);
    params.out[idx.y * params.w + idx.x] = p0;
}
extern "C" __global__ void __miss__ms() { optixSetPayload_0(0u); }
extern "C" __global__ void __closesthit__ch() { optixSetPayload_0(1u + optixGetPrimitiveIndex()); }
