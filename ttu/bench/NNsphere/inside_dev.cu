// NNsphere A: does any OptiX 8.1 option make the built in sphere report a hit for a ray that starts inside? (predictions: model/predictions_5090_batch10.md part A). OptiX only.
#include <optix.h>
struct Params { OptixTraversableHandle handle; const float4 *org; const float4 *dir; unsigned rayflags; float tmin; float4 *res; };   // res[i] = (chit hit, chit t, ah calls, ah first t) and res[n+i] = (ch hit kind, ch backface, ah first backface, ah first hitkind)
extern "C" __constant__ Params params;
extern "C" __global__ void __raygen__rg()
{
    const unsigned i = optixGetLaunchIndex().x; const float4 o = params.org[i], d = params.dir[i];
    unsigned p0 = 0, p1 = 0, p2 = 0, p3 = 0, p4 = 0, p5 = 0, p6 = 0, p7 = 0;
    optixTrace(params.handle, make_float3(o.x, o.y, o.z), make_float3(d.x, d.y, d.z), params.tmin, 1e16f, 0.f, OptixVisibilityMask(255), params.rayflags, 0, 1, 0, p0, p1, p2, p3, p4, p5, p6, p7);
    params.res[i] = make_float4(__uint_as_float(p0), __uint_as_float(p1), (float)p2, __uint_as_float(p3));
    params.res[gridDim.x * 0 + 4096 + i] = make_float4((float)p4, (float)p5, (float)p6, (float)p7);
}
extern "C" __global__ void __miss__ms() {}
extern "C" __global__ void __closesthit__ch()
{
    optixSetPayload_0(__float_as_uint(1.f)); optixSetPayload_1(__float_as_uint(optixGetRayTmax()));
    optixSetPayload_4(optixGetHitKind()); optixSetPayload_5(optixIsBackFaceHit() ? 1u : 0u);
}
extern "C" __global__ void __anyhit__ah()
{
    const unsigned c = optixGetPayload_2();
    if (c == 0) { optixSetPayload_3(__float_as_uint(optixGetRayTmax())); optixSetPayload_6(optixIsBackFaceHit() ? 1u : 0u); optixSetPayload_7(optixGetHitKind()); }
    optixSetPayload_2(c + 1u);   // accept the hit (no ignore): the closest hit program then runs when enabled
}
