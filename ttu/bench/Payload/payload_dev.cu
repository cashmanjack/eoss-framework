// Throughput_10 device code (OptiX only, no TTU code). Compile with -DNPAY=<payload values 0..32> -DNATTR=<attribute values 0..8> -DCUSTOM=<0 triangles | 1 custom AABB primitives>.
// Raygen traces with NPAY payload values (all live: filled by closest hit / miss from t, prim and the attributes, folded into the output), custom mode reports NATTR attributes from the intersection program.
#include <optix.h>
#include <utility>
#ifndef NPAY
#define NPAY 2
#endif
#ifndef NATTR
#define NATTR 2
#endif
#ifndef CUSTOM
#define CUSTOM 0
#endif
struct Ray8 { float o[3], tmin, d[3], tmax; };
struct Out { unsigned hit; float t; unsigned prim; unsigned chk; };
struct Params { const Ray8 *rays; Out *out; const float *boxes; OptixTraversableHandle handle; unsigned n; };
extern "C" __constant__ Params params;

template <size_t... I> __device__ __forceinline__ void trace_n(const Ray8 &r, unsigned *p, std::index_sequence<I...>)
{
    optixTrace(params.handle, make_float3(r.o[0], r.o[1], r.o[2]), make_float3(r.d[0], r.d[1], r.d[2]), r.tmin, r.tmax, 0.f, OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE, 0, 1, 0, p[I]...);
}
extern "C" __global__ void __raygen__rg()
{
    const unsigned i = optixGetLaunchIndex().x; if (i >= params.n) return;
    const Ray8 r = params.rays[i]; unsigned p[NPAY > 0 ? NPAY : 1];
#pragma unroll
    for (int k = 0; k < NPAY; ++k) p[k] = 0u;
    trace_n(r, p, std::make_index_sequence<NPAY>());
    unsigned chk = 0;
#pragma unroll
    for (int k = 0; k < NPAY; ++k) chk = chk * 31u + p[k];
    Out o; o.hit = NPAY > 0 ? p[0] : 0u; o.t = NPAY > 1 ? __uint_as_float(p[1]) : 0.f; o.prim = NPAY > 2 ? p[2] : 0u; o.chk = chk; params.out[i] = o;
}
template <int N> __device__ __forceinline__ void setp(int k, unsigned v)
{
#define SP(i) if (k == i && N > i) optixSetPayload_##i(v);
    SP(0) SP(1) SP(2) SP(3) SP(4) SP(5) SP(6) SP(7) SP(8) SP(9) SP(10) SP(11) SP(12) SP(13) SP(14) SP(15) SP(16) SP(17) SP(18) SP(19) SP(20) SP(21) SP(22) SP(23) SP(24) SP(25) SP(26) SP(27) SP(28) SP(29) SP(30) SP(31)
#undef SP
}
__device__ __forceinline__ void fill(float t, unsigned prim, unsigned hit, unsigned att)
{
#pragma unroll
    for (int k = 0; k < NPAY; ++k) {
        unsigned v = k == 0 ? hit : k == 1 ? __float_as_uint(t) : k == 2 ? prim : (prim * 2654435761u + k * 40503u + __float_as_uint(t));
        if (k >= 3) v ^= att;
        setp<NPAY>(k, v);
    }
}
extern "C" __global__ void __closesthit__ch()
{
#if CUSTOM
    unsigned att = 0;
#if NATTR > 0
    att = optixGetAttribute_0();
#if NATTR > 1
    att += optixGetAttribute_1();
#endif
#if NATTR > 3
    att += optixGetAttribute_3();
#endif
#if NATTR > 7
    att += optixGetAttribute_7();
#endif
#endif
    fill(optixGetRayTmax(), optixGetPrimitiveIndex(), 1u, att);
#else
    fill(optixGetRayTmax(), optixGetPrimitiveIndex(), 1u, __float_as_uint(optixGetTriangleBarycentrics().x));
#endif
}
extern "C" __global__ void __miss__ms() { fill(0.f, 0u, 0u, 0u); }
#if CUSTOM
extern "C" __global__ void __intersection__is()
{
    const unsigned pi = optixGetPrimitiveIndex(); const float *b = params.boxes + 6 * pi;
    const float3 o = optixGetObjectRayOrigin(), d = optixGetObjectRayDirection(); const float tmin = optixGetRayTmin(), tmax = optixGetRayTmax();
    float t0 = tmin, t1 = tmax;
#pragma unroll
    for (int a = 0; a < 3; ++a) {
        const float oa = a == 0 ? o.x : a == 1 ? o.y : o.z, da = a == 0 ? d.x : a == 1 ? d.y : d.z, inv = 1.f / da;
        float tn = (b[a] - oa) * inv, tf = (b[3 + a] - oa) * inv; if (tn > tf) { const float s = tn; tn = tf; tf = s; }
        t0 = fmaxf(t0, tn); t1 = fminf(t1, tf);
    }
    if (t0 > t1) return;
#if NATTR == 0
    optixReportIntersection(t0, 0);
#elif NATTR == 1
    optixReportIntersection(t0, 0, pi);
#elif NATTR == 2
    optixReportIntersection(t0, 0, pi, pi + 1);
#elif NATTR == 4
    optixReportIntersection(t0, 0, pi, pi + 1, pi + 2, pi + 3);
#elif NATTR == 8
    optixReportIntersection(t0, 0, pi, pi + 1, pi + 2, pi + 3, pi + 4, pi + 5, pi + 6, pi + 7);
#else
#error NATTR must be 0, 1, 2, 4 or 8 in custom mode
#endif
}
#endif
