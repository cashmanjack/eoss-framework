#pragma once
#include <optix_types.h>
// Sphere_1: shared by host and device. One point set (centers uniform in [-1,1]^3, one radius), four representations of the same spheres.
struct Params {
    OptixTraversableHandle handle;
    const float3 *centers;            // custom arm intersection program reads the center
    float radius;
    unsigned n, mode;                 // mode 0: ray i from the hash of the launch index; mode 1: one warp, every lane traces the ray id in ray_id
    unsigned ray_id;
    unsigned scene;                   // 0: random rays (origin in the cube, random direction); 1: line rays (+z, origin jittered by jit around the axis, hash of the index)
    float jit;
    unsigned tps;                     // triangles per sphere (1 for sphere and custom arms), turns a primitive index into a sphere index
    uint4 *out;                       // non null: verify launch, per ray (hit, t bits, sphere index, intersection program calls)
    unsigned *sink;                   // one word per warp, written only when an impossible value appears
    unsigned long long *cal;          // mode 0 calibration: 4 words per warp (clock64 start/end, globaltimer start/end, lane 0)
    unsigned long long *cyc;          // mode 1: per lane cycles of the optixTrace call
};
// the ray of index i (same function on host and device)
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif
HD static inline unsigned hash32(unsigned x) { x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16; return x; }
HD static inline float u01(unsigned h) { return (float)(h >> 8) * (1.0f / 16777216.0f) * 2.f - 1.f; }
HD static inline void make_ray(unsigned i, float *o, float *d)
{
    const unsigned h0 = hash32(i * 4u + 1u), h1 = hash32(i * 4u + 2u), h2 = hash32(i * 4u + 3u), h3 = hash32(i * 4u + 4u);
    o[0] = u01(h0); o[1] = u01(h1); o[2] = u01(h2);
    float v0 = u01(h3), v1 = u01(hash32(h3)), v2 = u01(hash32(h3 + 7u));
    float l = 1.0f / sqrtf(v0 * v0 + v1 * v1 + v2 * v2 + 1e-12f); d[0] = v0 * l; d[1] = v1 * l; d[2] = v2 * l;
}
