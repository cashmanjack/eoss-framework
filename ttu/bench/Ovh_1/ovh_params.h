#pragma once
#include <optix_types.h>
struct Params {
    OptixTraversableHandle handle;
    unsigned n, nrep, scene;            // scene 0: tri1 rays (origin in a small square, +z), 1: soup (random origin in the cube, random direction)
    unsigned *sink;                     // one word per warp, written only when an impossible value appears
    unsigned long long *cal;            // non null: calibration launch, 4 words per warp: clock64 start/end, globaltimer start/end (lane 0)
};
