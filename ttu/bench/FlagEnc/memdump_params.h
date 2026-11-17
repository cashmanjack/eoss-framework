#pragma once
#include <optix_types.h>

struct Params {
    unsigned int *out;
    OptixTraversableHandle handle;
    unsigned int w, h;
};
