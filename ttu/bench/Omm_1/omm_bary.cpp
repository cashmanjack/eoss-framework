// host wrapper for the OptiX micromap helper (the header only has a host path in a plain C++ translation unit)
#include <vector_types.h>
#include <optix_micromap.h>
void omm_bary(unsigned t, unsigned L, float *b) { float2 b0, b1, b2; optixMicromapIndexToBaseBarycentrics(t, L, b0, b1, b2); b[0] = b0.x; b[1] = b0.y; b[2] = b1.x; b[3] = b1.y; b[4] = b2.x; b[5] = b2.y; }
