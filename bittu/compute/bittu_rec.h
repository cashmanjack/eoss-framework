// Shared by the tool and its injected device functions: one record per thread and group instance.
#pragma once
#include <stdint.h>
#define BITTU_MAXR 24      // result registers kept per thread (TTULD destinations)
#define BITTU_INST 2       // group instances kept per thread
struct BittuRec { uint64_t t0a, t0b, t1; uint32_t ray[8]; uint32_t res[BITTU_MAXR]; uint32_t mask_pre, mask_post; };
