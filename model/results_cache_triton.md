# Cache_1 / Cache_2 on triton (RTX 6000 Ada, AD102, 142 SMs, 96 MB L2): results, 2026-11-17

Same program and runner as the 4090 run (ttu/bench/Cache_1/cache1.cu, run.py), plain CUDA, no TTU, GPU 1 idle before the run, all modes exit 0, one extra l2lat run in ascending address order. No new predictions were written: this is a repeat of F202 on a second AD102 board and is compared with the 4090 numbers (F202). Raw text is private (notes/cache1_triton).

| item | 4090 (F202) | RTX 6000 Ada (triton) |
|---|---|---|
| L1 hit latency | 40 cycles | 39.9 |
| L2 hit latency (1 MB scan, stride 32 or more) | 285 | 292 (about 2 percent higher) |
| L1 sector | 32 B (miss fraction stride over 32) | same: 71, 103, 166, 292 cycles at stride 4, 8, 16, 32 |
| capacity, no miss up to | 88 KB | 88 KB |
| capacity, half way point (stride 32 and 64) | 100 KB | about 97 KB (145 at 96, 200 at 100) |
| capacity, half way point (stride 128) | 104 KB | 104 KB |
| full miss from | 124 to 128 KB | 124 to 128 KB |
| stride 256 onset | knee at 204 KB (same number of lines) | onset 180 KB, half way about 205 KB: same |
| middle band | 85 to 89 percent miss | 195 to 265 cycles at 104 to 124 KB, i.e. a soft edge, not 100 percent miss |
| L1 associativity (up to 128 addresses, strides 128 B to 1 MB) | no conflict at any stride | no conflict at any stride (maximum 39.9) |
| L1 lines 600 to 1,100 at strides 128 B to 128 KB | knee at the capacity | knee at the capacity (the same lines at every stride) |
| L2 associativity (up to 160 addresses, strides 4 KB to 2 MB) | no knee, 276 to 298 | no knee, 277 to 303 |
| L2 per line latency over 16 MB | 291 to 352, median 319 | 294 to 366, median 325 (excluding the artifact) |
| lines within 15 percent of the median | 100 percent | 100 percent |
| per line values, random against ascending order | r 0.95 | r 0.96 (100,447 lines) |
| address dependence (ascending order, blocks of 4,096 lines) | none (318 in every block) | none (block means 324.8 to 324.9) |
| the 1 in 8 timing artifact | present | present, exactly 12.5 percent in both orders |

Findings in words: every L1 and L2 result of the 4090 repeats on the 6000 Ada within a few percent (same 32 B sector, same 128 B tag line, same soft L1 capacity edge around 96 KB, no way count found in L1 or L2, a graded per line L2 latency with no near and far split). The L2 hit is about 2 percent slower on the 6000 Ada (292 against 285 cycles; 325 against 319 median per line), consistent with the larger 96 MB L2 and 142 SMs. Not measured here: L2 replacement and the L2 ramp beyond the cache size, node state lines, and io (see notes/q_cache/cache_io.sh for the 4070 Ti run, which Jack starts on io GPU 0). The 1 in 8 artifact is also present, so it is a property of this program on both chips, not of one board.
