# R4, R5 and BITTU on the rehosted kernel (2026-11-24/25, triton, GPU 1)
Predictions first: model/predictions_rehost_r4_r5.md (71922ec). Raw (private): notes/rehost/r4_*, r5_*, notes/rehost/bittu/. No fault, no hang, in any run.
## R4: many warps, exact at every step
Stage 0 (injected answers, counts up to 63, 5 rounds) first at all 10 geometries, 0 bad lanes in each. Stage 1 (65,536 points, r 0.03, seed 11, one query per lane, 3 launches each): blocks x warps = 1x2, 1x4, 1x8, 1x16 (the one SM maximum, see below), 4x16, 16x16, 71x8, 142x8, 142x16 (72,704 queries), 284x8 (72,704 queries). Every query equals OptiX's and the CPU grid search's (0 bad, 0 unfinished). "32 warps on one SM" was not possible: the kernel declares 128 registers per thread and an SM has 65,536, so 16 warps (512 threads) is the most one SM holds.
Sessions per query 13.3 to 13.6 across all geometries (within 3 percent: prediction 2 held). Cycles in sessions per session: 1x8 2,541, 142x8 3,425 (1.35 times, prediction 3 held at the low edge of 1.3 to 3.5). Lane efficiency 0.61 to 0.63.
## R5: timing against OptiX and RTNN (queries are the points themselves, 8 warps per block)
Timing build (no clock reads, no last answer slots, check_seq ALL PASS): exact in all five inputs (up to 262,144 queries, 0 bad against OptiX and the CPU; 6 boundary pairs ambiguous in BIG by float rounding, as in the harness definition).
| input | N | r | ours ms | OptiX ref ms | RTNN search compute ms | ours ns/query | ours / OptiX | RTNN / ours |
|---|---|---|---|---|---|---|---|---|
| C1 | 4,096 | 0.08 | 0.0513 | 0.0553 | 0.0826 | 12.5 | 0.93 | 1.61 |
| C2 | 4,096 | 0.12 | 0.1206 | 0.1208 | 0.1202 | 29.4 | 1.00 | 1.00 |
| C3 | 65,536 | 0.03 | 0.1209 | 0.0881 | 0.1391 | 1.84 | 1.37 | 1.15 |
| C4 | 1,024 | 0.2 | 0.1164 | 0.1044 | 0.1267 | 113.7 | 1.12 | 1.09 |
| BIG | 262,144 | 0.021 | 0.6298 | 0.3676 | 0.3513 | 2.40 | 1.71 | 0.56 |
Ours: median of 6 launches (first dropped); OptiX reference of this harness (same GAS, same intersection expression): median of 20; RTNN (K 50, radius mode, no partition, no sort): median of 3 of its own "search compute" (its separate sort/partition step takes 31 ms and is not included; the program needs RTNN_DRAM_GB set on this GPU). Small launches (C1, C4, C2) are mostly launch and tail effects, not per query cost. On the larger inputs our rehosted kernel is 1.4 to 1.7 times slower than OptiX's own and 1.15 (C3) to 1.8 (BIG) times slower than RTNN's search kernel: no win.
Where the time goes (mean cycles per lane in the loop, from clock instrumented builds, 8 warps; "inline test" = full build minus the build without it, whose answers are wrong so its session counts differ: approximate):
| input | loop | sessions | inline test | rest (clocks, reloads, moves, branches) | answers per query |
|---|---|---|---|---|---|
| C1 | 80,935 | 34% | 25% | 41% | 15.7 |
| C2 | 215,452 | 29% | 31% | 40% | 42.1 |
| C3 | 191,628 | 46% | 33% | 21% | 14.4 |
| C4 | 209,802 | 25% | 33% | 42% | 42.6 |
| BIG | 291,918 | 47% | 32% | 21% | 19.3 |
Sessions are 25 to 47 percent, the inline test about a third, the rest 21 to 42 percent: the unit is not the bottleneck of this kernel, the SASS around it is.
## BITTU on the rehosted kernel (C1, one warp, stage 1 build, exact; build with BITTU_INST 128)
460 group instances in 32 threads, equal to the kernel's own session count in every lane (prediction 10). Per session the tool gives what OptiX never exposed: the ray words, the 22 result registers, the active lane mask, three clocks.
- Count field distribution of the answers: 1 in 312, 2 in 112, 3 in 10 (the earlier trace, same data), 26 miss forms (the last session of 26 of 32 lanes; the last session of the other 6 is a hit form, which the stop rule accepts when no pending entry remains).
- Active lanes: 23 warp executions of the group, mean 20.0 lanes = 0.625 of 32, equal to the kernel's lane efficiency (prediction 11 held); only 192 of 460 lane sessions saw a full warp.
- Sessions per thread 6 to 23, median 13.
- Latency: BITTU's raw group window median 11,717 cycles, probe overhead 1,201, corrected 10,516; the kernel's own uninstrumented window is 1,525 cycles per session (r3_c1), so the probe pair makes a session 7.7 times longer (prediction 13 held) and the loop 8.5 times (382,575 against 45,246 cycles). The corrected window shows no dependence on the answer count (10,516 for counts 1 and 2, 10,667 for 3, 10,015 for misses) or on active lanes (about 9,000 to 11,400): it is the probe, not the unit. Per session latency from BITTU is therefore not usable; counts, masks, order and ray and result words are.
