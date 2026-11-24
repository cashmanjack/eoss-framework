# TTUCCTL as a profiler: worked example (predictions model/predictions_cctl_profiler.md, committed first, 9bd6044)

2026-11-24, triton GPU 1 (TTU_GPU=1, idle before), the three builds that already ran here (single session, two sessions of the same ray, two sessions with one TTUCCTL.IVALL between them), check_seq ALL PASS on all six builds (stage 0 and 1), stage 0 of each (30 launches, first by mistake on GPU 0, which has no TTU code in stage 0; rerun on GPU 1 before any stage 1), one stage 1 launch per arm at 16 and 1,024 triangles, then 3 repetitions of 30 launches per cell. 0 faults, exactly one TTU block per launch in every launch. Raw: notes/cctl1/cctl1_20261124T095727.json (private).

Method: C = 1 - T_keep / T_flush, T_keep = cycles of phase 2 with nothing between the phases, T_flush = with one TTUCCTL.IVALL between them.

| triangles (BVH size) | single | phase 2 keep | phase 2 flush | flush / single | C (prediction, accept) | C per repetition |
|---|---|---|---|---|---|---|
| 1,024 (68 KB) | 100,597 | 38,231 | 100,311 | 0.997 | 0.619 (0.62, 0.56 to 0.68) | 0.620 0.618 0.619 |
| 2,048 (132 KB) | 176,039 | 80,807 | 176,166 | 1.001 | 0.541 (0.54, 0.47 to 0.61) | 0.544 0.542 0.540 |
| 4,096 (263 KB) | 369,880 | 283,939 | 369,901 | 1.000 | 0.232 (0.23, 0.15 to 0.31) | 0.233 0.234 0.231 |

All five predictions held: T_flush within 1 percent of T_single (largest 0.3 percent), C inside all three ranges, the order, spread over repetitions at most 0.004 (limit 0.02). The profiler needs no change to the program except the one instruction.
Reading: with phase 2 the same ray as phase 1, 62 percent of its time at 68 KB (54 at 132 KB, 23 at 263 KB) is state that an earlier session left behind, and one TTUCCTL puts it back to a fresh session. It reproduces F150, F162 and F256 with a new build set and a new check of all builds, so the profile is not a new measurement of the cache; it is the method written down with its resolution (0.004 here).
Limits (stated before the run): the same ray is the best case, so C is an upper bound for a multi query run whose queries overlap less; TTUCCTL also empties the L1 data cache (F142), so C is node state plus L1 lines, not nodes alone; one warp, one SM; a TTUCCTL inside a loop over different rays is a new sequence class, not validated, not run.
