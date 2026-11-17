# Knee map: results (2026-11-17, tethys GPU 1, RTX 4090; predictions model/predictions_knee.md committed first, 068a276)

Stage 0 (largest scene, 32 warps, no TTU) clean, smallest TTU launch clean, 96 runs (4 sizes x W = 4 8 16 32 x g = 1 2 4 8 16 32), all exit 0, one block each, no fault, GPU idle after. Same rays as F199 design B (regenerated from the same seeds: same classes 6/3/5, 9/3/7, 12/4/9, 15/5/10). Raw notes/unit3g/k_*.json (private); analysis model/analyze_knee.py, notes/unit3g/analysisK.txt. Rebuild script ttu/bench/Unit_3/rebuild.sh.

## Measured c (cycles per ray per SM), g = 1 2 4 8 16 32; knee g (none = top / floor under 1.5)
| soup | W = 4 | W = 8 | W = 16 | W = 32 |
|---|---|---|---|---|
| 1,000 | 34.1 35.8 35.7 36.4 37.7 40.1: none | 23.1 24.0 23.8 23.5 24.4 26.4: none | 38.7 18.7 16.0 14.9 15.7 18.3: 1.42 | 65.3 18.5 11.6 10.5 10.9 12.8: 1.48 |
| 10,000 | 41.6 43.8 45.0 45.8 47.0 47.2: none | 27.1 27.6 29.3 30.0 30.0 30.4: none | 67.6 47.7 23.1 21.3 18.8 19.7: 2.24 | 77.4 75.6 19.6 14.4 12.6 13.0: 2.91 |
| 100,000 | 54.7 56.2 56.0 56.2 56.6 58.6: none | 33.9 35.0 35.1 35.3 35.4 39.5: none | 72.1 58.6 28.1 24.1 23.0 26.2: 2.48 | 85.9 79.0 37.5 20.9 16.0 17.9: 3.15 |
| 1,000,000 | 67.6 66.4 67.9 67.0 67.7 68.5: none | 41.1 41.3 42.0 41.3 41.4 42.5: none | 72.7 66.9 44.4 29.9 29.5 28.8: 3.29 | 82.7 80.8 54.6 25.7 22.6 20.2: 4.31 |

## Against the predictions (configurations within a factor 2 of the measured knee, 15 scored, calibration soup:100,000 W = 32 excluded)
- H as stated (distinct node lines per step, step_max, C = 9.5 from soup:100,000): 5 of 15, size trend FAILS (it predicts g 3.37 everywhere, measured 1.5 to 4.3 at W = 32): MISSED. The step based count is 32 / g at every size, so it cannot see the size dependence, and it has no warp count.
- step_mean: 7 of 15, trend fails. total (distinct nodes per warp over the whole traversal): 7 of 15, trend ok. step_max x W: 14 of 15 within a factor 2 but trend FAILS (constant 3.37 at W = 32 against 1.5 to 4.3). total x W: 11 of 15, trend ok. No statistic passes rules 1 and 2. (My probability that some statistic passes was 0.4; that the knee depends on W 0.5: held, it disappears at W = 4 and 8; that the floor is similar at all W 0.3: missed, 12.8 to 40 at soup:1,000.)

## What the knee tracks (rule 3, post hoc, 8 configurations with a knee)
- No knee at W = 4 and 8: the cost per ray does not depend on the number of distinct rays at all (within 10 percent; soup:1,000 W = 4: 34 to 40, soup:1,000,000: 66 to 69). At W = 4 the launch span is 1.09 to 1.18 times the one warp latency (F199 design B 3,707 / 4,595 / 6,407 / 7,679 cycles) at every g: four warps in flight are latency bound, whatever they fetch; at W = 8 1.35 to 1.6 times, still flat in g.
- The distinct ray penalty (and with it the knee) only exists at W = 16 and 32, where the unit is the bottleneck. There the knee g is similar at 16 and 32 warps (0.75 to 0.97 times) and rises with scene size (1.4 to 4.3).
- Distinct node lines at the knee (log2 D, spread over the 8 configurations): step_mean 3.35 (sd 0.32, about 10 lines per step, spread a factor 1.25), step_max 3.66 (sd 0.49), total 6.39 (sd 0.55), step_max x W 8.16 (0.59), total x W 10.89 (0.66). So the location of the knee at W >= 16 is best described by the MEAN number of distinct node lines a warp requests per step (about 10), which follows the size trend that the maximum per step misses; that is post hoc, 8 points, three statistics tried.
- H therefore holds in part: the quantity the knee tracks is distinct lines per step per warp against a capacity of about 10 lines (mean over steps), but only when the unit is the bottleneck (W >= 16); below that the latency of the warps sets the cost and sharing does nothing. The capacity form without a warp count is wrong.

## Not determined
Why W = 8 has no knee but W = 16 does (a bottleneck switch between latency and the unit; a W between 8 and 16, e.g. 12, was not run); whether the capacity of about 10 lines per step is a unit queue, a request merge window or an L1 bank effect; distinct rays that share a path (F28); all warps ran the same 32 rays (not workload like in the cross warp sense).
