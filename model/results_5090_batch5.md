# RTX 5090 (pc14) session 5: Memory_1 against Memory_2 at one ray (F414), scored against model/predictions_5090_batch5.md part A (commit a6923fd, before the run), 2026-11-19

Run: ttu/bench/pc14/mem1b_pc14.py, checked m1 builds (check_seq ALL PASS, rebuilt by the script), stage 0 first (2 launches, then with --smem), one stage 1 launch (1 warp, ray 0), then 16 + 8 rays x 2 carve outs x 11 launches (first dropped), median over launches and warps. nvidia-smi check before every process (run.py), timeout -k 2 30, GPU 0. No fault, hang, timeout or wrong exit. Raw (private): notes/pc14_core/mem1b_20261119T124632.

## Reading of the old data (no GPU)
The "1.08 at one ray" of the Memory_2 latency script is the R = 16 row, not R = 1. The script's R = 1 row is 1.24 (1,718 against 2,138 cycles, median of 4 pools); pool by pool 1.55 (seed 2026), 0.99, 1.00, 1.37. Memory_1 used only ray 0 of seed 2026 pool (one warp), which is the pool with the slowest ray. (The F413 and results_5090_core.md texts carried the wrong row.)

## T1: one warp, one ray per launch, rays 0 to 15 of the seed 2026 pool
| ray | big L1 cycles | small L1 | ratio |
|---|---|---|---|
| 0 | 2,529 | 6,045 | 2.39 |
| 1 | 5,886 | 17,156 | 2.91 |
| 2 | 958 | 958 | 1.00 |
| 3 | 1,149 | 1,149 | 1.00 |
| 4 | 1,564 | 2,181 | 1.39 |
| 5 | 976 | 976 | 1.00 |
| 6 | 1,008 | 1,008 | 1.00 |
| 7 | 955 | 955 | 1.00 |
| 8 | 2,704 | 3,865 | 1.43 |
| 9 | 1,223 | 1,223 | 1.00 |
| 10 | 1,148 | 1,148 | 1.00 |
| 11 | 4,908 | 12,251 | 2.50 |
| 12 | 1,873 | 2,506 | 1.34 |
| 13 | 1,348 | 1,613 | 1.20 |
| 14 | 1,553 | 2,432 | 1.57 |
| 15 | 1,575 | 1,833 | 1.16 |
Median ratio 1.18 (min 1.00, max 2.91). Spearman between big L1 time and ratio 0.90 (p 2e-6); with the excess cycles 0.92. Seven rays (all with a second trace under about 1,150 cycles) have identical medians in both carve outs; the ratio starts above 1 at about 1,350 cycles.

## T2: 16 warps (512 lanes) on one ray, rays 0 to 7
| ray | big | small | ratio | T1 ratio |
|---|---|---|---|---|
| 0 | 2,657 | 4,098 | 1.54 | 2.39 |
| 1 | 19,761 | 20,648 | 1.04 | 2.91 |
| 2 | 1,001 | 995 | 0.99 | 1.00 |
| 3 | 1,209 | 1,185 | 0.98 | 1.00 |
| 4 | 1,641 | 1,829 | 1.11 | 1.39 |
| 5 | 998 | 983 | 0.98 | 1.00 |
| 6 | 1,012 | 1,032 | 1.02 | 1.00 |
| 7 | 961 | 949 | 0.99 | 1.00 |
T2 ratio is at most the T1 ratio in 7 of 8 rays (ray 6: 1.02 against 1.00, within noise); Spearman of the two ratios 0.79. Ray 0 gives 1.54: the Memory_2 value for the same ray (1.55). Unexplained and not pursued: ray 1 with 16 warps has a big L1 second trace of 19,761 cycles against 5,886 with one warp (3.4 times); no other ray shows that.

## Scoring
1. Ray 0 at 1 warp 2.2 to 2.5: HELD (2.39).
2. Smallest at most 1.10 and largest at least 2.0: HELD (1.00 and 2.91).
3. Spearman at least 0.5: HELD (0.90).
4. Median between 1.05 and 1.60: HELD (1.18).
5. T2 ray 0 between 1.3 and 1.8: HELD (1.54); below T1 for ray 0: HELD; at most T1 in at least 6 of 8: HELD (7 of 8).
6 predictions of 6 held.

## Which is right
Both measured what they say, on different rays. Memory_1's one ray number (2.35, reproduced as 2.39) is correct for ray 0, which is among the three longest rays of 16; it is not a typical ray. Memory_2's R = 1 value (1.24) is a median over 4 pools (two with no effect), so it hides the spread. The properties that decide: (a) the ray's path length (here its big L1 second trace time): short paths are not affected at all, long paths up to 2.9 times; (b) the number of warps tracing the same ray: 16 warps soften the slowdown (ray 0: 2.39 to 1.54). The sentence "the reservation slows even one ray 2.35 times" in F413 should read "slows long paths". Consequence for Ada: the Ada result "no effect at one ray" may only be a short ray (the Ada Memory_1 ray choice was not checked here): the Ada versus 5090 difference in F413 is not established.
Not tested: why long paths lose (L1 capacity of the node lines, a node cache minimum, L1 latency with the carve out), the path length itself (node count; big L1 time is the proxy), Ada rays.
