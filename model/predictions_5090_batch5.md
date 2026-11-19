# RTX 5090 (pc14), session 5 predictions, 2026-11-19. Written BEFORE any run of the items below on this GPU.

## A. Memory_1 TTU half against the Memory_2 latency script, at one ray (reconciliation)

Reading of existing raw files (no GPU, done before this file): the "1.08 at one ray" quoted for the Memory_2 latency script is the R = 16 row. At R = 1 the script gives 1.24 (median over 4 ray pools: 1,718 against 2,138 cycles), pool by pool 1.55 (seed 2026), 0.99 (2027), 1.00 (2028), 1.37 (2029). Memory_1 used R = 1 from the seed 2026 pool only (ray 0), one warp. Seed 2026 ray 0 is the ray with the largest big L1 time (2,517 to 2,716 cycles, against about 1,000 for seeds 2027 and 2028). Same ray, different warps: 1 warp 2.36 (Memory_1), 16 warps 1.55 (Memory_2). Two candidate causes: (a) which ray (path length), (b) how many warps trace it.

Test (stage 1, checked m1 builds, one warp first, timeout -k 2 30, nvidia-smi before every process): second trace latency, L1 big and --smem 98304, 11 launches (first dropped), median over launches and warps, scene soup:100000, pool seed 2026.
- T1: 1 warp (32 lanes, one ray), rays 0 to 15 of the pool, both carve outs.
- T2: 16 warps (512 lanes, one ray), rays 0 to 7, both carve outs.

Predictions (ratio = small L1 over big L1):
1. Ray 0 at 1 warp reproduces Memory_1 (2.2 to 2.5): 85 percent.
2. T1 ratios differ by ray: smallest at most 1.10 and largest at least 2.0: 75 percent.
3. Across the 16 rays, Spearman between the big L1 latency (a path length proxy) and the ratio at least 0.5: 70 percent.
4. Median T1 ratio over the 16 rays between 1.05 and 1.60: 65 percent.
5. T2 ray 0 ratio between 1.3 and 1.8 (reproduces Memory_2's 1.55): 70 percent. T2 ratio below T1 ratio for ray 0: 80 percent. For rays 0 to 7, T2 ratio at most the T1 ratio in at least 6 of 8: 65 percent.
6. Verdict rule: if 2 and 3 hold, the one ray slowdown of Memory_1 is path specific, Memory_1's R = 1 and 4 points were the worst case ray not a typical one, and neither script was wrong about its own ray. If 2 fails (ratios the same for every ray), the warp count is the cause. Both held: both causes.

## B. Workloads on Blackwell: model v1 (frozen, model/priorworks/FORM_combined_v1.md) with the 5090's measured memory constants

Status when written: the builds (RTNN, own kNN original and register heap, Arkade fixed, TrueKNN port), their patches and the generator seeds live in the triton/tethys home (notes/rebuild, notes/q15/diffs, ~/thirdparty/div1), not on this machine. These predictions are committed first so that the order is on record; the runs need those files. Inputs as on tethys (F255): RTNN u100000 (r 0.05, 0.025, K 50, -p 0 -s 0 -m 0), kNN 200,000 uniform points K 16 r0 0.004 doubling, Arkade 2D 110,000 points r 0.005 100,000 queries (device buffer, float), TrueKNN u400000 k 16 r0 0.004. I have not seen any 5090 timing of these programs.

Form: time = cycles / (SM x CLK), SM 142 to 170, CLK kept at the frozen 2.5e9 (the 5090 runs 2,595 MHz in every sample; not used). Measured memory constants of the 5090 (F403, F409): L1 46 cycles (same as Ada), L2 hit 349 (Ada 290, x 1.20), DRAM 862 (Ada 650, x 1.33). Applied only to latency terms: the chain constants L and the latency bound last rounds (x 1.20); every data set here is under 26 MB so nothing is DRAM bound and the DRAM factor is not used. Throughput terms: SM count only. Rule: within a factor 2 (measured / predicted 0.5 to 2); median absolute log error.

kNN basis: the F138 times are the 4070 Ti's (F257), so the basis is 4070 Ti x 60 / 170 (0.3529). Two lines per kNN row: (v1) pure SM scaling; (v1+F257) the fixed part plus slope fitted on the 4090 in F257 (0.034 ms + 1.28 x scaled original, 0.021 ms + 1.13 x scaled register heap), a pre registered test of whether that fit extrapolates to 170 SMs.

| row | v1 predicted (ms) | v1+F257 (ms) | basis |
|---|---|---|---|
| RTNN search compute u100000 r 0.05 | 0.144 | | (110 + 4.9 x 102.2) x 1e5 / (170 x 2.5e9), counts as in predictions_phase10f.md |
| RTNN search compute r 0.025 | 0.044 | | (110 + 4.9 x 15.3) x 1e5 / (170 x 2.5e9) |
| kNN original rounds 1 to 4 | 0.233 / 0.346 / 1.236 / 3.889 (total 5.70) | 0.333 / 0.476 / 1.616 / 5.012 (total 7.44) | F138 0.661 / 0.979 / 3.502 / 11.02 x 0.3529 |
| kNN register heap rounds 1 to 4 | 0.058 / 0.079 / 0.244 / 1.038 (total 1.42) | 0.087 / 0.111 / 0.297 / 1.193 (total 1.69) | F138 0.165 / 0.225 / 0.692 / 2.94 x 0.3529 |
| register heap speedup, rounds 1 to 4 total | 4.02 | 4.41 | F138 |
| Arkade fixed, 100,000 queries, GT span | 0.553 | | triton F207 reference launch 0.5714 (the corrected basis found in F255, disclosed) x 142 / 170 x 1.16 (callback payload scan, about 80 percent of the call, is L2 latency: 0.2 + 0.8 x 1.20) |
| TrueKNN u400000 total of round spans (5 rounds) | 10.02 | | rounds 1 to 4 11.14 (triton, from F255's 12.95 less 0.60 over 1.109) x 142 / 170 = 9.30, plus round 5 0.60 x 1.20 = 0.72 |

Probabilities:
1. RTNN both rows inside a factor 2: 25 percent (F250, F255: the form without a callback calibration is a lower bound; 3.9 to 4.1 times on the 4090). Measured above predicted: 90 percent.
2. kNN original, all four rounds inside a factor 2 (v1): 40 percent; the 5090 is further below perfect SM scaling than the 4090 was (170 SMs, shorter launches), so measured above v1 in rounds 1 and 2: 75 percent. v1+F257 inside a factor 2 in all four rounds: 60 percent.
3. kNN register heap, all four rounds inside a factor 2 (v1): 35 percent; (v1+F257) 55 percent.
4. Register heap speedup between 3.0 and 5.5: 70 percent.
5. Arkade inside a factor 2: 50 percent. TrueKNN total inside a factor 2: 60 percent.
6. Headline rows (7: RTNN 2, kNN original total, register heap total, speedup, Arkade, TrueKNN) at least 4 inside a factor 2 under v1: 55 percent; at least 6: 15 percent.
7. Cross generation: the 4090 result was 3 of 7 inside. The 5090 is a different architecture; I expect v1 no better than on the 4090: median absolute log error over all scored rows above 0.5: 60 percent.
