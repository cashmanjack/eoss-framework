# RTX 5090 (pc14): model v1 on Blackwell, the four workload programs (F415), scored against model/predictions_5090_batch5.md part B (commit a6923fd, before any run), 2026-11-19

Programs: OptiX only (no TTU code of ours), rebuilt for sm_120 from the triton recipes (notes/rebuild, patches q15 for the register heap; paths and arch only, outside git, /scratch/tgrogers-pc14-disk01/cashman3/thirdparty/div1_sm120). Changes against the recipes: RTNN CMake arch sm_89 to sm_120 and compute_120 (also NVRTC flags) and the cxx wrapper arch list; Arkade and TrueKNN CMAKE_CUDA_ARCHITECTURES 120; the own kNN host got per round CUDA event timing around optixLaunch (printed as "LAUNCH round N active M ms X", the line the Phase 10f scorer reads; the 4090 host copy with this line was not saved, so it is a re-implementation); the register heap patch (knn_anyhit_reg.diff) applied with a hunk offset of 6 lines (compiled, not compared bit for bit with the earlier build). Inputs: u100000, u400000 and ark_a1e5 are the triton files (copied, same seeds); the kNN file u200000 (seed 21) and ark2e5 (200,000 2D points, seed 11, first 110,000 rows equal ark_a1e5) are regenerated (the 4090 runs also used other files than F138; generation of their files is not recorded). Each run: pc14_gpucheck.sh first, timeout -k 2 60, GPU 0, SM clock sampled at 50 ms (2,595 MHz in all), 3 repetitions, medians. A first one repetition run (timestamp 20261119T145736) is kept: its Arkade 100,000 query run stopped with "Insufficient file size" (queries are read after the points: file too short, no GPU work), fixed with the 200,000 point file, then the 3 repetitions were run; only those are scored. No fault.

Prediction file basis: kNN rows from the 4070 Ti F138 times x 60 / 170 (v1) and with the F257 two parameter fit (v1+F257); Arkade from the corrected triton reference launch (known from the 4090 scoring, disclosed); RTNN from the CPU counts of the 4090 predictions (same input structure).

Scorer: model/score_5090_workloads.py. Output:

knn_orig round 5 (active 66, not predicted) ms 0.4527
knn_reg round 5 (active 66, not predicted) ms 0.3677
| row | measured | v1 predicted | ratio | inside 2 | v1+F257 | ratio | inside 2 | SM MHz |
|---|---|---|---|---|---|---|---|---|
| rtnn_r05 search compute | 0.5772 | 0.144 | 4.01 | NO | | | | 2595 |
| rtnn_r025 search compute | 0.1877 | 0.044 | 4.27 | NO | | | | 2595 |
| knn_orig round 1 | 0.2248 | 0.233 | 0.96 | yes | 0.333 | 0.68 | yes | 2595 |
| knn_orig round 2 | 0.3048 | 0.346 | 0.88 | yes | 0.476 | 0.64 | yes | 2595 |
| knn_orig round 3 | 0.9941 | 1.236 | 0.80 | yes | 1.616 | 0.62 | yes | 2595 |
| knn_orig round 4 | 3.141 | 3.889 | 0.81 | yes | 5.012 | 0.63 | yes | 2595 |
| knn_orig rounds 1 to 4 total | 4.665 | 5.704 | 0.82 | yes | 7.437 | 0.63 | yes | 2595 |
| knn_reg round 1 | 0.0882 | 0.058 | 1.52 | yes | 0.087 | 1.01 | yes | 2595 |
| knn_reg round 2 | 0.1078 | 0.079 | 1.36 | yes | 0.111 | 0.97 | yes | 2595 |
| knn_reg round 3 | 0.2666 | 0.244 | 1.09 | yes | 0.297 | 0.90 | yes | 2595 |
| knn_reg round 4 | 1.054 | 1.038 | 1.02 | yes | 1.193 | 0.88 | yes | 2595 |
| knn_reg rounds 1 to 4 total | 1.517 | 1.419 | 1.07 | yes | 1.688 | 0.90 | yes | 2595 |
| register heap speedup | 3.075 | 4.02 | 0.76 | yes | 4.41 | 0.70 | yes | 2595 |
| arkade fixed 100k queries GT span | 0.5041 | 0.553 | 0.91 | yes | | | | 2595 |
| trueknn 400k total of round spans | 10.07 | 10.02 | 1.01 | yes | | | | 2595 |

v1 median abs log error over 15 rows: 0.20; inside factor 2: 13
trueknn per round spans: 0.6034 0.8134 2.599 5.457 0.5869
arkade 10k CUDA event ms 0.1329, 100k 0.5239
Predictions against outcome:
1. RTNN both rows inside a factor 2 (25 percent): MISSED, 4.01 and 4.27 times above (the 4090: 3.9 and 4.06). Measured above predicted (90 percent): held.
2. kNN original all four rounds inside a factor 2 under v1 (40 percent): HELD (0.80 to 0.96). Measured above v1 in rounds 1 and 2 (75 percent): MISSED (0.96 and 0.88, below). v1+F257 inside in all four (60 percent): HELD (0.62 to 0.68), but it is the worse of the two lines here (every round 0.6 to 0.7).
3. kNN register heap all four rounds inside under v1 (35 percent): HELD (1.02 to 1.52, above prediction in all four, short rounds most); v1+F257 (55 percent): HELD (0.88 to 1.01).
4. Register heap speedup between 3.0 and 5.5 (70 percent): HELD at the edge (3.08 against 4.02 on the 4070 Ti and 4.45 on the 4090; v1 4.02 is 0.76 of measured).
5. Arkade inside a factor 2 (50 percent): HELD (0.91). TrueKNN total inside (60 percent): HELD (1.01).
6. Headline rows (RTNN 2, kNN original total, register heap total, speedup, Arkade, TrueKNN) at least 4 inside under v1 (55 percent): HELD (5 of 7; RTNN both outside). At least 6 (15 percent): MISSED.
7. Median absolute log error above 0.5 (60 percent): MISSED (0.20 over 15 rows; 13 of 15 inside a factor 2; the 4090 had 0.61 and 10 of 15, but with the wrong kNN basis).
Reading: the form with the 5090's SM count and L2 latency on latency terms predicts kNN, Arkade and TrueKNN within 0.76 to 1.52 on a different architecture and 170 SMs. RTNN is again 4 times above the lower bound, with almost the same factor as on the 4090 (4.0 and 4.3 against 3.9 and 4.06), so what the frozen form misses for RTNN (the callback calibration, F250) is a chip independent factor, not a chip error. Caveats: Arkade's basis was corrected using what the 4090 scoring showed, so that row is less blind than the rest; the kNN basis is the 4070 Ti (60 SMs) scaled to 170 SMs; the register heap speedup 3.08 is lower than both Ada chips (4.0, 4.45), which no part of the form explains (register heap round 1 and 2 are 1.4 to 1.5 times above prediction while the original is at 0.9, so short launches on 170 SMs cost more for the register heap build); not tested: other structures, the Arkade 10,000 query row (0.133 ms CUDA event, unscored), TrueKNN rounds 1 to 3 (per round spans 0.60 / 0.81 / 2.60 / 5.46 / 0.59 ms).
