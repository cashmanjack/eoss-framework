**SUPERSEDED (2026-11-18, F245): the RT against CLOVER comparison of this file is replaced by results_fair_table.md (one table, RTNN knn mode, TrueKNN, CLOVER hubs and warpwise, every dataset, K 5, 50 and 128, with an exactness column). The RTNN rows below are radius mode, which is not a kNN answer. The CLOVER timings below stand. The TrueKNN rows are the device buffer build of our port.**

# CLOVER (non RT baseline) against the RT methods (2026-11-17, triton GPU 1, plain CUDA for CLOVER, OptiX for RTNN; no TTU code)

Predictions: predictions_clover_vs_rt.md (commit f890048, before any CLOVER timing). Raw: notes/q38/raw (private). Harness: clover_bench.cu (calls CLOVER's own dispatch_knn, its printed span, device synchronized), run script notes/q38/run_clover_rt.sh, summary model/priorworks/summ_clover_rt.py.
CLOVER = github.com/ampslab/clover-knn 03cfaa2; "bitonic patched" = threshold lane (K-1) mod 32 (F221, our patch, not the authors'); "stock" = unpatched. Every row: every point is a query (dragon warpwise: 100,000 random queries x 36.1, linear in queries; bunny check below). Time in ms, [n] = runs (median). Checked: 1,000 sampled queries per run against a CPU double brute force, tie tolerance 1e-6: 0 wrong in every run, every variant (stock bitonic included, at k 5 and 50).

## CLOVER

| dataset | k | queries | warpwise ms | hubs ms | bitonic patched ms | bitonic stock ms | wrong in 1000 sampled (warp / hubs / bit patched / bit stock) |
|---|---|---|---|---|---|---|---|
| 3DRoad | 5 | all points | 7022.1 [1] | 34.2 [3] | 321.7 [3] | 324.3 [3] | 0 / 0 / 0 / 0 |
| Porto 1M | 5 | all points | 37801.1 [1] | 94.4 [3] | 2406.9 [3] | 2420.9 [3] | 0 / 0 / 0 / 0 |
| uniform 1M | 5 | all points | 38849.8 [1] | 89.8 [3] | 2708.2 [3] | 2715.8 [3] | 0 / 0 / 0 / 0 |
| bunny | 50 | all points | 6075.5 [2] | 30.9 [3] | 360.3 [3] | 356.8 [3] | 0 / 0 / 0 / 0 |
| dragon | 50 | all points | 627159.3 (x36.1 from 100000 queries) [1] | 409.7 [1] | 32454.3 [1] | not run | 0 / 0 / 0 / - |

RTNN K 50 rows (rt_rec fresh rebuild, 3 runs; launch = CUDA event of the search launch; "total on device" = sort + geometry + search compute):
  bunny r2: launch 0.214 ms, sort 32.0, geometry 1.94, search compute 0.46, total on device 34.4 ms
  bunny r99: launch 1.032 ms, sort 32.3, geometry 1.92, search compute 1.28, total on device 35.5 ms
  dragon r2: launch 7.646 ms, sort 41.0, geometry 8.97, search compute 10.55, total on device 60.5 ms
  dragon r99: launch 7.553 ms, sort 40.8, geometry 8.97, search compute 10.46, total on device 60.3 ms

## RT rows used (from F207 / F218 / F220 and the exactness check; TrueKNN = fixed build, k 5, our OWL port)

| dataset | TrueKNN launches ms | TrueKNN wall ms (rounds, host copies, refits) | hubs / TrueKNN wall | hubs / TrueKNN launches | warpwise / TrueKNN wall | patched bitonic / TrueKNN wall |
|---|---|---|---|---|---|---|
| 3DRoad | 17.6 | 85.8 | 0.40 (hubs 2.5 times faster) | 1.9 (TrueKNN launches 1.9 times faster) | 82 | 3.7 |
| Porto 1M | 1,068.7 | 1,282.5 | 0.074 (hubs 13.6 times faster) | 0.088 (hubs 11.3 times faster) | 29 | 1.9 |
| uniform 1M | 9.3 | 65.1 | 1.38 (TrueKNN wall 1.4 times faster) | 9.6 | 597 | 42 |

RTNN at k 50 (fresh rebuild of the recorder build; launch = CUDA event of the search launch; total on device = query sort + geometry build + search compute; RTNN radius mode returns at most 50 points inside r, NOT checked to be the 50 nearest):
bunny r 0.0012273 (F207 r2, only 14.6 candidates per query: not a k 50 answer): launch 0.214 ms, total 34.4 ms.
bunny r 0.0062 (99 % of queries have 50 neighbours in r, chosen from a CPU scan of 3,000 queries): launch 1.03 ms, total 35.5 ms; CLOVER hubs 30.9 ms.
dragon r 0.6133 (F207 r2, this radius is already above the 99 % radius 0.587): launch 7.65 ms, total 60.5 ms; CLOVER hubs 409.7 ms; patched bitonic 32,454 ms; warpwise about 627 s.
dragon r 0.5875 (99 % radius): launch 7.55 ms, total 60.3 ms.

## Predictions scored

1. Warpwise ms at k 5: MISSED, high. 7,022 (road, interval 800 to 5,000), 37,801 (Porto, 4,000 to 30,000), 38,850 (uniform, 4,000 to 30,000).
2. Hubs against warpwise: MISSED, far more: 206 times (road), 400 (Porto), 433 (uniform) against the intervals 3 to 50 and 1.5 to 10. Pruning works on uniform 3D at k 5 too.
3. Patched bitonic within 3 times of warpwise: MISSED: 12 to 22 times faster than warpwise at k 5, 17 times at k 50 on bunny. (Stock and patched bitonic time the same: 322 vs 324, 2,407 vs 2,421 ms.)
4. Correct: HELD (0 wrong in 1,000 sampled queries in every run).
5. RT vs CLOVER at k 5: RT wins over warpwise by 82, 29 and 597 times (wall) = HELD (more than 10 on road and uniform, 3 to 30 on Porto: 29). Against hubs: road within a factor 3 = HELD (hubs 2.5 times faster by wall, TrueKNN 1.9 times faster by launches); Porto hubs faster = HELD (13.6 times); uniform "RT wins by more than 2": MISSED by wall (1.38 times), held by launches (9.6 times). CLOVER's three programs 0 wrong in the samples.
6. Bunny warpwise 0.3 to 2 s: MISSED (6.08 s). RTNN not comparable at the F207 radius: HELD as written (and at a radius with 50 neighbours the time is 35.5 ms total, 1.03 ms launch).
7. Dragon warpwise 50 to 600 s: MISSED just above (627 s); hubs 5 to 100 s: MISSED (0.41 s, 12 times below).
8. Scaling: warpwise bunny 36,000 queries 565 ms x 10.06 = 5.69 s against 6.08 s full: 6 percent: HELD.

## Reading

- Against the linear scan baseline (warpwise) the RT methods win by 30 to 600 times. That is the baseline CLOVER's authors compare against; it is not the strongest non RT baseline.
- Against the pruned scan (CLOVER hubs, which builds its index inside the timed span) the RT result is mixed: TrueKNN (k 5) wins launches on road (1.9 times) and uniform (9.6 times) but loses on wall time on road (2.5 times slower), Porto (13.6 times slower, launches 11.3 times slower) and gains only 1.4 times on uniform. RTNN at K 50 wins launches by 30 times (bunny) and 54 times (dragon) but its total on device (query sort 32 to 41 ms and geometry) is 1.15 times slower than hubs on bunny and 6.8 times faster on dragon.
- Where RT wins: large dense surface data (dragon 3.6M), uniform 3D at k 5, launch time alone. Where it loses: clustered 2D data with a long round tail (Porto), small inputs where RTNN's query sort dominates (bunny), and any time the answer must be exact (equal correctness in results_fair_table.md; RTNN's radius mode answers are not the nearest neighbours, see results_rtnn_exact.md).
- Caveats: CLOVER spans exclude host to device copies and the result copy (RTNN total on device excludes the 285 ms result copy for dragon); TrueKNN launches exclude host work, its wall includes it. RTNN and TrueKNN are not the same problem as CLOVER (radius bounded and first K found, or box stop rule). One GPU, one run for the slow cells.
- Observation, not diagnosed: CLOVER bitonic and hubs crash with an illegal memory access when the queries are a random subset of the points (bunny 36,000 of 362,272 for both; 5,000 of 20,000 for bitonic); warpwise accepts subsets. They work when the queries are every point in order, as in the mesh runs. Possibly their API contract, not a bug of ours; not checked in the source.
- Not done: k about sqrt(n); CLOVER on the other RT datasets (Gowalla, Glove 3D, Manuscript, 2D or 3D variants), hubs with K 128 as in CLOVER's mesh runs; kNN quality of RTNN.
