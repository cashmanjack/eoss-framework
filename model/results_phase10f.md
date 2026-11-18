# Results Phase 10f (tethys, RTX 4090, 2026-11-18)

## Item 3: model v1 on a second chip, synthetic inputs (F255). Predictions model/predictions_phase10f.md (commit 0d36050, before any run of these programs on this GPU)
Programs: ~/thirdparty/div1 builds (RTNN with -p 0 -s 0 -m 0 -k 50, own kNN original and the register heap variant built from notes/q15/diffs with per round CUDA event timing added to the host, Arkade fixed with DEVOUT=1, TrueKNN port), OptiX only, GPU 1 idle before every run, 3 repetitions in the order RTNN, kNN, Arkade, TrueKNN, SM clock sampled (nvidia-smi, 50 ms) in every run (medians 2,520 to 2,745 MHz; the model uses one constant 2.5 GHz). Scored by model/score_phase10f.py (raw in notes/p10f, private). Inputs: seeds as on triton but not the same files.

| row | measured | predicted | measured / predicted | log error | inside factor 2 | SM clock MHz (median) |
|---|---|---|---|---|---|---|
| rtnn_r05 search compute | 0.7453 | 0.191 | 3.90 | +1.36 | NO | 2520 |
| rtnn_r025 search compute | 0.2355 | 0.058 | 4.06 | +1.40 | NO | 2745 |
| knn_orig round 1 | 0.427 | 0.733 | 0.58 | -0.54 | yes | 2745 |
| knn_orig round 2 | 0.6103 | 1.086 | 0.56 | -0.58 | yes | 2745 |
| knn_orig round 3 | 2.161 | 3.884 | 0.56 | -0.59 | yes | 2745 |
| knn_orig round 4 | 6.648 | 12.22 | 0.54 | -0.61 | yes | 2745 |
| knn_orig rounds 1 to 4 total | 9.846 | 17.92 | 0.55 | -0.60 | yes | 2745 |
| knn_reg round 1 | 0.1147 | 0.183 | 0.63 | -0.47 | yes | 2745 |
| knn_reg round 2 | 0.1331 | 0.25 | 0.53 | -0.63 | yes | 2745 |
| knn_reg round 3 | 0.3891 | 0.767 | 0.51 | -0.68 | yes | 2745 |
| knn_reg round 4 | 1.576 | 3.26 | 0.48 | -0.73 | NO | 2745 |
| knn_reg rounds 1 to 4 total | 2.213 | 4.46 | 0.50 | -0.70 | NO | 2745 |
| register heap speedup (orig / reg, rounds 1 to 4) | 4.449 | 4.02 | 1.11 | +0.10 | yes | 2745 |
| arkade fixed 100k queries, GT span (committed basis F151 0.18 x 1.109) | 0.6083 | 0.2 | 3.04 | +1.11 | NO | 2745 |
| trueknn 400k total of round spans | 13.84 | 12.95 | 1.07 | +0.07 | yes | 2520 |

median absolute log error over 15 rows: 0.61; inside factor 2: 10 of 15
extra, unscored: arkade 10,000 queries CUDA event ms 0.1434; 100,000 queries CUDA event ms 1.0533; triton F207 ref fixed (pinned output) GT span 0.5714 ms x 1.109 = 0.634 (4090 measured 0.6083)
trueknn per round spans (ms, median): 0.8131 1.056 3.509 7.841 0.6175

Predictions against outcome:
1. RTNN both rows inside a factor 2 (60 percent): MISSED, measured 3.9 and 4.1 times the prediction (my expectation of "above predicted" held, my 1.0 to 2.5 range missed).
2. kNN original all four rounds inside a factor 2 (60 percent): HELD (0.54 to 0.58, but all BELOW the prediction); register heap: rounds 1 to 3 inside, round 4 (0.48) and the total (0.496) just outside: MISSED at the edge. Total within 25 percent: MISSED (0.55 and 0.50).
3. Register heap speedup between 3.0 and 5.5: HELD (4.45 against 4.02 on triton).
4. Arkade fixed inside a factor 2: MISSED (3.0, measured above).
5. TrueKNN total inside a factor 2: HELD (1.07); above predicted: held (just).
6. At least 6 of the 8 headline rows (there are 7: RTNN 2, kNN original total, kNN register heap total, speedup, Arkade, TrueKNN) inside a factor 2: MISSED, 3 of 7 (speedup, kNN original total, TrueKNN; the register heap total is 0.496). Median absolute log error over all 15 rows 0.61 (10 of 15 inside a factor 2).
What this says: (a) the scaling by SM count and the cycle form carries a RATIO (register heap speedup 4.45 against 4.02) and TrueKNN (1.07), but absolute kNN launch times on the 4090 are 1.8 to 2.1 times SHORTER than triton's times scaled by 142 / 128 (so the 4090 is faster per SM than that, about 1.8x, beyond its 10 percent clock lead: not explained, memory system or L2 differences are candidates, not tested); (b) the RTNN form without a callback calibration is a lower bound by about 4 times on this chip too (F250 on triton: same direction); (c) the Arkade basis I committed (F151, 0.18 ms GT span) was the wrong triton number to scale: triton's F207 reference launch of the same shape (fixed build, GT span 0.5714 ms, pinned output) scaled by 1.109 is 0.634 against 0.608 measured (ratio 0.96): that is a post hoc observation, not a prediction, and the two triton numbers differ by 3 times for reasons I did not resolve (F151 device buffer on a different build). Notes: the register heap variant was checked only for equal active counts per round (200,000 / 200,000 / 200,000 / 197,755 / 46 in both), not for bit identical heaps; the 4090 kNN points are another random file than F138's; the first start of the script failed on NVRTC (LD_LIBRARY_PATH overwritten), no data (kept in notes/p10f/failed).
