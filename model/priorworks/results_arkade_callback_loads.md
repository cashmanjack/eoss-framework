# Arkade callback loads under load (float build, device buffer, KN 50, NORM 2; 2026-11-17, triton GPU 1; median of 3 runs; cycles are clock64 SM cycles)

a = dependent load of the primitive data (data_pts[primID]) per callback lane call; b = scan of the 50 result slots per execution (49 dependent loads on the payload); warp call = CB probe: cycles of one whole callback call of a warp (leader lane).
Chase level 0 = none; 1 = 18 blocks, 8 = 144 blocks of 128 threads (one chain per thread, 1 GiB) started after the warm up launch. Chase cycles per step: thread 0 of each block, alone before the launch / during the timed launch.

| dataset | queries | chase | launch ms | a cycles/call | b cycles/exec | whole warp call cycles | a / a(32 queries, no chase) | b / b(32, no chase) | warp call / warp call (32, no chase) | warps per used SM | chase cycles/step alone | during |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| manuscript | 32 | 0 | 0.787 | 502 | 3440 | 3964 | 1.00 | 1.00 | 1.00 | 1.0 |  |  |
| manuscript | 32 | 1 | 1.075 | 1286 | 3602 | 4686 | 2.56 | 1.05 | 1.18 | 1.0 | 1297 | 1304 |
| manuscript | 32 | 8 | 2.236 | 5408 | 4663 | 8046 | 10.77 | 1.36 | 2.03 | 1.0 | 6325 | 6323 |
| manuscript | 320 | 0 | 0.877 | 510 | 3558 | 4058 | 1.01 | 1.03 | 1.02 | 1.9 |  |  |
| manuscript | 320 | 1 | 1.197 | 1296 | 3956 | 4923 | 2.58 | 1.15 | 1.24 | 1.9 | 1300 | 1303 |
| manuscript | 320 | 8 | 2.543 | 5624 | 6051 | 8933 | 11.20 | 1.76 | 2.25 | 1.8 | 6311 | 6353 |
| manuscript | 3200 | 0 | 1.000 | 518 | 4224 | 4554 | 1.03 | 1.23 | 1.15 | 3.6 |  |  |
| manuscript | 3200 | 1 | 1.382 | 1329 | 5028 | 5676 | 2.65 | 1.46 | 1.43 | 3.6 | 1297 | 1357 |
| manuscript | 3200 | 8 | 3.634 | 5807 | 9737 | 11376 | 11.56 | 2.83 | 2.87 | 3.0 | 6326 | 6614 |
| manuscript | 10000 | 0 | 1.025 | 513 | 4213 | 4549 | 1.02 | 1.22 | 1.15 | 3.5 |  |  |
| manuscript | 10000 | 1 | 1.427 | 1380 | 4529 | 5317 | 2.75 | 1.32 | 1.34 | 3.3 | 1298 | 1474 |
| manuscript | 10000 | 8 | 4.120 | 6195 | 10300 | 11848 | 12.33 | 2.99 | 2.99 | 2.8 | 6325 | 7056 |
| manuscript | 40000 | 0 | 2.103 | 678 | 8121 | 8120 | 1.35 | 2.36 | 2.05 | 7.1 |  |  |
| manuscript | 40000 | 1 | 2.227 | 1508 | 8341 | 8781 | 3.00 | 2.42 | 2.22 | 6.8 | 1299 | 1718 |
| manuscript | 40000 | 8 | 6.843 | 7876 | 24521 | 25384 | 15.68 | 7.13 | 6.40 | 6.6 | 6325 | 8151 |
| manuscript | 160000 | 0 | 5.499 | 808 | 9738 | 10268 | 1.61 | 2.83 | 2.59 | 11.7 |  |  |
| manuscript | 160000 | 1 | 6.056 | 1647 | 9825 | 10850 | 3.28 | 2.86 | 2.74 | 11.9 | 1300 | 1991 |
| manuscript | 160000 | 8 | 20.424 | 9957 | 35060 | 37275 | 19.82 | 10.19 | 9.40 | 12.0 | 6312 | 9442 |
| gowalla | 32 | 0 | 179.539 | 491 | 3412 | 4311 | 1.00 | 1.00 | 1.00 | 1.0 |  |  |
| gowalla | 32 | 1 | 222.542 | 1251 | 3660 | 5152 | 2.55 | 1.07 | 1.20 | 1.0 | 1297 | 1298 |
| gowalla | 32 | 8 | 408.443 | 5083 | 5271 | 8893 | 10.36 | 1.54 | 2.06 | 1.0 | 6313 | 6326 |
| gowalla | 320 | 0 | 187.848 | 488 | 3459 | 4351 | 0.99 | 1.01 | 1.01 | 1.8 |  |  |
| gowalla | 320 | 1 | 231.355 | 1222 | 3830 | 5220 | 2.49 | 1.12 | 1.21 | 1.8 | 1300 | 1303 |
| gowalla | 320 | 8 | 436.548 | 4910 | 5491 | 8716 | 10.01 | 1.61 | 2.02 | 1.7 | 6326 | 6364 |
| gowalla | 3200 | 0 | 205.440 | 478 | 3612 | 4464 | 0.97 | 1.06 | 1.04 | 3.3 |  |  |
| gowalla | 3200 | 1 | 246.575 | 1124 | 4062 | 5280 | 2.29 | 1.19 | 1.22 | 3.3 | 1296 | 1320 |
| gowalla | 3200 | 8 | 514.262 | 4674 | 6920 | 9274 | 9.53 | 2.03 | 2.15 | 2.9 | 6321 | 6502 |
| gowalla | 10000 | 0 | 204.548 | 473 | 3601 | 4448 | 0.96 | 1.06 | 1.03 | 3.3 |  |  |
| gowalla | 10000 | 1 | 242.997 | 955 | 3849 | 4934 | 1.95 | 1.13 | 1.14 | 3.1 | 1298 | 1344 |
| gowalla | 10000 | 8 | 513.417 | 4442 | 6823 | 8923 | 9.05 | 2.00 | 2.07 | 2.8 | 6325 | 6768 |
| gowalla | 40000 | 0 | 335.848 | 483 | 5473 | 6069 | 0.98 | 1.60 | 1.41 | 5.9 |  |  |
| gowalla | 40000 | 1 | 435.327 | 819 | 5844 | 6651 | 1.67 | 1.71 | 1.54 | 5.2 | 1298 | 1354 |
| gowalla | 40000 | 8 | 1020.837 | 4590 | 14732 | 15020 | 9.35 | 4.32 | 3.48 | 4.9 | 6312 | 7013 |
| gowalla | 160000 | 0 | 1056.555 | 490 | 8250 | 9384 | 1.00 | 2.42 | 2.18 | 11.5 |  |  |
| gowalla | 160000 | 1 | 1220.823 | 801 | 8280 | 9585 | 1.63 | 2.43 | 2.22 | 11.5 | 1292 | 1282 |
| gowalla | 160000 | 8 | 2701.260 | 4933 | 21262 | 23557 | 10.05 | 6.23 | 5.46 | 11.3 | 6291 | 7355 |

## Probe overhead: the same code without the segment probes (ld0: only the CB probes) against the probe build, no chase

| dataset | queries | launch ms probe build | launch ms without segment probes | warp call cycles probe build | without | probe overhead on a warp call |
|---|---|---|---|---|---|---|
| manuscript | 10000 | 1.025 | 1.016 | 4549 | 4008 | 1.13 |
| manuscript | 160000 | 5.499 | 6.669 | 10268 | 12255 | 0.84 |
| gowalla | 10000 | 204.548 | 185.287 | 4448 | 3912 | 1.14 |
| gowalla | 160000 | 1056.555 | 1004.849 | 9384 | 9087 | 1.03 |

## Share of one warp call taken by (a) + (b) at 32 queries, no chase, and growth of the rest

manuscript q 32: b executes in 0.68 of the lane calls; (a + that share of b) / warp call cycles = 0.72 (lane cycles of a serial lane vs warp call cycles, see note)
manuscript q 10000: b executes in 0.69 of the lane calls; (a + that share of b) / warp call cycles = 0.75 (lane cycles of a serial lane vs warp call cycles, see note)
manuscript q 160000: b executes in 0.69 of the lane calls; (a + that share of b) / warp call cycles = 0.73 (lane cycles of a serial lane vs warp call cycles, see note)
gowalla q 32: b executes in 0.91 of the lane calls; (a + that share of b) / warp call cycles = 0.84 (lane cycles of a serial lane vs warp call cycles, see note)
gowalla q 10000: b executes in 0.93 of the lane calls; (a + that share of b) / warp call cycles = 0.86 (lane cycles of a serial lane vs warp call cycles, see note)
gowalla q 160000: b executes in 0.93 of the lane calls; (a + that share of b) / warp call cycles = 0.87 (lane cycles of a serial lane vs warp call cycles, see note)

## Part A: Glove 3D count mismatch (CPU, model/priorworks/check_ark_dim.py, output notes/q37/ark_dim_20261117.txt)

The Arkade sample used for F207 and F220 (src/s01-knn, also in the pristine repo) builds every sphere and query as (x, y, 0), so it drops the third column; the repo's other sample (s02-withTrueknn, used for F216's reproduction) keeps it. Counts, mean intersection calls per query (x 1.10 hardware factor on the CPU counts), 10,000 queries:

| dataset | CPU 3D (F207 replay) | CPU 2D (x, y only) | recorder | recorder / 3D | recorder / 2D | Spearman recorder vs 3D | vs 2D |
|---|---|---|---|---|---|---|---|
| Glove 3D | 44,173 | 121,899 | 113,154 | 2.56 | 0.93 | 0.79 | 1.0000 |
| Gowalla (z = 0 column) | 13,658 | 13,658 | 12,487 | 0.91 | 0.91 | 1.0000 | 1.0000 |
| Manuscript (z = 0 column) | 162.6 | 162.6 | 157.9 | 0.97 | 0.97 | 0.27 | 0.27 |

Scoring: P_A1 HELD (1.1 x 2D over recorder 1.08; recorder over 2D 0.93), P_A2 HELD (44,172.6 equals F220's replay), P_A3 HELD (median per query ratio 0.93, Spearman 1.0000). Gowalla and Manuscript unchanged, as predicted. Answer: the whole 2.56 is the dropped third column; the remaining 7 percent is the recorder over the 1.10 factor, the same size as on Gowalla (0.91) and Manuscript (0.97). The F207 Glove 3D Arkade rows were predicted from 3D counts for a program that searched the 2D projection, so those two rows are not comparable (F220's decomposition without the count term: fixed 8.69 / 1.66 = 5.2, original 3.26 / 1.67 = 1.95; arithmetic on F220's factors, not a rerun). F216's reproduction (s02, 3D) is unaffected.

## Part B scoring (predictions: predictions_arkade_callback.md, commit 00bd6b5)

- P_B1 (a, the primitive data load): 32 queries 250 to 500 cycles: HELD on Gowalla (491), missed by 2 on Manuscript (502). 10,000 queries 1.0 to 1.6 times: HELD (1.02, 0.96). 160,000 queries 1.3 to 2.5 times: HELD on Manuscript (1.61), MISSED on Gowalla (1.00, no growth). Chase at least 2 times at every Q: HELD for the 18 block chase up to 10,000 queries (2.0 to 2.7 times), MISSED at 40,000 and 160,000 on Gowalla (1.7, 1.6); the 144 block chase gave 9 to 12 times (above my 5 times cap: MISSED), it saturates DRAM (chase 6,300 cycles per step).
- P_B2 (b, the 49 dependent compare loads on the payload): 800 to 2,000 cycles at 32 queries: MISSED (3,440 and 3,412, about 70 cycles per load and compare). Growth at 160,000 queries 1.3 to 3 times: HELD (2.83, 2.42). Chase changes b by under 20 percent: HELD for the 18 block chase at small Q (1.05 to 1.07), MISSED for the 144 block chase (1.4 to 1.5 at 32 queries, up to 6 to 10 times at 160,000).
- P_B3 (a + b are 30 to 60 percent of one callback call): MISSED (72 to 87 percent; b alone is about 3,400 of about 4,000 to 4,300 cycles). Growth from 32 to 10,000 queries explaining under half of the growth of the call: MISSED (a + b account for nearly all of the small growth).
- P_B4 (whole call at 10,000 queries 1.3 to 3 times its 32 query value): MISSED (1.15 Manuscript, 1.03 Gowalla); the alternative "no growth" is nearly what happened. Loaded growth appears only from about 6 warps per SM (40,000 queries: 1.4 to 2.1 times) and at 12 warps per SM (160,000 queries: 2.2 to 2.6 times), mostly in b.
- Probe overhead: the build without the segment probes has warp calls 0.84 to 1.14 of the probe build at 10,000 and 160,000 queries (no consistent direction): the probes do not change the picture, but are inside the numbers (one repeat set per cell, 3 runs).
- Answer: at the paper's 10,000 queries (3.3 warps per SM) the callback's dependent loads are NOT slowed by occupancy (1.03 to 1.15 times the single warp value); they slow down with more warps per SM (b, the payload scan, 2.4 to 2.8 times at 12 warps per SM) and strongly under a DRAM heavy neighbour (a 2.5 times with a light chase, 10 times with a saturating one). So F220's "float per call latency 2.5 to 3.2 times the model" at 10,000 queries is not load: the unloaded float call is already about 4,000 cycles, of which about 3,400 is the 49 load scan of the payload.
- Not done: the same for the double (original) build, other L2 sizes or GPUs, a chase of intermediate strength, whether co resident work from another kernel in a real application resembles the chase, why b costs 70 cycles per load (generic load to local memory, not isolated), TrueKNN or RTNN callbacks.
