# Prior works on their datasets: predictions BEFORE any run (2026-11-17, triton GPU 1)

Parameters the papers do not state are marked CHOSEN and fixed here, before any run. Counts are exact CPU counts on the files we actually have (see the notes on each dataset). Units: ms.

## RTNN (range search, K = 50 repo default, queries = points, flags -p 0 -s 0 -m 0 (the validated configuration of F116, no partitioning, default Morton grid sort), RTNN_DRAM_GB 11.5958 as in F167)

The paper states no radius, K or query count for any dataset (checked in the paper text 2026-11-17). CHOSEN: the only radius the paper mentions is r = 0.1 on Buddha, whose points lie in a 131 cube: r1 = 7.6e-4 x the largest bounding box side, and r2 = 4 x r1 ("above 0.1 the speedup falls"). Not our own scaled units of the Stanford files: bunny scans merged with bun.conf transforms, 362,272 points; Dragon = xyzrgb_dragon vertices, 3,609,600 points. Buddha (4.6M) and KITTI 1M/12M/25M, Millennium: not available.

| dataset | N | r | cube cand / query (C) | sphere hits (S) | C_eff | predicted ms |
|---|---|---|---|---|---|---|
| bunny | 362272 | 0.00030682 | 1.01 | 1.00 | 1.01 | 0.123 |
| bunny | 362272 | 0.0012273 | 13.32 | 10.14 | 13.32 | 0.191 |
| dragon | 3609600 | 0.15333 | 10.38 | 6.79 | 10.38 | 1.737 |
| dragon | 3609600 | 0.6133 | 172.18 | 113.95 | 75.33 | 5.297 |

What can fail: more than a factor 2 off (the model is fitted on the 4070 Ti with 1e5 to 1e6 point uniform clouds, no miss term: the dragon BVH is about 250 MB, beyond the 96 MB L2); surface data has far fewer candidates per box than the uniform clouds of F116.

## TrueKNN (OWL port in /dev/shm/pw, original = managed memory frame buffer read by the host, fixed = device buffer; the distance is already float: no float fix applies)

Paper parameters: first n points, k = 5 and k = sqrt(n), start radius = the smallest distance among the 4 nearest neighbours of 100 random points (we take the smallest NONZERO distance, since duplicates give 0: CHOSEN), radius doubles, time per round. Files: 3DRoad (repo file, 434,874 lines, paper 430K), Porto first 1,000,000 points of train.csv (which 1M the paper used: not stated, CHOSEN), UniformDist 1M (seed 11). k = sqrt(n) only for 3DRoad (k = 659): at 1M points k = 1000 needs 16 GB per round copied to the host, not run.

| dataset | n | k | start radius | rounds | active per round | hits per round (M) | predicted total ms |
|---|---|---|---|---|---|---|---|
| road | 434874 | 5 | 2.745e-05 | 13 | 434874 434739 433992 427768 398303 315364 177687 69333 13091 624 24 3 3 | 0.52 0.55 0.70 1.03 1.51 1.89 1.52 0.84 0.33 0.04 0.00 0.00 0.02 | 5.63 |
| road | 434874 | 659 | 2.745e-05 | 13 | 434874 434874 434874 434874 434874 434874 434874 434874 434183 398585 290407 90096 1166 | 0.52 0.56 0.71 1.14 2.17 4.83 12.43 35.10 102.72 213.26 269.61 207.15 4.49 | 538.20 |
| porto1m | 1000000 | 5 | 9e-06 | 20 | 1000000 910999 696419 354376 147109 66297 34137 18410 10011 5089 2271 597 8 1 1 1 1 1 1 1 | 2.57 3.72 4.93 3.36 1.51 0.74 0.42 0.24 0.13 0.06 0.03 0.01 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.99 | 11.78 |
| uni1m | 1000000 | 5 | 0.001834 | 5 | 1000000 1000000 999958 793013 75 | 1.05 1.39 4.12 19.79 0.01 | 16.60 |

What can fail: the rule is fitted on uniform 3D, k = 16 only (F158 held out total within 1.5 on two sizes only); 2D data with duplicates and heavy clusters, k = 5 and 659 not seen; managed memory effect not in the rule (original) at all.

## Arkade (sample01-knn, KN = 50, NORM = 2, 10,000 queries = a random subset (seed 7) of the data points appended to the file, device buffer + float square = fixed; original = pinned host buffer + double pow)

Paper parameters: 10,000 queries, k = 1, 50, 100 (main k not stated: CHOSEN 50), radius from the TrueKNN procedure (the code for it is not in the sample: CHOSEN, smallest r0 x 2^i such that 99 percent of the 10,000 queries have 50 neighbours within r in L2, r0 as in TrueKNN). Datasets we have: Gowalla (unique lat,lon of the check ins: 1,256,693 against 1,270,969 in the paper, outliers kept), Glove 3D (PCA of the 25 d Twitter GloVe, 1,193,514 points against 1,183,514: our inference of the vector set), Manuscript (2,155,617 against 2,145,617). Cali OSM, Kitti 4M, Randnet, Gbif: not available.

Calibration: 100,000 uniform 2D points and queries, r 0.005: cube candidates per query 10.93 (x 1.10 hardware), total 1202595.

| dataset | N | r | cand / query | A throughput ms | B latency ms | fixed device ms | original device ms | original pinned ms |
|---|---|---|---|---|---|---|---|---|
| gowalla | 1256693 | 0.7881 | 13658.1 | 223.736 | 440.064 | 39.606 | 440.064 | 441.384 |
| glove3d | 1193514 | 0.60924 | 44172.6 | 723.602 | 1233.133 | 110.982 | 1233.133 | 1234.453 |
| manuscript | 2155617 | 0.55414 | 162.6 | 2.664 | 1.672 | 0.240 | 2.664 | 3.984 |

What can fail: 10,000 queries are 313 warps on 142 SMs, a launch size never used in the calibration (100,000 queries, one wave); B is a bound that was never tested alone; 0.09 for float is from one uniform 2D set; real data have heavy candidate skew between lanes. Success rule for all three programs (fixed before): within a factor 2 of measured per row; reported as ratios, no refit.
