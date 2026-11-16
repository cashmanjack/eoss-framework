# MVA with traversal unit and hand off stations: predictions (written before any comparison; form in model/q18/FORM_mva2.md, committed first)
Constants: X_tu = 0.14 lanes per cycle per TPC (F105), hand off capacity 1 / 0.00774 = 129 cycles per warp call per SM (highest call rate of the non target launches, an upper bound, not a saturation test), clock 2.5 GHz, issue multiplier 1.
Inputs per launch: calls = warp calls of the callback program, e = lane efficiency, W = warps in flight per SM (ncu), f = floor cycles per call (F131), L_unl and C = unloaded callback latency and trace window cycles per call (one warp per SM launch). Output: loaded latency per call L (cycles) and launch time T (ms).

| launch | calls | e | W | f | L unloaded | C (trace per call) | D_tu | issue only: L | issue only: T ms | full network: L | full network: T ms | utilisation issue / TU / hand off |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| knn reg round 1 | 15427 | 0.513 | 14.2 | 498 | 4091 | 9868 | 117 | 4892 | 0.077 | 4880 | 0.078 | 0.66 / 0.31 / 0.17 |
| knn reg round 2 | 38366 | 0.427 | 14.2 | 351 | 3214 | 6651 | 98 | 3835 | 0.131 | 3818 | 0.132 | 0.68 / 0.38 / 0.25 |
| knn reg round 3 | 157598 | 0.554 | 14.7 | 373 | 2957 | 5227 | 127 | 4286 | 0.468 | 4195 | 0.475 | 0.82 / 0.56 / 0.29 |
| knn reg round 4 | 893589 | 0.745 | 15.1 | 310 | 2570 | 4316 | 170 | 3765 | 2.175 | 3399 | 2.373 | 0.78 / 0.85 / 0.32 |
| knn orig round 1 | 15427 | 0.513 | 14.7 | 2358 | 12531 | 18043 | 117 | 31023 | 0.255 | 31008 | 0.255 | 0.95 / 0.09 / 0.05 |
| knn orig round 2 | 38366 | 0.427 | 14.6 | 1566 | 9362 | 12738 | 98 | 20809 | 0.424 | 20791 | 0.424 | 0.94 / 0.12 / 0.08 |
| knn orig round 3 | 157598 | 0.554 | 15.0 | 1450 | 8372 | 10705 | 127 | 20422 | 1.593 | 20393 | 1.593 | 0.96 / 0.17 / 0.09 |
| knn orig round 4 | 893589 | 0.745 | 15.3 | 900 | 5801 | 7569 | 170 | 12743 | 5.658 | 12656 | 5.666 | 0.95 / 0.36 / 0.14 |
| arkade device buffer, io (60 SMs) | 105089 | 0.581 | 15.3 | 5234 | 22610 | 28764 | 133 | 75983 | 3.764 | 75974 | 3.764 | 0.97 / 0.05 / 0.02 |
| arkade device buffer, triton (142 SMs) | 105089 | 0.581 | 15.3 | 5234 | 22610 | 28764 | 133 | 75983 | 1.590 | 75974 | 1.590 | 0.97 / 0.05 / 0.02 |
