# Arkade (sample01-knn) on the 4070 Ti against the committed predictions (model/predictions_arkade_4070ti.md, commit 15d07d5)
Synthetic uniform 2D data (not Arkade's datasets, not available), 100,000 queries, KN 16, L2 build (NORM 2). Launch time by CUDA events (second launch of the process, 5 runs), recorder candidates and lane efficiency (one run).
| config | candidates / query predicted | measured | lane eff predicted | measured | launch ms predicted | measured (min to max of 5) |
|---|---|---|---|---|---|---|
| 100,000 points r 0.005 | 10.91 | 10.90 | 0.581 | 0.584 | 0.112 | 23.0 to 27.0 |
| 100,000 points r 0.0112 | 54.02 | 53.86 | 0.759 | 0.760 | 0.252 | 27.5 to 32.1 |
| 1,000,000 points r 0.0016 | 10.62 | 11.24 | 0.507 | 0.591 | 0.111 | 21.1 to 21.6 |
Candidates and lane efficiency right except the 1M point case (5.5 percent, 0.08). Time wrong by 100 to 200 times. ncu (100,000 points, r 0.005): the launch is 26.7 ms, 85 percent of stall samples are long scoreboard (loads), issue slots 0.65 percent active, warps 32 percent of peak, FP64 pipe 13 percent busy. The program's L2 build calls pow on doubles (87 of 170 warp instructions of the intersection function are double precision, 19 SM cycles each, measured). The callback term (with the double precision cost added) gives 3.7 ms: a throughput floor, 7 times below the measurement because the callback is latency bound at this occupancy.
