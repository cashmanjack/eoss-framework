# Arkade predictions (written before any measurement of the search launch), 2026-11-15, RTX 4070 Ti (io GPU 0)

Data: synthetic uniform 2D points (the program reads x and y only, z = 0), 100,000 queries (the lines after the data points), one round, radius estimate given. Real Arkade datasets are not available here. Build: KN 16, NORM 2.
Replay: model/replay/replay_aabb.cpp over the OptiX GAS of the cube boxes (dumped after the build, no search launch) and the real rays. Hardware candidates = 1.006 x primitives in entered leaf slots (Finding 125).
Time: fixed part of Finding 127 (c0 110, cq 4.9 cycles per candidate per SM; per set fits c0 93 to 120, cq 4.2 to 5.9), miss term omitted; this program has no heap in global memory but a 16 entry scan per candidate in registers or local memory (any callback cost not in this prediction: see Finding 131).

| config | data points | radius | queries | nodes / query | leaf slots / query | primitives in slots / query | hardware candidates / query | lane efficiency | predicted launch ms (c0 110, cq 4.9) |
|---|---|---|---|---|---|---|---|---|---|
| a1e5_r5 | 100000 | 0.005 | 100000 | 7.62 | 8.45 | 10.84 | 10.91 | 0.581 | 0.112 |
| a1e5_r11 | 100000 | 0.0112 | 100000 | 12.26 | 34.82 | 53.69 | 54.02 | 0.759 | 0.252 |
| a1e6_r16 | 1000000 | 0.0016 | 100000 | 8.70 | 8.55 | 10.56 | 10.62 | 0.507 | 0.111 |

What can fail: candidates per query beyond 1 x 1.006; lane efficiency beyond 0.05; time beyond a factor 2 (the callback and miss terms are not in the prediction).
