# RTNN predictions on the 4070 Ti (io, GPU 0), written 2026-11-15 BEFORE any run on these point clouds
The only RTNN run so far is the 8 point sample file (smoke test of the build: CUDA 12.8, OptiX 7.1 headers, driver 580.178.04, patched copy in /dev/shm/rt, source in ~/rtnn untouched). Nothing was measured on the point clouds below.

Data: uniform random points in [0,100]^3 (seeds in model/predict_rtnn.py gen): u1e5 (100,000 points) and u1e6 (1,000,000). Queries = points. Radius search, K = 50, flags `-p 0 -s 0` (no query partitioning, no GAS sort), default grid Morton sort. Measured quantity: "time search compute" (sum over launches) printed by RTNN.
Model: model/predict_rtnn.py (candidates per query exact on the CPU; time = N (C_eff + 1) c / (SMeq f), c in 7.4 to 20 cycles, SMeq 41.4 to 60, f 2.4 to 2.6 GHz; Findings 24, 31, 49, 81, 90, 105). A prediction counts as right if the measured time is inside the range; the point is the geometric mean.
Known holes: fixed launch cost is not in the model (matters for the 0.04 to 1 ms cases); the ray ends after K sphere hits (C_eff formula assumes random order of candidates); c is a bracket, not a measurement of the IS program cost (F49 measured the round trip latency, not throughput).

```
u1e5  N  100000 r 2.0 | C_cube    7.21  sphere hits    4.29  C_eff    7.21 | predicted search compute ms: 0.039 .. 0.165  point 0.080
u1e5  N  100000 r 4.0 | C_cube   49.18  sphere hits   26.62  C_eff   49.18 | predicted search compute ms: 0.238 .. 1.010  point 0.490
u1e5  N  100000 r 6.0 | C_cube  158.64  sphere hits   85.48  C_eff   91.98 | predicted search compute ms: 0.441 .. 1.872  point 0.909
u1e6  N 1000000 r 1.0 | C_cube    8.87  sphere hits    5.14  C_eff    8.87 | predicted search compute ms: 0.468 .. 1.987  point 0.965
u1e6  N 1000000 r 2.0 | C_cube   63.09  sphere hits   33.75  C_eff   63.07 | predicted search compute ms: 3.039 .. 12.897  point 6.261
u1e6  N 1000000 r 3.0 | C_cube  207.37  sphere hits  110.28  C_eff   93.77 | predicted search compute ms: 4.495 .. 19.076  point 9.260

ratios (point predictions): u1e5 r4 / r2: 6.11, u1e5 r6 / r4: 1.85, u1e6 r2 / r1: 6.49, u1e6 r3 / r2: 1.48
unsorted queries and points (-qs 0 -ps 0): u1e6 at least 1.5x slower than sorted, u1e5 at most 1.2x (BVH of 1e6 AABBs is about 70 to 80 MB: beyond the 48 MB L2; 1e5 is 7 MB)
```
