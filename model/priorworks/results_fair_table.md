# RT against CLOVER, one table, one timing scope (2026-11-18, scope matched in Phase 12 (predictions_phase12.md part A, commit 9fcc6c3); triton GPU 1, OptiX and plain CUDA only, no TTU code)

Predictions: predictions_phase11.md part B (commit 4e9f485), written before any run. Raw (private): notes/q40. Scripts: fair_prep.py (radii, in this directory), fair_table.py (this table); orchestrator and RTNN checker are in notes (private).
Methods: CLOVER hubs and warpwise (github.com/ampslab/clover-knn 03cfaa2, own timer); RTNN knn mode (K is a compile time macro, one build per K; radius = smallest radius at which 99 percent of 3,000 sampled queries have K other points inside, so about 1 percent of lists are short by construction); TrueKNN = the TrueKNN sample program of the Arkade/OWL tree, total time of its rounds as it prints it (one run per cell).
Datasets as F207 and F237 (the Arkade sets as the repo preprocessing gives them; RTNN reads the whole Arkade file, which carries 10,000 appended query rows, 0.8 percent more points). Every point is a query for hubs, RTNN and TrueKNN. Warpwise: 10,000 random queries scaled linearly to all points (checked on bunny in F227: within 6 percent).
One timing scope for every method (device scope): data already on the device, index or BVH build plus search, no host set up, no result copy. Time in ms. Host time is a separate column group.
Device scope: hubs and warpwise = CLOVER's own span (device synchronized, includes hubs' index build; read from the source: the span starts after its point and query copies and the result stays on the device; a rerun of all 24 hubs cells with the same build plus a host timer agrees with the F245 numbers, median ratio 1.00, range 0.95 to 1.11; warpwise 0.97 to 1.00); warpwise = 10,000 random queries scaled linearly to all points (checked on bunny in F227: within 6 percent). RTNN knn mode = query sort + create and upload geometry + search compute (its own timers, median of 3, as F245). TrueKNN = BVH build + every round's launch + every round's refit, each device synchronized, from the original sample with timers added (a copy; the only change is the timers and one output line; rounds and radii are the same as the unmodified sample's; 1 run per cell, so run to run spread, 1 to 9 percent against F245's total, is not known beyond that).
Host columns: hubs and warpwise = host wall time of CLOVER's whole call (allocation, host to device copies, search, result copy back to the host; median of reps, the first rep includes CUDA context start up; warpwise for its 10,000 queries, not scaled); RTNN = its printed total search time minus the three device timers (almost all of it the result copy to the host, K x N x 4 bytes); TrueKNN = the loop total the sample prints minus the device scope, which is the host scan of the managed frame buffer after every round (measured separately: it equals the host part to within 1 percent) plus parameter updates. "TrueKNN printed total" is the old scope (build + whole round loop) of the same run.
Datasets as F207 and F237 (the Arkade sets as the repo preprocessing gives them; RTNN reads the whole Arkade file, which carries 10,000 appended query rows, 0.8 percent more points). Every point is a query for hubs, RTNN and TrueKNN. RTNN radius = smallest radius at which 99 percent of 3,000 sampled queries have K other points inside, so about 1 percent of lists are short by construction. RTNN knn mode excludes coincident points (exact only under that convention). hubs cannot answer a query subset (F237). TrueKNN exactness is not reported in this public file.
Exactness: share of 5,000 sampled queries whose list is exactly the K nearest by distance (kd tree, float32 points as double, tolerance 1e-6 relative, equal distance ties are not errors). hubs: query point counts as one of the K. RTNN: column "coincident points excluded" scores that convention (truth = K nearest points at distance above 0), "coincident points count" scores the K nearest other points by index; they agree where there are no duplicates (share of sampled queries with a duplicate: 3DRoad 16 to 18 percent, Porto 25, Gowalla 17, Glove 3D 3, Manuscript 72, others 0). Warpwise: CLOVER's own check of 300 sampled queries.

| dataset | N | K | hubs | warpwise (scaled) | RTNN knn | TrueKNN (device) | fastest | hubs host | warpwise host (10,000 q) | RTNN host | TrueKNN host | TrueKNN printed total (old scope) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3DRoad | 434874 | 5 | 34.1 | 9140 | 35.9 | 54.4 | hubs | 43.8 | 354.6 | 3.5 | 33.4 | 87.8 |
| 3DRoad | 434874 | 50 | 31.3 | 10244 | 44.5 | 218.9 | hubs | 116.1 | 371.5 | 33.9 | 202.9 | 421.9 |
| 3DRoad | 434874 | 128 | 34.5 | 15457 | 71.9 | 2477.1 | hubs | 242.9 | 511.4 | 85.6 | 508.5 | 2985.6 |
| Porto 1M | 1000000 | 5 | 93.0 | 47941 | 85.5 | 878.4 | RTNN | 111.8 | 626.0 | 8.0 | 142.2 | 1020.6 |
| Porto 1M | 1000000 | 50 | 95.9 | 54400 | 827.2 | 3199.2 | hubs | 293.0 | 677.8 | 77.6 | 946.3 | 4145.5 |
| Porto 1M | 1000000 | 128 | 116.5 | 81467 | 3957.0 | 10137.2 | hubs | 580.2 | 968.4 | 199.0 | 2110.9 | 12248.1 |
| uniform 1M | 1000000 | 5 | 89.6 | 49546 | 37.5 | 34.6 | TrueKNN | 109.0 | 622.9 | 7.9 | 29.3 | 63.9 |
| uniform 1M | 1000000 | 50 | 105.8 | 55220 | 46.3 | 1448.4 | RTNN | 291.6 | 683.5 | 77.2 | 264.1 | 1712.5 |
| uniform 1M | 1000000 | 128 | 129.8 | 83154 | 79.0 | 6108.5 | RTNN | 606.5 | 966.9 | 198.3 | 662.0 | 6770.6 |
| bunny | 362272 | 5 | 29.6 | 6422 | 34.8 | 75.8 | hubs | 38.4 | 322.2 | 3.2 | 45.3 | 121.2 |
| bunny | 362272 | 50 | 30.8 | 7393 | 39.3 | 514.3 | hubs | 100.1 | 347.4 | 29.0 | 337.2 | 851.5 |
| bunny | 362272 | 128 | 33.1 | 11308 | 62.0 | 2709.4 | hubs | 208.6 | 451.7 | 71.7 | 814.3 | 3523.7 |
| dragon | 3609600 | 5 | 319.6 | 660171 | 53.4 | 305.7 | RTNN | 390.5 | 1967.0 | 28.3 | 396.9 | 702.6 |
| dragon | 3609600 | 50 | 399.4 | 750762 | 75.7 | 6483.8 | RTNN | 1038.9 | 2207.6 | 276.8 | 5015.4 | 11499.2 |
| dragon | 3609600 | 128 | 496.0 | 1126316 | 181.8 | 44667.7 | RTNN | 2065.4 | 3245.6 | 719.7 | 11095.3 | 55763.0 |
| Gowalla | 1280969 | 5 | 104.7 | 81620 | 261.8 | 253.8 | hubs | 134.5 | 778.9 | 11.8 | 61.8 | 315.6 |
| Gowalla | 1280969 | 50 | 112.0 | 92221 | 856.0 | 3695.8 | hubs | 360.7 | 865.8 | 111.6 | 367.5 | 4063.3 |
| Gowalla | 1280969 | 128 | 135.1 | 136863 | 2606.6 | 13577.3 | hubs | 743.5 | 1228.9 | 280.1 | 913.6 | 14490.9 |
| Glove 3D | 1193514 | 5 | 121.6 | 71279 | 1809.0 | 72.8 | TrueKNN | 145.2 | 730.5 | 11.0 | 36.1 | 108.9 |
| Glove 3D | 1193514 | 50 | 151.2 | 80049 | 4685.8 | 892.7 | hubs | 383.5 | 818.7 | 104.4 | 325.8 | 1218.5 |
| Glove 3D | 1193514 | 128 | 206.0 | 120636 | 7670.7 | 8062.9 | hubs | 765.6 | 1146.8 | 259.7 | 822.0 | 8884.9 |
| Manuscript | 2155617 | 5 | 164.6 | 236923 | 121.2 | 74.9 | TrueKNN | 207.9 | 1247.7 | 19.7 | 62.1 | 137.0 |
| Manuscript | 2155617 | 50 | 214.7 | 267209 | 354.4 | 1730.8 | hubs | 600.9 | 1387.0 | 190.4 | 564.0 | 2294.8 |
| Manuscript | 2155617 | 128 | 265.6 | 405864 | 1019.7 | 20315.2 | hubs | 1277.9 | 2026.0 | 457.4 | 1422.4 | 21737.6 |

Exactness (unchanged from F245):

| dataset | K | hubs exact | warpwise wrong in 300 | RTNN exact (coincident points excluded) | RTNN exact (coincident points count) | RTNN short lists |
|---|---|---|---|---|---|---|
| 3DRoad | 5 | 100.00 % | 0 | 98.94 % | 83.18 % | 1.06 % |
| 3DRoad | 50 | 100.00 % | 0 | 99.20 % | 81.80 % | 0.80 % |
| 3DRoad | 128 | 100.00 % | 0 | 99.22 % | 81.32 % | 0.78 % |
| Porto 1M | 5 | 100.00 % | 0 | 98.72 % | 74.26 % | 1.28 % |
| Porto 1M | 50 | 100.00 % | 0 | 98.64 % | 73.50 % | 1.36 % |
| Porto 1M | 128 | 100.00 % | 0 | 99.02 % | 74.44 % | 0.98 % |
| uniform 1M | 5 | 100.00 % | 0 | 98.76 % | 98.76 % | 1.24 % |
| uniform 1M | 50 | 100.00 % | 0 | 99.34 % | 99.34 % | 0.66 % |
| uniform 1M | 128 | 100.00 % | 0 | 99.20 % | 99.20 % | 0.80 % |
| bunny | 5 | 100.00 % | 0 | 98.82 % | 98.82 % | 1.18 % |
| bunny | 50 | 100.00 % | 0 | 99.32 % | 99.32 % | 0.68 % |
| bunny | 128 | 100.00 % | 0 | 99.08 % | 99.08 % | 0.92 % |
| dragon | 5 | 100.00 % | 0 | 98.70 % | 98.70 % | 1.30 % |
| dragon | 50 | 100.00 % | 0 | 98.84 % | 98.84 % | 1.16 % |
| dragon | 128 | 100.00 % | 0 | 98.74 % | 98.74 % | 1.26 % |
| Gowalla | 5 | 100.00 % | 0 | 99.90 % | 82.76 % | 0.00 % |
| Gowalla | 50 | 100.00 % | 0 | 99.96 % | 82.58 % | 0.00 % |
| Gowalla | 128 | 100.00 % | 0 | 99.90 % | 82.46 % | 0.00 % |
| Glove 3D | 5 | 100.00 % | 0 | 100.00 % | 97.16 % | 0.00 % |
| Glove 3D | 50 | 100.00 % | 0 | 100.00 % | 96.76 % | 0.00 % |
| Glove 3D | 128 | 100.00 % | 0 | 100.00 % | 96.86 % | 0.00 % |
| Manuscript | 5 | 100.00 % | 0 | 100.00 % | 28.10 % | 0.00 % |
| Manuscript | 50 | 100.00 % | 0 | 100.00 % | 27.62 % | 0.00 % |
| Manuscript | 128 | 100.00 % | 0 | 100.00 % | 27.76 % | 0.00 % |

Fastest label changed by the scope (old TrueKNN total -> device scope): 2 of 24 cells: uniform 1M K 5: RTNN to TrueKNN; Manuscript K 5: RTNN to TrueKNN

## Predictions scored
B1 hubs fastest or within 2 times of the fastest of the three (hubs, RTNN, TrueKNN) on at least 18 of 24 cells: HELD (19 of 24 as first scored with the TrueKNN total; 18 of 24 with the device scope; misses: uniform 1M K 5 and 50 (RTNN 2.3 to 2.4 times faster), dragon at all K (RTNN 2.7 to 6 times faster)).
B2 RTNN beats hubs only on dragon at K 5 and 50 and on at most two other cells at K 5 and 50, and hubs beats RTNN on every dataset at K 128 including dragon: MISSED. RTNN is faster than hubs on dragon at all three K (including K 128: 182 against 496 ms), on uniform 1M at all three K, on Porto 1M and Manuscript at K 5, and on none of the others; the K 5 and 50 cells where RTNN wins: uniform 1M (2), dragon (2), Porto K 5, Manuscript K 5 = 6 (I allowed 3). RTNN's search time grew about 3.4 times from K 5 to K 128 on dragon (53 to 182 ms), not enough to lose.
B3 RTNN exact on every cell, hubs 100 percent, warpwise 0 wrong: PARTLY. hubs 100.00 percent and warpwise 0 wrong in 300 in every cell (held). RTNN exact only when coincident points are excluded: 98.6 to 100 percent of sampled queries with the 0.7 to 1.4 percent short lists of the 99 percent radius on the non Arkade sets, and 99.9 to 100 percent on the Arkade sets; counting coincident points as neighbours it is 28 to 83 percent on the sets with duplicates. My first scoring (convention by index only) gave 0.00 on the Arkade sets, a checker fault (truth tree built from the first N points while RTNN also reads the 10,000 appended rows), found and fixed before the numbers above.
B4 TrueKNN does not finish within 300 s on at least 4 cells: MISSED, every cell finished (slowest dragon K 128, 61 s).
B5 RTNN fails at K 128 on some dataset: MISSED, it ran everywhere.

## Reading (scope matched)
- A pruned linear scan on the GPU (hubs) is the fastest or within 2 times of the fastest in 18 of 24 cells (F245 had 19 with the TrueKNN total), 100 percent exact, and costs almost nothing more at K 128 (1.0 to 1.5 times K 5). Misses: uniform 1M K 5 and 50 (2.6, 2.3 times), dragon at all K (6.0, 5.3, 2.7), Manuscript K 5 (2.2).
- The scope change moves the TrueKNN column only: its device scope is 6 to 57 percent below its printed total (the host scan of the managed buffer, largest at K 5 and 50 on the large sets: dragon K 5 and 50 56 and 44 percent). Fastest label changes in 2 of 24 cells: uniform 1M K 5 and Manuscript K 5 go from RTNN to TrueKNN. TrueKNN (device scope) is faster than hubs on four K 5 cells (uniform 1M 0.39 of hubs, Manuscript 0.46, Glove 3D 0.60, dragon 0.96) and 5.9 to 100 times slower at K 50 and 128 (F245: 8 to 124).
- RTNN wins where the data are a dense 3D surface or uniform and large (dragon, uniform 1M: 1.6 to 6 times faster than hubs at all K), unchanged. Where RT loses: clustered or duplicate heavy data at larger K (Porto K 50 and 128: RTNN 8.6 and 34 times slower, Gowalla 7.6 and 19, Glove 3D 31 and 37).
- Host time is a different story: the result copy makes the whole call of hubs 1.2 to 7.0 times its device time (K 128 dragon 2,065 ms host against 496 ms device) and adds to RTNN 3 to 720 ms (0.006 to 4.0 times its device time; larger than the device time on six cells only: 3DRoad K 128, uniform 1M K 50 and 128, bunny K 128, dragon K 50 and 128). The host columns are not like for like: hubs's host wall includes allocating and copying the points in and the result out, RTNN's host part (printed total minus device timers) does not include its set up (context, pipeline) or its point upload. Adding each method's host part to its device time (hubs host wall against RTNN device + host) would flip 7 cells to RTNN (3DRoad and bunny at all K, Manuscript K 50) but that comparison favors RTNN by construction, so it is not used for any conclusion.
- Warpwise (the plain linear scan) is 200 to 6,000 times slower than the best method everywhere (device scope).
- Caveats: one GPU; one run for TrueKNN; RTNN radius chosen to give 99 percent of queries K points; hubs answers all points only.
