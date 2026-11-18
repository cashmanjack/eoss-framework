# RT against CLOVER, one table (2026-11-18, triton GPU 1, OptiX and plain CUDA only, no TTU code)

Predictions: predictions_phase11.md part B (commit 4e9f485), written before any run. Raw (private): notes/q40. Scripts: fair_prep.py (radii, in this directory), fair_table.py (this table); orchestrator and RTNN checker are in notes (private).
Methods: CLOVER hubs and warpwise (github.com/ampslab/clover-knn 03cfaa2, own timer); RTNN knn mode (K is a compile time macro, one build per K; radius = smallest radius at which 99 percent of 3,000 sampled queries have K other points inside, so about 1 percent of lists are short by construction); TrueKNN = the TrueKNN sample program of the Arkade/OWL tree, total time of its rounds as it prints it (one run per cell).
Datasets as F207 and F237 (the Arkade sets as the repo preprocessing gives them; RTNN reads the whole Arkade file, which carries 10,000 appended query rows, 0.8 percent more points). Every point is a query for hubs, RTNN and TrueKNN. Warpwise: 10,000 random queries scaled linearly to all points (checked on bunny in F227: within 6 percent).
Time (ms unless stated): device time to build the index and search, data already on the device, result left there. hubs = CLOVER's span (includes its index build); RTNN = query sort + geometry + search compute (median of 3; excludes context and pipeline creation and the result copy to the host); TrueKNN = whole program total of its rounds (includes host work between rounds). These are not identical scopes: the RTNN number leaves out result copy and set up, the TrueKNN number includes host work.
Exactness: share of 5,000 sampled queries whose list is exactly the K nearest by distance (kd tree, float32 points as double, tolerance 1e-6 relative, equal distance ties are not errors). hubs: query point counts as one of the K. RTNN knn mode never returns any point at distance 0 from the query, so a query with duplicate points gets a list without all of them: column "coincident points excluded" scores that convention (truth = K nearest points at distance above 0), column "coincident points count" scores the K nearest other points by index; the two agree where there are no duplicates (the share of sampled queries with a duplicate: 3DRoad 16 to 18 percent, Porto 25, Gowalla 17, Glove 3D 3, Manuscript 72, others 0). Warpwise: CLOVER's own check of 300 sampled queries. TrueKNN exactness is not reported in this public file.

| dataset | N | K | hubs ms | hubs exact | warpwise ms (scaled) | warpwise wrong in 300 | RTNN knn ms (sort+geom+search) | RTNN exact (coincident points excluded) | RTNN exact (coincident points count) | RTNN short lists | TrueKNN s (rounds) | fastest |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3DRoad | 434874 | 5 | 34.1 | 100.00 % | 9140 | 0 | 35.9 | 98.94 % | 83.18 % | 1.06 % | 0.089 (13) | hubs |
| 3DRoad | 434874 | 50 | 31.3 | 100.00 % | 10244 | 0 | 44.5 | 99.20 % | 81.80 % | 0.80 % | 0.427 (13) | hubs |
| 3DRoad | 434874 | 128 | 34.5 | 100.00 % | 15457 | 0 | 71.9 | 99.22 % | 81.32 % | 0.78 % | 2.993 (13) | hubs |
| Porto 1M | 1000000 | 5 | 93.0 | 100.00 % | 47941 | 0 | 85.5 | 98.72 % | 74.26 % | 1.28 % | 1.028 (20) | RTNN |
| Porto 1M | 1000000 | 50 | 95.9 | 100.00 % | 54400 | 0 | 827.2 | 98.64 % | 73.50 % | 1.36 % | 4.150 (20) | hubs |
| Porto 1M | 1000000 | 128 | 116.5 | 100.00 % | 81467 | 0 | 3957.0 | 99.02 % | 74.44 % | 0.98 % | 12.338 (20) | hubs |
| uniform 1M | 1000000 | 5 | 89.6 | 100.00 % | 49546 | 0 | 37.5 | 98.76 % | 98.76 % | 1.24 % | 0.064 (5) | RTNN |
| uniform 1M | 1000000 | 50 | 105.8 | 100.00 % | 55220 | 0 | 46.3 | 99.34 % | 99.34 % | 0.66 % | 1.716 (6) | RTNN |
| uniform 1M | 1000000 | 128 | 129.8 | 100.00 % | 83154 | 0 | 79.0 | 99.20 % | 99.20 % | 0.80 % | 6.783 (6) | RTNN |
| bunny | 362272 | 5 | 29.6 | 100.00 % | 6422 | 0 | 34.8 | 98.82 % | 98.82 % | 1.18 % | 0.122 (9) | hubs |
| bunny | 362272 | 50 | 30.8 | 100.00 % | 7393 | 0 | 39.3 | 99.32 % | 99.32 % | 0.68 % | 0.851 (9) | hubs |
| bunny | 362272 | 128 | 33.1 | 100.00 % | 11308 | 0 | 62.0 | 99.08 % | 99.08 % | 0.92 % | 3.530 (9) | hubs |
| dragon | 3609600 | 5 | 319.6 | 100.00 % | 660171 | 0 | 53.4 | 98.70 % | 98.70 % | 1.30 % | 0.699 (9) | RTNN |
| dragon | 3609600 | 50 | 399.4 | 100.00 % | 750762 | 0 | 75.7 | 98.84 % | 98.84 % | 1.16 % | 11.488 (11) | RTNN |
| dragon | 3609600 | 128 | 496.0 | 100.00 % | 1126316 | 0 | 181.8 | 98.74 % | 98.74 % | 1.26 % | 61.277 (11) | RTNN |
| Gowalla | 1280969 | 5 | 104.7 | 100.00 % | 81620 | 0 | 261.8 | 99.90 % | 82.76 % | 0.00 % | 0.326 (26) | hubs |
| Gowalla | 1280969 | 50 | 112.0 | 100.00 % | 92221 | 0 | 856.0 | 99.96 % | 82.58 % | 0.00 % | 4.086 (26) | hubs |
| Gowalla | 1280969 | 128 | 135.1 | 100.00 % | 136863 | 0 | 2606.6 | 99.90 % | 82.46 % | 0.00 % | 14.501 (26) | hubs |
| Glove 3D | 1193514 | 5 | 121.6 | 100.00 % | 71279 | 0 | 1809.0 | 100.00 % | 97.16 % | 0.00 % | 0.109 (11) | TrueKNN |
| Glove 3D | 1193514 | 50 | 151.2 | 100.00 % | 80049 | 0 | 4685.8 | 100.00 % | 96.76 % | 0.00 % | 1.224 (12) | hubs |
| Glove 3D | 1193514 | 128 | 206.0 | 100.00 % | 120636 | 0 | 7670.7 | 100.00 % | 96.86 % | 0.00 % | 8.899 (12) | hubs |
| Manuscript | 2155617 | 5 | 164.6 | 100.00 % | 236923 | 0 | 121.2 | 100.00 % | 28.10 % | 0.00 % | 0.137 (8) | RTNN |
| Manuscript | 2155617 | 50 | 214.7 | 100.00 % | 267209 | 0 | 354.4 | 100.00 % | 27.62 % | 0.00 % | 2.233 (9) | hubs |
| Manuscript | 2155617 | 128 | 265.6 | 100.00 % | 405864 | 0 | 1019.7 | 100.00 % | 27.76 % | 0.00 % | 21.744 (9) | hubs |

"fastest" is by time only among hubs, warpwise, RTNN and TrueKNN with the scope differences above.

## Predictions scored
B1 hubs fastest or within 2 times of the fastest of the three (hubs, RTNN, TrueKNN) on at least 18 of 24 cells: HELD (19 of 24; misses: uniform 1M K 5 and 50 (RTNN 2.3 to 2.4 times faster), dragon at all K (RTNN 2.7 to 6 times faster)).
B2 RTNN beats hubs only on dragon at K 5 and 50 and on at most two other cells at K 5 and 50, and hubs beats RTNN on every dataset at K 128 including dragon: MISSED. RTNN is faster than hubs on dragon at all three K (including K 128: 182 against 496 ms), on uniform 1M at all three K, on Porto 1M and Manuscript at K 5, and on none of the others; the K 5 and 50 cells where RTNN wins: uniform 1M (2), dragon (2), Porto K 5, Manuscript K 5 = 6 (I allowed 3). RTNN's search time grew about 3.4 times from K 5 to K 128 on dragon (53 to 182 ms), not enough to lose.
B3 RTNN exact on every cell, hubs 100 percent, warpwise 0 wrong: PARTLY. hubs 100.00 percent and warpwise 0 wrong in 300 in every cell (held). RTNN exact only when coincident points are excluded: 98.6 to 100 percent of sampled queries with the 0.7 to 1.4 percent short lists of the 99 percent radius on the non Arkade sets, and 99.9 to 100 percent on the Arkade sets; counting coincident points as neighbours it is 28 to 83 percent on the sets with duplicates. My first scoring (convention by index only) gave 0.00 on the Arkade sets, a checker fault (truth tree built from the first N points while RTNN also reads the 10,000 appended rows), found and fixed before the numbers above.
B4 TrueKNN does not finish within 300 s on at least 4 cells: MISSED, every cell finished (slowest dragon K 128, 61 s).
B5 RTNN fails at K 128 on some dataset: MISSED, it ran everywhere.

## Reading
- A pruned linear scan on the GPU (hubs) is the best or within 2 times of the best method in 19 of 24 cells, 100 percent exact, and costs almost nothing more at K 128 (1.0 to 1.5 times K 5). RT wins where the data are a dense 3D surface or uniform and large (dragon, uniform 1M): RTNN 1.6 to 6 times faster at all K, by the scope above.
- Where RT loses: clustered or duplicate heavy data at larger K (Porto K 50 and 128: RTNN 8.6 and 34 times slower, Gowalla 7.6 and 19, Glove 3D 31 and 37): the radius needed for K points is large there, so the ball holds many points. TrueKNN is slower than hubs by 8 to 124 times at K 50 and 128 everywhere, and wins only on uniform 1M K 5 (0.72 of hubs), Glove 3D K 5 (0.89) and Manuscript K 5 (0.83).
- Warpwise (the plain linear scan) is 200 to 6,000 times slower than the best method everywhere.
- Caveats: one GPU; one run for TrueKNN; the three time scopes differ; RTNN radius chosen to give 99 percent of queries K points (a smaller radius would be faster but leave more lists short); hubs cannot answer a query subset (F237) so its time is for all points.
