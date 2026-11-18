# RTNN's own query partitioning (-p 1) in exact mode (-a 0) on the cells where RT lost to hubs (2026-11-18, triton GPU 1, OptiX only, no TTU code)

Predictions: predictions_phase12.md part B (commit 9fcc6c3), written before the run. Same builds, data, radius (F245 r99, the largest search radius) and flags as F245 (-s 0, -m 0) except -p 1 -a 0; RTNN's default automatic batching. Device scope as F247: query sort and partition + geometry + search compute, SUMMED over batches (RTNN prints them per batch; my first parser took the last batch only, found by the batch count and the total search time not adding up, and replaced by notes/q42/part_parse.py before anything was used). Median of 3 runs. Exactness: 5,000 sampled queries per batch against a kd tree built from the batch's own points, weighted by batch size, coincident points excluded (as F245). hubs = F245 (rerun F247 agrees within 5 to 11 percent).

| cell | batches (queries in the first) | unpartitioned RTNN ms (F245) | partitioned ms (sort+partition / geometry / search) | speedup | hubs ms | partitioned / hubs | exact, unpartitioned | exact, partitioned | short lists, partitioned |
|---|---|---|---|---|---|---|---|---|---|
| Porto K 50 | 1 (1,000,000) | 827.2 | 831.9 (28.6 / 3.6 / 799.4) | 0.99 | 95.9 | 8.7 | 98.64 % | 98.78 % | 1.22 % |
| Porto K 128 | 2 (947,936) | 3,957.0 | 2,237.2 (28.4 / 7.5 / 2,201.3) | 1.77 | 116.5 | 19.2 | 99.02 % | 98.93 % | 1.07 % |
| Gowalla K 50 | 5 (1,287,049) | 856.0 | 464.1 (28.8 / 24.1 / 411.2) | 1.84 | 112.0 | 4.1 | 99.96 % | 99.89 % | 0.007 % |
| Gowalla K 128 | 8 (1,277,052) | 2,606.6 | 1,287.4 (28.9 / 38.8 / 1,219.7) | 2.02 | 135.1 | 9.5 | 99.90 % | 99.95 % | 0.009 % |
| Glove 3D K 50 | 68 (1,189,654) | 4,685.8 | 780.5 (365.8 / 305.0 / 103.2) | 6.00 | 151.2 | 5.2 | 100.00 % | 99.996 % | 0.004 % |
| Glove 3D K 128 | 94 (1,154,893) | 7,670.7 | 1,540.0 (710.6 / 421.2 / 408.1) | 4.98 | 206.0 | 7.5 | 100.00 % | 99.995 % | 0.005 % |

Result: partitioning does not flip any of the six cells (partitioned RTNN is still 4.1 to 19 times slower than hubs). It makes RTNN 1.8 to 6.0 times faster on five cells and does nothing on Porto K 50 (one batch).
- Almost all queries stay in the first batch (the one with the full radius or the largest one): 95 to 99.7 percent (Porto K 50: all). The partition moves only the sparse tail to other radii.
- On Glove 3D the partition itself (sorting and making 68 to 94 geometry builds) costs 86 percent of the device time at K 50 and 73 percent at K 128; on Gowalla and Porto the search is 89 to 98 percent.
- Exactness: the partitioned lists are exact under the same convention at the same rate within 0.15 percentage points of the unpartitioned runs (differences have both signs, 5,000 queries per batch). The short lists of the 99 percent radius remain on Porto (1.1 to 1.2 percent); on Gowalla and Glove 3D they were already about zero.
- Why Porto K 50 has one batch: RTNN's grid cell size is radius over a ratio; the automatic ratio fell to 1.45 (cell size 0.003642 for radius 0.005271) because the data's bounding box is 6 by 8 by 1 units against a radius of 0.005, which leaves one distance level to partition. Forcing the ratio to 8 (-ac 0 -cr 8, one run): the grid has 2,147,483,648 cells (2 to the 31) and the program stops with an illegal memory access (cudaErrorIllegalAddress), consistent with a 32 bit cell index overflow (cause not confirmed in the source). So the partition cannot be refined on this data.
- Caveats: one GPU; default automatic batching only for the table (a run of Porto K 50 with batching forced to the maximum (-ab 0 -nb -1) also had one available batch); radius r99; the exactness check uses sampled queries; the first run of Porto K 50 with default settings (before the parser fix) gave the same one batch.

Written predictions: completes on all six HELD; exactness at least as high on every cell MISSED (four cells lower by 0.004 to 0.09 points, within sampling noise) and short lists under 0.1 percent HELD for Gowalla and Glove 3D only (Porto 1.1 to 1.2 percent); time lower by 1.5 to 10 times on every cell MISSED (five of six; Porto K 50 0.99); flips at most one of the six HELD (none flipped).
