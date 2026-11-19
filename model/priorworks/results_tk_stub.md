# TrueKNN at K 50 and 128: replay bound with a free callback (2026-11-19, triton GPU 1; file tkstub_20261119T102257.jsonl)

Original = Vani's program, recorder build (adds one byte store per query per round in raygen), device scope = build + launches + refits (F247). Stub = intersection program returns at once, same radii, same refits, masks of the original replayed (raygen reads the mask). Stub median of 3. Hubs and RTNN knn from model/priorworks/results_fair_table.md (F245); F245 TrueKNN device column for reference. ms.

| dataset | K | rounds | rounds, radii and active counts equal in all stub runs | original (this session) | F245 TrueKNN | stub median (min to max) | stub / original | hubs | RTNN knn | stub < hubs | stub < RTNN | stub build + refit share | callback cost a fix may leave for TrueKNN to tie hubs (share of the current callback cost) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bunny | 50 | 9 | yes | 519.4 | 514.3 | 44.6 (44.4 to 47.3) | 0.086 | 30.8 | 39.3 | no | no | 0.56 | none (the stub alone is slower) |
| 3DRoad | 50 | 13 | yes | 224.2 | 218.9 | 43.5 (43.0 to 45.5) | 0.194 | 31.3 | 44.5 | no | yes | 0.76 | none (the stub alone is slower) |
| uniform 1M | 50 | 6 | yes | 1457.4 | 1448.4 | 46.3 (43.5 to 46.4) | 0.032 | 105.8 | 46.3 | yes | yes | 0.74 | 4.2 percent |
| Porto 1M | 50 | 20 | yes | 3180.1 | 3199.2 | 427.4 (425.8 to 429.6) | 0.134 | 95.9 | 827.2 | no | yes | 0.18 | none (the stub alone is slower) |
| Gowalla | 50 | 26 | yes | 3666.6 | 3695.8 | 544.9 (542.2 to 545.2) | 0.149 | 112.0 | 856.0 | no | yes | 0.22 | none (the stub alone is slower) |
| Glove 3D | 50 | 12 | yes | 889.0 | 892.7 | 143.1 (143.0 to 143.2) | 0.161 | 151.2 | 4685.8 | yes | yes | 0.42 | 1.1 percent |
| Manuscript | 50 | 9 | yes | 1670.5 | 1730.8 | 103.6 (99.8 to 104.1) | 0.062 | 214.7 | 354.4 | yes | yes | 0.73 | 7.1 percent |
| dragon | 50 | 11 | yes | 6538.2 | 6483.8 | 280.6 (278.4 to 282.7) | 0.043 | 399.4 | 75.7 | yes | no | 0.51 | 1.9 percent |
| bunny | 128 | 9 | yes | 2704.3 | 2709.4 | 47.7 (47.2 to 50.2) | 0.018 | 33.1 | 62.0 | no | yes | 0.54 | none (the stub alone is slower) |
| 3DRoad | 128 | 13 | yes | 2497.7 | 2477.1 | 48.1 (47.6 to 50.6) | 0.019 | 34.5 | 71.9 | no | yes | 0.70 | none (the stub alone is slower) |
| uniform 1M | 128 | 6 | yes | 6110.2 | 6108.5 | 50.5 (48.8 to 51.2) | 0.008 | 129.8 | 79.0 | yes | yes | 0.68 | 1.3 percent |
| Porto 1M | 128 | 20 | yes | 10241.6 | 10137.2 | 458.2 (455.3 to 459.0) | 0.045 | 116.5 | 3957.0 | no | yes | 0.17 | none (the stub alone is slower) |
| Gowalla | 128 | 26 | yes | 13630.6 | 13577.3 | 696.4 (694.4 to 707.2) | 0.051 | 135.1 | 2606.6 | no | yes | 0.18 | none (the stub alone is slower) |
| Glove 3D | 128 | 12 | yes | 8054.8 | 8062.9 | 189.8 (189.1 to 198.8) | 0.024 | 206.0 | 7670.7 | yes | yes | 0.32 | 0.2 percent |
| Manuscript | 128 | 9 | yes | 20311.3 | 20315.2 | 124.4 (124.4 to 125.5) | 0.006 | 265.6 | 1019.7 | yes | yes | 0.59 | 0.7 percent |
| dragon | 128 | 11 | yes | 44599.3 | 44667.7 | 338.1 (338.0 to 339.8) | 0.008 | 496.0 | 181.8 | yes | no | 0.42 | 0.4 percent |

K 50: cells run 8; stub below hubs in 4; stub below RTNN in 6; stub below the original in 8; replay exact in 8; median stub / original 0.110 (range 0.032 to 0.194).
K 128: cells run 8; stub below hubs in 4; stub below RTNN in 7; stub below the original in 8; replay exact in 8; median stub / original 0.018 (range 0.006 to 0.051).
All: stub above hubs in 8 of 16 cells; stub above RTNN in 3 of 16.
