# Winner prediction with model v1 unchanged (2026-11-18, triton, CPU only, NOT BLIND: F245 is known)

Predictions: predictions_phase12.md part D (commit 9fcc6c3). Model: model_v1_rtnn.py (v1 RTNN form, callback cost = uniform reference, K generalized from 50), replay counts from a kd tree on the F245 points and r99 radii. Units ms. Predicted = search launch only; measured = RTNN search compute (F245 median of 3).

| dataset | K | mean q | max q | throughput | chain | predicted launch | measured search | measured / predicted | RTNN sort+geom (not predicted) | RTNN total | hubs | call A | call B | actual winner |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3DRoad | 5 | 7.6 | 70 | 0.180 | 0.016 | 0.180 | 1.83 | 10.15 | 34.1 | 35.9 | 34.1 | RTNN | hubs | hubs |
| 3DRoad | 50 | 68.3 | 227 | 0.545 | 0.052 | 0.545 | 10.84 | 19.91 | 33.6 | 44.5 | 31.3 | RTNN | hubs | hubs |
| 3DRoad | 128 | 173.1 | 530 | 1.174 | 0.122 | 1.174 | 38.54 | 32.83 | 33.4 | 71.9 | 34.5 | RTNN | hubs | hubs |
| Porto 1M | 5 | 7.4 | 500 | 0.412 | 0.115 | 0.412 | 41.77 | 101.42 | 43.7 | 85.4 | 93.0 | RTNN | RTNN | RTNN |
| Porto 1M | 50 | 69.1 | 1066 | 1.263 | 0.245 | 1.263 | 783.73 | 620.50 | 43.5 | 827.2 | 95.9 | RTNN | RTNN | hubs |
| Porto 1M | 128 | 174.3 | 3103 | 2.716 | 0.714 | 2.716 | 3914.07 | 1441.29 | 42.9 | 3957.0 | 116.5 | RTNN | RTNN | hubs |
| uniform 1M | 5 | 11.0 | 34 | 0.462 | 0.008 | 0.462 | 1.21 | 2.63 | 36.3 | 37.5 | 89.6 | RTNN | RTNN | RTNN |
| uniform 1M | 50 | 104.1 | 154 | 1.746 | 0.035 | 1.746 | 9.34 | 5.35 | 36.8 | 46.2 | 105.8 | RTNN | RTNN | RTNN |
| uniform 1M | 128 | 264.4 | 338 | 3.959 | 0.078 | 3.959 | 42.59 | 10.76 | 36.4 | 79.0 | 129.8 | RTNN | RTNN | RTNN |
| bunny | 5 | 9.0 | 27 | 0.157 | 0.006 | 0.157 | 0.53 | 3.37 | 34.3 | 34.8 | 29.6 | RTNN | hubs | hubs |
| bunny | 50 | 85.3 | 283 | 0.539 | 0.065 | 0.539 | 5.50 | 10.21 | 33.8 | 39.3 | 30.8 | RTNN | hubs | hubs |
| bunny | 128 | 220.3 | 758 | 1.214 | 0.174 | 1.214 | 28.89 | 23.80 | 33.1 | 62.0 | 33.1 | RTNN | hubs | hubs |
| dragon | 5 | 9.1 | 22 | 1.570 | 0.005 | 1.570 | 3.95 | 2.51 | 49.4 | 53.4 | 319.6 | RTNN | RTNN | RTNN |
| dragon | 50 | 84.3 | 262 | 5.318 | 0.060 | 5.318 | 26.32 | 4.95 | 49.4 | 75.7 | 399.4 | RTNN | RTNN | RTNN |
| dragon | 128 | 213.2 | 616 | 11.743 | 0.142 | 11.743 | 138.76 | 11.82 | 43.2 | 182.0 | 496.0 | RTNN | RTNN | RTNN |
| Gowalla | 5 | 8.6 | 1410 | 0.548 | 0.324 | 0.548 | 218.96 | 399.41 | 42.7 | 261.7 | 104.7 | RTNN | RTNN | hubs |
| Gowalla | 50 | 75.4 | 4387 | 1.731 | 1.009 | 1.731 | 813.47 | 470.01 | 42.5 | 856.0 | 112.0 | RTNN | RTNN | hubs |
| Gowalla | 128 | 202.3 | 7960 | 3.974 | 1.831 | 3.974 | 2563.45 | 645.08 | 42.4 | 2605.9 | 135.1 | RTNN | RTNN | hubs |
| Glove 3D | 5 | 11.4 | 26 | 0.558 | 0.006 | 0.558 | 1762.96 | 3160.63 | 46.0 | 1809.0 | 121.6 | RTNN | RTNN | hubs |
| Glove 3D | 50 | 105.0 | 178 | 2.099 | 0.041 | 2.099 | 4640.54 | 2210.38 | 45.7 | 4686.3 | 151.2 | RTNN | RTNN | hubs |
| Glove 3D | 128 | 265.5 | 524 | 4.743 | 0.121 | 4.743 | 7626.31 | 1607.75 | 45.0 | 7671.3 | 206.0 | RTNN | RTNN | hubs |
| Manuscript | 5 | 6.5 | 12 | 0.862 | 0.003 | 0.862 | 76.41 | 88.64 | 44.8 | 121.2 | 164.6 | RTNN | RTNN | RTNN |
| Manuscript | 50 | 75.1 | 97 | 2.904 | 0.022 | 2.904 | 309.70 | 106.66 | 45.0 | 354.7 | 214.7 | RTNN | RTNN | hubs |
| Manuscript | 128 | 181.8 | 229 | 6.077 | 0.053 | 6.077 | 974.10 | 160.28 | 45.6 | 1019.7 | 265.6 | RTNN | RTNN | hubs |

Launch time: 0 of 24 cells inside a factor 2; median absolute log error 3.99 (factor 53.9); median ratio 60.73; chain term binds in 0 cells.
Winner, call A (launch only against hubs): right in 8 of 24 cells.
Winner, call B (launch + measured sort and geometry against hubs): right in 14 of 24 cells.
Baseline "always hubs": right in 16 of 24 cells. Truth: RTNN wins 8 cells.
Wrong calls callA: 3DRoad K 5 (called RTNN, truth hubs); 3DRoad K 50 (called RTNN, truth hubs); 3DRoad K 128 (called RTNN, truth hubs); Porto 1M K 50 (called RTNN, truth hubs); Porto 1M K 128 (called RTNN, truth hubs); bunny K 5 (called RTNN, truth hubs); bunny K 50 (called RTNN, truth hubs); bunny K 128 (called RTNN, truth hubs); Gowalla K 5 (called RTNN, truth hubs); Gowalla K 50 (called RTNN, truth hubs); Gowalla K 128 (called RTNN, truth hubs); Glove 3D K 5 (called RTNN, truth hubs); Glove 3D K 50 (called RTNN, truth hubs); Glove 3D K 128 (called RTNN, truth hubs); Manuscript K 50 (called RTNN, truth hubs); Manuscript K 128 (called RTNN, truth hubs)
Wrong calls callB: Porto 1M K 50 (called RTNN, truth hubs); Porto 1M K 128 (called RTNN, truth hubs); Gowalla K 5 (called RTNN, truth hubs); Gowalla K 50 (called RTNN, truth hubs); Gowalla K 128 (called RTNN, truth hubs); Glove 3D K 5 (called RTNN, truth hubs); Glove 3D K 50 (called RTNN, truth hubs); Glove 3D K 128 (called RTNN, truth hubs); Manuscript K 50 (called RTNN, truth hubs); Manuscript K 128 (called RTNN, truth hubs)
