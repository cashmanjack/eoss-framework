# MVA with issue, L1, L2, DRAM stations: predictions (written before any comparison; form and kill rule in model/q19/FORM_mva3.md, committed first)
Constants: c_i 0.25, c_tag 1.00, c_x 1.90 (F131), lane sharing 1.1 (F123), DRAM 9.52 cycles per sector per SM (504 GB/s over 60 SMs), X_tu 0.14, hand off 1 / 0.00774 = 129 cycles per call, clock 2.5 GHz.
Per call demands in SM cycles (D_i issue, D_l1, D_l2, D_d DRAM); n_f = RT node fetches per call; h = L1 hit rate of global loads, L2h = L2 sector hit rate (ncu, same launch). F149 = the single station f of F149 with TU and hand off.

| launch | calls | e | W | n_f | h | L2h | D_i | D_l1 | D_l2 | D_d | f (F149) | C | L unl | split: L | split: T ms | F149: L | F149: T ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| knn reg round 1 | 15427 | 0.513 | 14.2 | 72.6 | 0.637 | 0.833 | 74 | 366 | 597 | 742 | 498 | 9868 | 4091 | 7352 | 0.095 | 4880 | 0.078 |
| knn reg round 2 | 38366 | 0.427 | 14.2 | 34.8 | 0.636 | 0.884 | 73 | 243 | 342 | 290 | 351 | 6651 | 3214 | 4173 | 0.138 | 3818 | 0.132 |
| knn reg round 3 | 157598 | 0.554 | 14.7 | 12.8 | 0.622 | 0.962 | 75 | 231 | 236 | 71 | 373 | 5227 | 2957 | 3516 | 0.430 | 4195 | 0.475 |
| knn reg round 4 | 893589 | 0.745 | 15.1 | 4.8 | 0.597 | 0.992 | 74 | 181 | 166 | 12 | 310 | 4316 | 2570 | 2847 | 2.267 | 3399 | 2.373 |
| knn orig round 1 | 15427 | 0.513 | 14.7 | 72.6 | 0.654 | 0.941 | 81 | 1495 | 1329 | 748 | 2358 | 18043 | 12531 | 21025 | 0.185 | 31008 | 0.255 |
| knn orig round 2 | 38366 | 0.427 | 14.6 | 34.8 | 0.671 | 0.961 | 80 | 998 | 801 | 294 | 1566 | 12738 | 9362 | 14146 | 0.308 | 20791 | 0.424 |
| knn orig round 3 | 157598 | 0.554 | 15.0 | 12.8 | 0.674 | 0.990 | 77 | 909 | 634 | 67 | 1450 | 10705 | 8372 | 13232 | 1.093 | 20393 | 1.593 |
| knn orig round 4 | 893589 | 0.745 | 15.3 | 4.8 | 0.649 | 0.997 | 64 | 545 | 392 | 11 | 900 | 7569 | 5801 | 8132 | 3.937 | 12656 | 5.666 |
