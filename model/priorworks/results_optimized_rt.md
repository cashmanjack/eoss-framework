# Optimized RT, kill test: the known callback fixes on RTNN knn mode and TrueKNN (2026-11-18, triton GPU 1, OptiX only, no TTU code)

Predictions: predictions_phase12.md part C (commit 9fcc6c3), written before any run. Device scope as F247, median of 3 runs (TrueKNN: interleaved with the original, one dump pair per dataset).

What was applicable (read from the sources first):
- Distance in float (the F151 fix): both programs already compute it in float (RTNN `float sqdist`, TrueKNN `std::sqrt` of floats; the build uses fast math). Nothing to change.
- Register heap (F138): needs the K entries in registers. F138's heap had 16 entries; K 50 and 128 need 100 and 256 registers for keys and indices and do not fit (255 register limit), so the fix was built and tested at K 5 only.
  - TrueKNN K 5 (copy of the timer build, tk_orig_vani/src_reg, one macro): the 5 entries (index, distance) of the query are loaded with independent loads, the insertion is done in registers with constant indices, only the shifted entries are stored back; the counters (intersections, numNeighbors) are updated as before. The final neighbour buffer (index, distance, counters of every query) is compared byte for byte with the original's dump: identical on all 8 datasets.
  - RTNN K 5 (thirdparty/pw/rt_fix, built on the recorder tree with the probes off): RTNN's heap is two local arrays behind a pointer; in the fix the 5 keys, the position of the maximum, the size and the maximum key travel in the 8 payload registers (the limit of the OptiX 7.1 headers RTNN uses), and the intersection program writes the index straight to the result buffer (write only). Same insertion rule including the rescan for the maximum. RTNN orders points and queries differently between runs (sort ties), so lists cannot be compared by index across runs: per query (found by coordinates) the sorted distance lists of the two builds are compared: identical for every query on all 8 datasets (an original against an original gives the same, 1.0000).
- K 50 and 128: bounded instead with RTNN's stub build (the intersection program returns at once: same traversal, same program invocations, no distance, no insert, no result), built for K 5, 50 and 128 (the K 50 stub is the one of F216). If a free callback body does not change the winner, no fix of the callback body can.
TrueKNN has no stub (its rounds stop when the lists fill, which a stub cannot do), so TrueKNN at K 50 and 128 is not bounded here.

K 5 register heap (TrueKNN device scope; RTNN search compute):

| dataset | TrueKNN K 5 device ms, original | register heap | speedup | whole neighbour buffer identical | RTNN K 5 search ms, original | payload heap | speedup | distance lists identical | hubs K 5 ms |
|---|---|---|---|---|---|---|---|---|---|
| 3DRoad | 56.1 | 52.9 | 1.06 | yes | 1.83 | 1.93 | 0.95 | 1.0000 | 34.1 |
| Porto 1M | 879.9 | 736.4 | 1.19 | yes | 42.30 | 42.65 | 0.99 | 1.0000 | 93.0 |
| uniform 1M | 36.2 | 35.6 | 1.02 | yes | 1.21 | 1.22 | 1.00 | 1.0000 | 89.6 |
| bunny | 76.5 | 67.9 | 1.13 | yes | 0.53 | 0.54 | 0.99 | 1.0000 | 29.6 |
| dragon | 301.8 | 300.4 | 1.00 | yes | 2.89 | 2.89 | 1.00 | 1.0000 | 319.6 |
| Gowalla | 252.8 | 228.4 | 1.11 | yes | 203.04 | 211.87 | 0.96 | 1.0000 | 104.7 |
| Glove 3D | 72.5 | 69.6 | 1.04 | yes | 1725.50 | 1762.33 | 0.98 | 1.0000 | 121.6 |
| Manuscript | 75.8 | 72.6 | 1.04 | yes | 76.10 | 78.42 | 0.97 | 1.0000 | 164.6 |

Free callback bound (RTNN stub, device scope; hubs from F245):

| dataset | K | RTNN search ms (F245 builds) | RTNN search ms, free callback (stub) | RTNN device total ms | with free callback | hubs ms | winner with free callback |
|---|---|---|---|---|---|---|---|
| 3DRoad | 5 | 1.8 | 1.9 | 35.9 | 35.5 | 34.1 | hubs |
| 3DRoad | 50 | 10.8 | 11.1 | 44.5 | 44.8 | 31.3 | hubs |
| 3DRoad | 128 | 38.5 | 43.9 | 71.9 | 77.6 | 34.5 | hubs |
| Porto 1M | 5 | 41.8 | 42.7 | 85.5 | 86.3 | 93.0 | RTNN |
| Porto 1M | 50 | 783.7 | 811.0 | 827.2 | 854.7 | 95.9 | hubs |
| Porto 1M | 128 | 3914.1 | 4003.7 | 3957.0 | 4047.5 | 116.5 | hubs |
| uniform 1M | 5 | 1.2 | 1.2 | 37.5 | 37.4 | 89.6 | RTNN |
| uniform 1M | 50 | 9.3 | 9.4 | 46.3 | 45.6 | 105.8 | RTNN |
| uniform 1M | 128 | 42.6 | 41.9 | 79.0 | 78.3 | 129.8 | RTNN |
| bunny | 5 | 0.5 | 0.5 | 34.8 | 34.1 | 29.6 | hubs |
| bunny | 50 | 5.5 | 5.4 | 39.3 | 39.0 | 30.8 | hubs |
| bunny | 128 | 28.9 | 28.5 | 62.0 | 62.0 | 33.1 | hubs |
| dragon | 5 | 3.9 | 2.9 | 53.4 | 52.3 | 319.6 | RTNN |
| dragon | 50 | 26.3 | 26.3 | 75.7 | 75.6 | 399.4 | RTNN |
| dragon | 128 | 138.8 | 136.4 | 181.8 | 179.8 | 496.0 | RTNN |
| Gowalla | 5 | 219.0 | 214.8 | 261.8 | 258.0 | 104.7 | hubs |
| Gowalla | 50 | 813.5 | 808.7 | 856.0 | 851.2 | 112.0 | hubs |
| Gowalla | 128 | 2563.4 | 2444.4 | 2606.6 | 2486.9 | 135.1 | hubs |
| Glove 3D | 5 | 1763.0 | 1812.4 | 1809.0 | 1857.7 | 121.6 | hubs |
| Glove 3D | 50 | 4640.5 | 4437.5 | 4685.8 | 4482.3 | 151.2 | hubs |
| Glove 3D | 128 | 7626.3 | 8574.5 | 7670.7 | 8618.7 | 206.0 | hubs |
| Manuscript | 5 | 76.4 | 77.8 | 121.2 | 121.2 | 164.6 | RTNN |
| Manuscript | 50 | 309.7 | 312.3 | 354.4 | 357.2 | 214.7 | hubs |
| Manuscript | 128 | 974.1 | 1002.8 | 1019.7 | 1047.7 | 265.6 | hubs |

Result:
- K 5 register heap: TrueKNN 1.00 to 1.19 times faster in device scope (1.00 to 1.16 in total time), bit identical buffers; RTNN's search compute 0.95 to 1.00 times (no gain, slightly slower on the large Gowalla, Glove 3D and Manuscript cells), identical distance lists. No K 5 cell moves from hubs to RT: where TrueKNN already beat hubs (uniform 1M, dragon, Glove 3D, Manuscript) it still does and where it lost (3DRoad, Porto, bunny, Gowalla) it still loses by 1.5 to 7.9 times.
- Stub bound (second table): a free callback changes RTNN's search time by -27 to +14 percent (median +0.7 percent; the extremes are the cells with search times of 3 to 4 ms) in all 24 cells, and the winner against hubs does not change in any of them. So the cost of RTNN in the cells where it loses is not the callback body: it is the number of programs invoked per query (every point inside the query's box costs one invocation, F49: 850 to 900 cycles for a custom primitive) and the traversal that finds them, plus on 3DRoad and bunny a fixed 33 to 40 ms of query sort and geometry build (search is 1.8 and 0.5 ms at K 5).
- Cells that move from hubs to RT with all applicable fixes: 0 of 24 (K 5 real fix 0; K 50 and 128 free callback bound 0 for RTNN; TrueKNN at K 50 and 128 not bounded, it is 5.9 to 100 times slower than hubs there so it would need a callback speedup of that size).
- Written predictions: bit identical results HELD (8 of 8 TrueKNN, 8 of 8 RTNN); RTNN K 5 speedup 1.1 to 1.6 MISSED (0.95 to 1.00), TrueKNN 1.0 to 1.3 times HELD (1.00 to 1.16 total); at most 1 K 5 cell moves HELD (0); free callback still slower than hubs on the six clustered cells HELD, and RTNN beats hubs on 3DRoad and bunny at K 50 and 128 with a free callback MISSED (it does not; 40 percent); 0 to 3 cells move in total HELD (0).
- Caveats: the register heap is K 5 only; RTNN heap fix keeps the index writes in global memory (write only), so it is a payload register heap, not F138's exact form; one GPU; the stub still pays the program invocation, so it bounds callback body fixes only, not fixes that avoid invoking the program (for example hardware triangle tests, which F49 and F204 put at a different cost per candidate).
