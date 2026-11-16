# Arkade with the result buffer in device memory: predictions written before the io run (2026-11-16)
Test: sample01-knn, 100,000 uniform 2D points and queries, radius 0.005, KN 16, NORM 2, L2 build; kernel time by the global timer (first warp start to last warp end). Model: callback floor (F131) plus hand off (F137), no PCIe term, no fit to any device buffer run. Units: ms.
| GPU | program | predicted (model) | scaled from the other GPU by SM count | measured |
|---|---|---|---|---|
| triton (142 SMs) | full | 1.57 (3.667 + 0.040 from the io model, times 60 / 142) | n/a | 1.97 (5 runs, 1.94 to 1.98) |
| triton | empty intersection | 0.017 (hand off alone) | n/a | 0.131 (5 runs) |
| io (60 SMs, 4070 Ti) | full | 3.71 (3.667 + 0.040) | 4.66 (1.97 x 142 / 60) | not yet run (notes/q17/arkade_io.sh, GPU 0) |
| io | empty intersection | 0.04 | 0.31 | not yet run |
Pinned host output on io is known: 20.8 ms (empty), 25.7 ms (full). Prediction for io pinned minus device buffer (the PCIe write cost): about 20 ms for the empty build if the write path is the base cost, which is what the run tests.
What can fail: the model number for the full program on io outside 3.0 to 5.0 ms (the model uses one clock for both chips); the empty build above 0.5 ms on io (a fixed launch cost not in the model).
