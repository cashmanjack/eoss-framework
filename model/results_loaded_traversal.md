# Loaded traversal test: launch ramp or queueing (F169; form and decision rule: model/FORM_loaded_traversal.md, committed before the run)

2026-11-16, triton GPU 1, OptiX only (ttu/bench/Load_1; raw data private in notes/q28). 100,000 disjoint boxes on a lattice, one point query per box (tmax 1e-16), 1.03 intersection calls per query (1.000 at NQ 32), every query hits. Span = earliest warp start to latest warp end (global timer, 1.024 microsecond steps, so the 4 to 6 microsecond rows are coarse), median of 20 launches after 2 untimed; warp cycles = raygen start to end (late starts excluded); W = warps in flight per SM = sum of warp cycles / (142 x span in cycles); c = cycles per query per SM; SM clock 2.42 GHz from the warp cycles over warp time of the largest launch.


order lattice SM clock from the largest launch 2.419 GHz
| NQ | span us (median of 20) | cold us | warp cycles mean (median of 20) | warps/SM | W in flight | c cycles per query per SM | queries per ns | calls per query |
|---|---|---|---|---|---|---|---|---|
| 32 | 4.10 | 5.12 | 9105 | 0.01 | 0.01 | 43968.4 | 0.01 | 1.000 |
| 1024 | 5.12 | 6.14 | 10588 | 0.23 | 0.19 | 1717.5 | 0.20 | 1.034 |
| 4544 | 5.12 | 6.14 | 10586 | 1.00 | 0.85 | 387.0 | 0.89 | 1.032 |
| 9088 | 5.12 | 5.12 | 10826 | 2.00 | 1.75 | 193.5 | 1.77 | 1.043 |
| 18176 | 5.12 | 6.14 | 11396 | 4.00 | 3.68 | 96.8 | 3.55 | 1.039 |
| 36352 | 6.14 | 6.14 | 11918 | 8.00 | 6.42 | 58.1 | 5.92 | 1.033 |
| 100000 | 12.29 | 13.31 | 13456 | 22.01 | 9.96 | 42.2 | 8.14 | 1.029 |
| 409600 | 36.86 | 37.89 | 12188 | 90.14 | 12.32 | 30.9 | 11.11 | 1.029 |
| 1638400 | 134.14 | 135.17 | 11745 | 360.56 | 13.05 | 28.1 | 12.21 | 1.029 |
fit S = a + b NQ (NQ >= 4544): a 4.12 us, b 0.0794 ns per query (12.60 queries per ns); intercept share a / S(100,000) = 0.34
L(100,000) / L(4,544) = 1.27; L(36,352; 8 warps per SM) / L(4,544) = 1.13; c max/min over NQ >= 18,176 = 3.44; c(4,544) / c(100,000) = 9.2

order random SM clock from the largest launch 2.422 GHz
| NQ | span us (median of 20) | cold us | warp cycles mean (median of 20) | warps/SM | W in flight | c cycles per query per SM | queries per ns | calls per query |
|---|---|---|---|---|---|---|---|---|
| 32 | 4.10 | 6.14 | 10150 | 0.01 | 0.01 | 44025.9 | 0.01 | 1.000 |
| 1024 | 5.12 | 7.17 | 12305 | 0.23 | 0.22 | 1719.8 | 0.20 | 1.037 |
| 4544 | 5.63 | 6.14 | 12107 | 1.00 | 0.89 | 426.3 | 0.81 | 1.031 |
| 9088 | 6.14 | 6.14 | 12478 | 2.00 | 1.68 | 232.5 | 1.48 | 1.033 |
| 18176 | 6.14 | 7.17 | 13245 | 4.00 | 3.56 | 116.3 | 2.96 | 1.030 |
| 36352 | 8.19 | 8.19 | 15542 | 8.00 | 6.27 | 77.5 | 4.44 | 1.030 |
| 100000 | 16.90 | 18.43 | 19970 | 22.01 | 10.74 | 58.1 | 5.92 | 1.029 |
| 409600 | 53.25 | 53.25 | 17860 | 90.14 | 12.48 | 44.7 | 7.69 | 1.029 |
| 1638400 | 184.32 | 186.37 | 16285 | 360.56 | 13.15 | 38.7 | 8.89 | 1.029 |
fit S = a + b NQ (NQ >= 4544): a 5.36 us, b 0.1097 ns per query (9.12 queries per ns); intercept share a / S(100,000) = 0.32
L(100,000) / L(4,544) = 1.65; L(36,352; 8 warps per SM) / L(4,544) = 1.28; c max/min over NQ >= 18,176 = 3.00; c(4,544) / c(100,000) = 7.3

Verdict by the rule set before the run: NOT SEPARATED. Queueing needed intercept share <= 0.25 (measured 0.34 and 0.32), warp latency at 100,000 at least 2 times the one warp per SM value (1.27 lattice, 1.65 random) and c max/min <= 1.5 (3.4, 3.0). Launch ramp needed intercept share >= 0.5 (0.34, 0.32). Both fail.

What the data does show
- No load dependent latency worth the name: warp latency goes from 9,100 cycles (one warp) to 10,600 (one warp per SM) to 13,500 at 100,000 and then falls again to 11,700 at 1.6 million (lattice order); random order 12,100 to 20,000 to 16,300. Warps in flight saturate at 13 of 22. This is not the 7 times growth of Mochi's loaded warps (F168).
- The span is a straight line: 4.1 microseconds (about one warp latency) plus 0.079 ns per query, 12.6 queries per ns over the whole GPU (random order 5.4 microseconds plus 0.110 ns, 9.1 per ns). The unit bound of F105 is about 26 queries per ns (0.153 per cycle per TPC, 71 TPCs, 2.42 GHz), so the lattice reaches about half and the random order about a third.
- A 100,000 query launch with 1 candidate takes 12.3 microseconds (lattice) or 16.9 (random), 28 to 42 cycles per query per SM. The Mochi launch of the same size and about the same candidates took 100 microseconds (355 cycles per query per SM, F168). So the Mochi gap is NOT a generic cost of a loaded traversal launch: 6 to 8 times of it is something this lattice test does not have.
- The F127 form (110 cycles per query per SM fixed) gives 34 microseconds here, 2.8 times too high for this case (measured 12.3), consistent with F127's own statement that the constant fails at the smallest radius. The F168 model was low for Mochi and high here.

Not tested (candidates for the Mochi difference, none run): the Mochi tree is built with update support and refitted every frame (a different tree quality and node count; the node count per query here was not measured, the one warp latency is similar, 9,100 against 11,205 cycles); the Mochi intersection program and raygen do far more work than here (force gathers and scatter, F168 callbacks loaded 1,100 to 5,000 cycles per call against a trivial program here); the 400 launch sequence with other kernels between launches; query positions (a falling lattice against exact box centres). Stopped as instructed.
