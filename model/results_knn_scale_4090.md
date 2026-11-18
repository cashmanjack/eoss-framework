# kNN on the 4090: the scaling basis (model/knn_scale_4090.py, existing data only)

F138 kNN launch times were measured on the 4070 Ti (io, 60 SMs, 48 MB L2: model/results_mva3.md header, log queue 15 on io), not on triton. Phase 10f scaled them by 142/128 (wrong chip).

| build round | 4070 Ti ms | x142/128 (used) | measured/that | x60/128 | measured/that |
|---|---|---|---|---|---|
| orig 1 | 0.661 | 0.733 | 0.58 | 0.310 | 1.38 |
| orig 2 | 0.979 | 1.086 | 0.56 | 0.459 | 1.33 |
| orig 3 | 3.502 | 3.885 | 0.56 | 1.642 | 1.32 |
| orig 4 | 11.016 | 12.221 | 0.54 | 5.164 | 1.29 |
| reg 1 | 0.165 | 0.183 | 0.63 | 0.077 | 1.48 |
| reg 2 | 0.225 | 0.250 | 0.53 | 0.105 | 1.26 |
| reg 3 | 0.692 | 0.768 | 0.51 | 0.324 | 1.20 |
| reg 4 | 2.937 | 3.258 | 0.48 | 1.377 | 1.14 |

Two parameter fit over rounds 1 to 4 (least squares, T4090 = a + b x T_io x 60/128; b = 1 means perfect scaling with SM count):
- orig: a = 0.034 ms, b = 1.28 (efficiency 78 percent of SM count scaling), rms log residual 0.012
- reg: a = 0.021 ms, b = 1.13 (efficiency 89 percent of SM count scaling), rms log residual 0.038

rms log error of the corrected basis (x60/128, no fit): 0.27 (the 142/128 basis: 0.61). Not tested: the clocks on the 4070 Ti runs (not sampled in q15), so a clock difference stays open; DRAM bandwidth (504 vs 1,008 GB/s) and L2 (48 vs 72 MB) do not explain slower than scaled.
