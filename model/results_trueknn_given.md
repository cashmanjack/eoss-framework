# TrueKNN, callback cost GIVEN (measured per round in the same runs): NOT A PREDICTION

2026-11-16, triton GPU 1, rebuilt program (Arkade OWL tree port of s01-trueknn, /dev/shm/tk, third party code not committed). Raw runs stamp 20261116T134518 (private, notes/q21/tk_new). Points: uniform random in the unit cube, regenerated (numpy seeds 1, 2, 3), so they are NOT the F156 point files; k = 16, start radius 0.004, radius doubling, refit between rounds, one launch per round, 5 runs of A and 3 runs of each held size.

Constants from A (rebuilt, 5 runs, medians, pooled over rounds): rho (hardware warp calls over exact warp max box count) 1.044, g (warp cycles over callback cycles) 1.485, W (warps in flight per SM) 12.82. Per round callback cost c_r is GIVEN: callback cycles / warp calls of that round, median of 3 runs at that size. Model: T_r = rho x exact calls_r x c_r x g / (142 W clk). Second column group: calls_r taken from the measurement.

A per round, rebuilt vs the F156 build (medians of 5): span ms rebuilt 0.222 0.215 0.373 1.447 0.757
F156 build: 0.131 0.066 0.266 1.384 0.680; callback cycles per call rebuilt 80670 51599 39548 36556 2794
Per run spread of the round spans in A (min to max over 5 runs): 0.220-0.222 0.213-0.217 0.366-0.375 1.431-1.455 0.743-0.779

## 200000 points (3 runs)

| round | radius | active pred | active meas | calls pred | calls meas | c given (cycles per call) | T pred (calls pred) ms | T pred (calls meas) ms | T meas ms (min to max) | log err | log err (calls meas) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.004 | 200000 | 200000 | 13822 | 17986 | 72573 | 0.3273 | 0.4259 | 0.4147 (0.4106 to 0.4178) | -0.24 | +0.03 |
| 2 | 0.008 | 200000 | 200000 | 26986 | 33750 | 47001 | 0.4139 | 0.5176 | 0.4577 (0.4506 to 0.4608) | -0.10 | +0.12 |
| 3 | 0.016 | 200000 | 200000 | 86553 | 113056 | 37574 | 1.0612 | 1.3862 | 0.9718 (0.9615 to 0.9738) | +0.09 | +0.36 |
| 4 | 0.032 | 199766 | 199604 | 445521 | 608305 | 26823 | 3.8995 | 5.3243 | 3.3772 (3.3700 to 3.3864) | +0.14 | +0.46 |
| 5 | 0.064 | 174 | 142 | 23362 | 18129 | 2583 | 0.0197 | 0.0153 | 0.5509 (0.5499 to 0.5601) | -3.33 | -3.58 |
| total | | | | | | | 5.7216 | 7.6693 | 5.7723 | -0.01 | +0.28 |

Implied per round (medians of the runs), pooled A values in brackets: W = total warp cycles / (142 clk span) 15.7 15.8 16.4 16.5 3.6 [12.8]; g = total warp cycles / callback cycles 1.77 1.62 1.33 1.22 14.86 [1.48].

rounds predicted 5, measured 5; rms per round log error 1.50 (calls predicted), 1.62 (calls measured). F156 total log error for this size: -0.64; F158: +0.44.

## 400000 points (3 runs)

| round | radius | active pred | active meas | calls pred | calls meas | c given (cycles per call) | T pred (calls pred) ms | T pred (calls meas) ms | T meas ms (min to max) | log err | log err (calls meas) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.004 | 400000 | 400000 | 32422 | 43849 | 70887 | 0.7500 | 1.0143 | 0.8305 (0.8243 to 0.8479) | -0.10 | +0.20 |
| 2 | 0.008 | 400000 | 400000 | 75046 | 102662 | 40088 | 0.9817 | 1.3429 | 1.0076 (0.9964 to 1.0179) | -0.03 | +0.29 |
| 3 | 0.016 | 400000 | 400000 | 287139 | 419735 | 35644 | 3.3397 | 4.8819 | 2.9430 (2.9266 to 3.0300) | +0.13 | +0.51 |
| 4 | 0.032 | 307795 | 284365 | 1608174 | 2285571 | 13917 | 7.3030 | 10.3792 | 6.3590 (6.3283 to 6.5731) | +0.14 | +0.49 |
| 5 | 0.064 | 1 | 1 | 126 | 125 | 4012 | 0.0002 | 0.0002 | 0.6011 (0.5949 to 0.6216) | -8.20 | -8.21 |
| total | | | | | | | 12.3745 | 17.6185 | 11.7412 | +0.05 | +0.41 |

Implied per round (medians of the runs), pooled A values in brackets: W = total warp cycles / (142 clk span) 16.6 16.7 17.8 17.9 5.0 [12.8]; g = total warp cycles / callback cycles 1.57 1.46 1.24 1.27 2135.62 [1.48].

rounds predicted 5, measured 5; rms per round log error 3.67 (calls predicted), 3.69 (calls measured). F156 total log error for this size: -0.16; F158: +0.82.

Reading: the cost per call is given, so this checks only the rest of the model (calls, g, W). It does not predict the cost per call. The active set and call counts come from the CPU box rule as in F156. The small totals with predicted calls are partly a cancellation: the call count is under predicted by 1.3 to 1.4 times in the big rounds, and with the measured call count the same model is 1.3 to 1.5 times too high (+0.28, +0.41). The last round (1 or 142 active queries) is a latency bound chain that a throughput model cannot give (0.55 to 0.60 ms measured). The rebuilt A differs from the F156 build in rounds 1 to 3 (see above), so these numbers are not a like for like comparison with F156 or F158.
