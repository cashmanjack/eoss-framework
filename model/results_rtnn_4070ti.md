# RTNN on the 4070 Ti against the predictions of ca7620e (2026-11-15, io GPU 0, OptiX only)
Measured with `-m 0` (CUDA synchronizations on, so "time search compute" is the real launch time plus the fill of the output buffer). Mistake: the first set ran with RTNN's default `-m 1` (end to end mode, no synchronization), where that timer only measures the launch call (0.13 to 0.82 ms whatever the radius); it was discarded as a measurement of the predicted quantity and redone with `-m 0`. 3 repeats each, spread under 3 percent except u1e5 r4 unsorted (10.0 to 12.5 in the async run), median shown. Neighbors per query equal the CPU sphere count in all six cases (4.285, 26.62, 49.77, 5.14, 33.74, 49.98 against 4.29, 26.62, 49.8 expected) and `-c 1` found 0 wrong neighbors.

```
set r | pred range ms (point) | measured raw | inside | log err point | net of baseline | cycles per candidate per SM equivalent (SMeq 41.4 .. 60, 2.5 GHz) raw / net
u1e5 r2 | 0.039..0.165 (0.080) | 0.216 | NO | +0.99 | 0.048 | raw 27.2..39.5 net 6.1..8.8
u1e5 r4 | 0.238..1.010 (0.490) | 0.467 | yes | -0.05 | 0.299 | raw 9.6..14.0 net 6.2..8.9
u1e5 r6 | 0.441..1.872 (0.909) | 0.747 | yes | -0.20 | 0.579 | raw 8.3..12.1 net 6.4..9.3
u1e6 r1 | 0.468..1.987 (0.965) | 1.583 | yes | +0.49 | 0.401 | raw 16.6..24.1 net 4.2..6.1
u1e6 r2 | 3.039..12.897 (6.261) | 4.176 | yes | -0.40 | 2.994 | raw 6.7..9.8 net 4.8..7.0
u1e6 r3 | 4.495..19.076 (9.260) | 6.413 | yes | -0.37 | 5.231 | raw 7.0..10.2 net 5.7..8.3
rms log error of the point prediction, raw: 0.51 (factor 1.67)
ratios measured raw vs net vs predicted:
 u1e5 r2 -> r4: raw 2.16 net 6.23 predicted 6.11
 u1e5 r4 -> r6: raw 1.60 net 1.94 predicted 1.85
 u1e6 r1 -> r2: raw 2.64 net 7.47 predicted 6.49
 u1e6 r2 -> r3: raw 1.54 net 1.75 predicted 1.48
unsorted / sorted: u1e5 r4 22.9 (pred <= 1.2), u1e6 r2 3.98 (pred >= 1.5)
```

Reading (all of it is after the fact except the first column pair):
- 5 of 6 raw times are inside the predicted ranges; rms log error of the point prediction 0.51 (a factor 1.7). The one miss (u1e5 r2, 0.216 against at most 0.165 ms) is the case where the fixed cost matters: a run with radius 0.01 (almost no candidates) takes 0.168 ms (1e5) and 1.182 ms (1e6): fill of the K x N output buffer (20 and 200 MB) and the launch. The model had no such term (stated as a known hole in the predictions).
- Removing that baseline (post hoc, not a prediction): the ratios between radii follow the prediction (6.2 against 6.1, 1.9 against 1.85, 7.5 against 6.5, 1.75 against 1.5), i.e. time is linear in candidates per query. Per candidate cost 4 to 9 cycles per SM equivalent (net), 6.7 to 10 (raw, large cases). My bracket for c was 7.4 to 20 cycles: the measured value sits at or below the bottom, so the RT unit service rate, not the SM side, was the limit and the 850 to 900 cycle round trip of a custom primitive candidate (F49) is fully hidden by concurrency in throughput. (SMeq 41.4 to 60 assumed, 2.5 GHz assumed, not measured here.)
- Sorting: unsorted points and queries (-qs 0 -ps 0) cost 3.98 times (u1e6 r2; predicted at least 1.5, right) and 22.9 times (u1e5 r4; predicted at most 1.2, WRONG by a factor 19). The prediction misapplied Finding 5: a 2 MB BVH is far outside L1 (the SM's node cache holds about 60 KB), so random order costs the L1 miss and warp divergence penalty even though everything fits the L2.
- RTNN's default configuration (query partitioning and GAS sort) has the same compute time as `-p 0 -s 0` (0.456 against 0.467 ms; 4.02 against 4.18 ms) but its host side takes 0.86 s and 0.97 s end to end against 0.06 and 0.15 s: on these uniform clouds the partitioning machinery is pure overhead.
- Not tested: KNN mode, non uniform data, real point clouds, the RT core's own share of the time (no ncu run), 30 TPC versus 60 SM contribution (SMeq was a bracket, not measured).
