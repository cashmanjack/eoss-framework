# RTX 4090 (tethys GPU 1) session 7: L2 write against read cost (F423), RTNN callback calibration (F424), speedup from byte mix and measured costs (F425), 2026-11-19
Scored against model/predictions_4090_batch7.md (commit a34165c, before any run). OptiX and plain CUDA only, GPU 1 idle before every process, ncu with --clock-control none --cache-control none, no TTU code, no fault. Raw (private): notes/l2rw, notes/rtnn_stub4090, notes/ncu4090. Code: ttu/bench/L2rw (l2rw.cu, run_l2rw.py), ttu/bench/Phase10f/rtnn_stub_4090.sh, stub patch notes/rebuild/patches/rtnn_stub_radius_tethys.diff (tree ~/thirdparty/div1_stub, outside git), scorers model/score_4090_batch7a.py, b.py, c.py.
Scope: the 5090's L2 write/read microbenchmark named in the brief was not in the repo when I started (origin/main stopped at 67cdb25), so I wrote my own. The 5090 session pushed its own (ttu/bench/pc14/l2stream.cu, coalesced 512 byte accesses; its F420 to F422) while I worked; part D scores the 5090 side with its numbers. Its findings took F420 to F422, so mine are F423 to F425 (renumbered at the merge). l2rw v1 counted 32 sectors per lane32 access, it is 64 (found from the ncu sector counts); the 4090 raw file is v1, the scorer divides its lane32 values by 2; the program now prints v=2.

## A. L2 write against read cost (F423)
Time per L2 sector (ns) at write share 0, 0.25, 0.5, 0.75, 1 (3 repetitions, spread at most 8 percent; ncu sectors confirm the shares; L2 hit rate 99.3 to 100 percent; L1 bypassed):
| footprint | pattern | wq 0 | 1 | 2 | 3 | 4 | read TB/s | write TB/s | c_w / c_r |
|---|---|---|---|---|---|---|---|---|---|
| 32 MB | coal (128 B per warp) | 0.00816 | 0.00819 | 0.00825 | 0.01017 | 0.01240 | 3.92 | 2.58 | 1.52 |
| 32 MB | lane4 (4 B in its own sector, partial sector writes) | 0.00633 | 0.00593 | 0.00965 | 0.01339 | 0.01591 | 5.05 | 2.01 | 2.51 |
| 32 MB | lane32 (full sector per lane) | 0.00634 | 0.00536 | 0.00882 | 0.01252 | 0.01628 | 5.05 | 1.97 | 2.57 |
8 and 56 MB within 8 percent of 32 MB (lane4 and lane32 within 3 percent) at every mix. Mixes are sub additive: the 50 percent mix is 13 percent (lane4), 20 percent (coal) and 22 percent (lane32) below the mean of the two ends, and 25 percent writes is no slower than reads only for lane4 and lane32 (reads and writes use partly different paths).
Predictions: 1 hit rate at least 98 percent at 8 and 32 MB: HELD (100). 2 read only 3 to 6 TB/s: HELD (5.05 lane, 3.92 coal). 3 c_w / c_r lane4 between 2 and 8, below 6.8: HELD (2.51); not below 1.5. 4 partial sector writes cost more than full ones: MISSED (2.51 against 2.57, equal); coal within 20 percent of lane32: MISSED (24 percent lower per sector). 5 linear mix within 15 percent: HELD for lane4 only (0.87), MISSED for coal (0.80) and lane32 (0.78). 6 footprint independent within 15 percent: HELD (at most 1.08).
Reading: a written sector costs about 2.5 times a read sector in L2 time on this chip (about 1.5 for 128 B coalesced stores), not the 6.8 of the post hoc F419 fit. ncu lts pct of peak is about 95 percent for read only (5.05 TB/s) and for write only (2.0 TB/s) alike, so that percentage is a per unit bottleneck reading, not a linear cost; the F419 fit (cost in percent of peak) overstated the write cost. The F419 finding (original traffic is mostly writes and runs at 1.4 to 1.6 TB/s) stands.

## B. RTNN callback calibration (F424)
Stub = intersection program returns at once (same traversal and program invocations, no distance check, no write; as F216 b), tree ~/thirdparty/div1_stub. ncu kernel (optixLaunch) medians of 3, plain runs 6 each:
| | r 0.05 | r 0.025 |
|---|---|---|
| full kernel (ms) | 0.5816 / 0.5825 / 0.5733 | 0.1148 |
| stub kernel (ms) | 0.2704 / 0.2789 / 0.2787 | 0.0662 |
| stub / full | 0.48 | 0.58 |
| cb (cycles per candidate per SM) | 9.40 | 10.88 |
| cq = 4.9 + cb - 1.01 | 13.29 | 14.77 |
| calibrated prediction (ms) | 0.459 | 0.105 |
| uncalibrated (F255) | 0.191 | 0.058 |
| kernel / calibrated | 1.27 | 1.09 |
| scored (search compute) / calibrated | 1.52 | 2.28 |
Predictions: 1 stub within 0 to 0.05 ms of full: MISSED (0.30 and 0.049 ms; stub / full 0.48 and 0.58). 2 cb between 0 and 3: MISSED (9.4, 10.9). 3 calibrated within 20 percent of the uncalibrated: MISSED (+140 and +81 percent); calibrated kernel inside a factor 2 at r 0.05: happened (1.27; I gave 10 percent); at r 0.025: HELD (1.09). 4 12 of 15: see below, it is 14.
The callback is half the RTNN launch on this input (50 candidates per query end in a result write); triton's cb_ref (1.01, 10.6 candidates per query, a launch where write and test were a few microseconds) does not transfer, so the calibration measured on the chip is what matters here. Calibration used the same input it then scores (not blind).
Rescores (F255 15 rows, labelled; the original F255 and F419 numbers are not edited): as scored 10 of 15 inside a factor 2 (headline 3 of 7); RTNN on kernel time (F418) 11 of 15, headline 4 of 7; calibrated kernel basis (RTNN stub calibrated, kNN rows and speedup on ncu kernel time, Arkade and TrueKNN unchanged) 14 of 15 (median log error 0.56), headline 6 of 7 (the Arkade row, 3.04, is the one outside). The kNN rows on kernel time are 0.57 to 0.59 (original) and 0.52 to 0.56 (register heap), all rounds and totals inside (0.53 total register heap); the register heap speedup on kernel time is 4.34 against the 4.02 predicted (1.08). F419's speedup row: the committed 4.02 prediction is within 8 percent of the kernel speedup 4.34 (event: 4.45).

## C. Speedup from the byte mix and measured costs (F425)
T = R c_r + W c_w with R and W the L2 sectors read and written (ncu, F419 second run) and c_r, c_w from A (no fit to the kNN data); speedup = T(original) / T(register heap). Write shares: original 0.68, 0.70, 0.75, 0.71; register heap 0.27, 0.29, 0.39, 0.38 (rounds 1 to 4).
| 4090 (2 repetitions of the kernel times) | round 1 | 2 | 3 | 4 | total |
|---|---|---|---|---|---|
| measured kernel speedup | 4.34 / 4.44 | 4.92 / 4.64 | 5.50 / 5.11 | 4.20 / 3.92 | 4.48 / 4.20 |
| lane4 linear (primary, committed) | 4.39 | 4.68 | 4.95 | 3.67 | 3.98 |
| error per round | +1 / -1 percent | -5 / +1 | -10 / -3 | -13 / -6 | -11 / -5 |
| coal linear (secondary) | 3.62 | 3.88 | 4.25 | 3.20 | 3.44 (-23 / -18 percent) |
| byte ratio only | 3.06 | 3.28 | 3.67 | 2.81 | 3.00 (-33 / -29 percent) |
| lane4 curve (post hoc, labelled) | 6.03 | 6.32 | 6.18 | 4.51 | 4.94 (+10 / +18 percent) |
Predictions: 1 total within 15 percent: HELD (-11 and -5); within 30: HELD; at least 3 of 4 rounds within 25 percent: HELD (4 of 4 lane4); closer than the byte ratio in every round: HELD. 2 rounds 1 and 2 overpredicted: MISSED (within 5 percent, not over). 3 (5090): see D.
Reading: the measured costs, taken as c_w / c_r = 2.5 and the committed method, predict the 4090's register heap speedup within 5 to 11 percent in total and 1 to 13 percent per round, against 29 to 33 percent for the byte ratio alone. The primary pattern (lane4) was fixed before the runs; coal fits worse (18 to 24 percent low), the interpolated curve, which I added after seeing the sub additive mixes, overshoots (+7 to +39 percent), so the plain two cost form is the best of the three, possibly by partial cancellation of sub additivity against the register heap not being L2 bound in every round (not tested); I would not trust it to extrapolate. Not shown: any statement for the 4070 Ti.

## D. The same method on the 5090 (F425), scored after the 5090 session's numbers arrived (49b3f58; scorer model/score_4090_batch7d.py)
Disclosure: the 5090's measured speedups (F417) were already known; its sector mix was not measured when part C was written (C said: use the 4090 write shares), both are shown. Costs: the 5090 stream (coalesced, 24 MB): read 82.7, write 41.3 sectors per cycle (write 2.0 reads); its mix: write share original 75, 77, 83, 78 percent, register heap 28, 30, 40, 39 (round 1 to 4).
| 5090 | round 1 | 2 | 3 | 4 | total |
|---|---|---|---|---|---|
| measured kernel speedup (F417) | 2.99 | 3.34 | 4.00 | 2.98 | 3.17 |
| method, 5090 mix and costs | 3.96 | 4.35 | 4.78 | 3.47 | 3.77 (+19%) |
| method, 4090 write shares | 3.84 | 4.20 | 4.60 | 3.37 | 3.64 (+15%) |
| byte ratio only | 2.90 | 3.19 | 3.65 | 2.72 | 2.91 (-8%) |
Prediction C3 (if the 5090's write over read cost is at least 30 percent below the 4090's, the predicted 5090 speedup is below the 4090's and within 25 percent of 3.17): MISSED in its premise and in the ordering: the like for like (coalesced) write over read cost is 2.0 on the 5090 against 1.52 on the 4090 (31 percent HIGHER; my per lane pattern 2.51 has no 5090 counterpart), so the method predicts the 5090 total (3.77) above the 4090's coalesced prediction (3.44), and the measured order is the opposite (3.17 against 4.2 to 4.5); the size part (within 25 percent) holds (+19). The unweighted byte ratio is closer on the 5090 (-8 percent) than the weighted method.
Reading: the F425 method matches the 4090 (with its per lane costs) but overpredicts the 5090 by 15 to 32 percent and gets the cross chip order wrong with the like for like pattern. So the cost weighted explanation of Ada's higher speedup is NOT established: either the per lane sector costs differ on the 5090 (not measured there: my l2rw has to run on pc14), or something else makes the 5090 original relatively cheaper (the 5090 builds are at 37 and 41 sectors per cycle, below their stream limits, so the stream costs may not be what binds). The two chips' stream benchmarks are different programs (4 B per lane 128 B per warp here against 16 B per lane 512 B per warp there), so the 1.52 against 2.0 is not a clean chip difference either.
