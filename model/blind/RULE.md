# Blind test rule, version 2 (2026-11-24, tethys; written before any new dataset exists)

Replaces version 1 (commit b5196fa), which made the uncalibrated model v1 the test. That tested a form we already knew fails (Finding 245, Phase 13). The primary
test is now the calibrated model of F422 and F424 (`model/FROZEN.md`, "Versions under test"). Version 1 stays in git history; the tools and guards built for it are kept.
Changing this file after a calibration or prediction file of a new dataset exists voids that dataset.

## The rule

**Callback cost may be measured only through the fixed procedures CP-RTNN, CP-TK and CP-ARK below, each run once per dataset on the sample defined here. Every other
number in a prediction comes from the CPU or from constants in `model/FROZEN.md`. All predictions (primary, secondary, relative) are committed and pushed before any
full run of any program on the dataset.**

Allowed before the predictions are committed:
1. Anything computed on the CPU from the dataset file (counts, radii, replay).
2. The sample runs of the three procedures, exactly as written, and the Arkade reference launch (a known input, not new data).
3. The GPU's SM count from its specification (table in `model/FROZEN.md`).
4. Building the BVH of RTNN on the dataset and dumping it (RTNN_DUMP: the GAS, the points, the radius and the queries in launch order) in the sorted and in the unsorted configuration, **with no timing** (revision 2.3): the process is
   killed through its own handle as soon as the query file is complete, its output is discarded and never parsed or stored, and the dump files stay private (never in git). The dump is the input of the CPU replay of row R3.

Not allowed, in any form, before the predictions are committed:
- a full run (all queries or all points) of RTNN, TrueKNN, Arkade, kNN or any other program on the dataset, in any configuration, including the relative rows;
- any other sample (size, seed, fraction, subset, other radius or K), a second sample run chosen after seeing the first, or a change of a procedure after its first use;
- any ncu, counter, probe reading or timing taken on the dataset other than the output of the procedures (the dump of item 4 keeps no timing);
- a constant adjusted after seeing a measurement.
A procedure run that fails (crash, timeout) is reported and repeated identically (same seed and command); if it fails again that mapping is a missing row for that dataset.
After the full measurement a fitted or recalibrated variant may be computed, labelled POST HOC, and is never scored as a blind prediction.

## Rows

Per dataset and per K (the dataset list and the K list are fixed in `PASS.md` before the first calibration):

| row | version | predicted by | measured quantity (full run) |
|---|---|---|---|
| RTNN search launch | PRIMARY v1-S | CP-RTNN + replay | RTNN "time search compute", median of 3, radius mode `-r R -k K -p 0 -s 0 -m 0 -d 0` (no `-sm`; revision 2.2) |
| TrueKNN launch spans | PRIMARY v1-S | CP-TK + replay | sum over rounds of the recorder's per round GT span (ms), median of 3 |
| Arkade kernel, device output, float | PRIMARY v1-S | CP-ARK + reference launch + replay | `KERNEL queries` event ms, DEVOUT=1, recorder off, median of 3 |
| RTNN search launch | SECONDARY v1 uncalibrated | replay only (`blind_predict.py`) | same as the RTNN primary row |
| Arkade pinned over device kernel time | RELATIVE R1 | primary Arkade prediction + 0.33 ms per MB | pinned kernel ms / device kernel ms, same build, medians of 3 |
| RTNN unsorted over sorted queries and points | RELATIVE R3 | BVH dump + CPU replay + frozen reuse model (FROZEN.md) | RTNN `KERNEL queries` event ms, `-qs 0 -ps 0` over defaults, radius mode, medians of 3 |
| kNN register heap over original, launch time | RELATIVE R2 | 4.02 (F138) | sum of the GPU time of all `optixLaunch` kernels (ncu `gpu__time_duration`, clocks not locked), original / register heap, medians of 3 |

Excluded and why (so no row can be added after the fact): the ray ordering speedups of F113 to F123 are for triangle scenes with camera and bounce rays, not for point datasets; TrueKNN, Arkade and kNN have no frozen reordering predictor.
The kNN register heap predictor is the F138 constant, which does not depend on the dataset; it is scored because F255 scored it. The RTNN reordering row R3 (excluded in versions 2.0 to 2.2) is restored by revision 2.3.

## The procedures (all use the 4090 or whichever GPU the test names; one GPU, idle by nvidia-smi PIDs and utilization, never by process name; a 2 s warmup process before every run; `timeout -k 2`; SM clock recorded per run)

Common: the dataset is a text file with one point per line (3 columns, comma or space; Arkade uses columns 1 and 2, z = 0). Points are float32 widened to float64 for the CPU.
K and the datasets are those announced in `PASS.md`. Radius rule R(K) (F245): the smallest radius at which 99 percent of 3,000 queries drawn with `default_rng(5)` have K other points
inside (kd tree, L2, the K+1 th distance quantile 0.99). r0 (TrueKNN, Arkade) = the smallest nonzero distance among the 4 nearest neighbours of 100 points drawn with `default_rng(5)`.

**CP-RTNN** (variant S of Phase 13, F422/F424 definition). nq = min(N, max(round(0.02 N), 50,000)) (revision 2.2, see below); sample = the queries `idx = sort(default_rng(11).choice(N, nq, replace=False))`, written to a file,
points = the whole file. Run the real build and the stub build of RTNN (build for this K) 3 times each in radius mode: `-f DATA -q SAMPLE -r R(K) -k K -p 0 -s 0 -m 0 -d 0`, `RTNN_DRAM_GB=11.5958`.
T_real, T_stub = medians of "time search compute". cbS = max(0, (T_real - T_stub) x 1e-3 x SM x 2.5e9 / (nq x 1.10 x mean cube count of the sampled queries)), d_S = max(cbS - 1.0110154, -4.9).
Prediction = `model_v1_rtnn.predict(N, K, C, S, d_S)` with C, S the replay counts of all N queries. Stub build: the F216 stub (no distance check, no result write, traversal and calls unchanged).

**CP-TK** (F216 (b) with one stated extension). m = max(5,000, round(0.02 N)); sample = the points `sort(default_rng(21).choice(N, m, replace=False))` written to a file (points and queries both,
as the program does). Run the port (`sample03-trueknn SAMPLE m 3 r0 K OUT`, device buffer build) 3 times; per run take the sum over all rounds of f0_cycles. cpc = median over runs of
(sum of f0_cycles) / (sum over the same rounds of replay box hits on the sample, computed on the CPU with the same radii and stop rule). For the full dataset the CPU replay gives per round the
active set, the box hits h_r and the largest active box count c_r (F158 stop rule, radius doubling from r0, at most 60 rounds). Prediction = sum over rounds of max( cpc x h_r x 1.485 / (SM x 12.8) / 2.5e9 x 1e3 ,
c_r x 5,800 / 2.5e9 x 1e3 ) ms.

**CP-ARK** (F216 (b), F151). Queries Q = `default_rng(7).choice(N, 10000, replace=False)` (the 10,000 query points of the paper's protocol), appended after the N points; radius R_A = the smallest r0 x 2^i such
that 99 percent of the 10,000 queries have 50 points within r (L2 on columns 1 and 2). Probe build of the float device build (`-DNOCAND -DFLOATVAR=2`, KN 50, NORM 2, probes on), DEVOUT=1.
Reference launch (once per GPU session, before the first dataset): the same build on `ark2e5.txt` (100,000 points + 100,000 queries, radius 0.005, regenerated by `ttu/bench/Base_1/gen_inputs.py`), 3 runs, T_ref = median event ms
(`KERNEL queries`), F_ref = median f0_cycles. Sample run: the points and the first 1,000 queries of Q, radius R_A, 3 runs, F_s = median f0_cycles, F_d = F_s x 10.
A = T_ref x F_d / F_ref. B = (largest value over warps of the largest per lane count, count = 1.10 x exact cube count of radius R_A, lanes in Q order, 32 per warp) x 21,000 / 2.5e9 x 1e3 x 0.09 ms. Prediction = max(A, B).

**Relative.** R1 = predicted pinned over device = (P + 0.33 x MB) / P, P = the Arkade primary prediction (ms), MB = 10,000 x 50 x 8 / 1e6 = 4.0. Measured: the same probe build with the recorder off (`CB_OFF=1`), device
output against stock pinned output, medians of 3. R2 = 4.02; measured with the original and the register heap PTX (`notes/q15/diffs/knn_anyhit_reg.diff` applied to the same source), the same host binary, K 16, start radius r0, queries = points, 3 runs each,
under `ncu --clock-control none --kernel-name regex:optixLaunch --metrics gpu__time_duration.sum` (TMPDIR set to a private directory). Revision 2.1 (2026-11-24, after the dry run, before any new data): the first text measured the program's
`Total Search Time`, which includes host work and gave 1.29 on the dry run; F138's 4.02 is a launch time ratio, so the measured quantity is the launch time. The sum is over all launches including the small last rounds (F138 summed rounds 1 to 4).

## Order of events (each step is a commit and a push before the next step)

1. `blind_primary.py calibrate`: runs the sample procedures and the BVH dumps (item 4 of the allowed list), writes `model/blind/NAME/calibration_<mapping>_<ts>.json` (raw sample runs, clocks, hashes). Commit and push.
2. `blind_predict.py` (secondary, uncalibrated) and `blind_primary.py predict`: write `predictions_*` and `primary_predictions_*` (including the relative predictions), data hash, rule hash, tool hash. Commit and push.
3. `blind_primary.py measure` and `blind_measure.py`: refuse unless the calibration and prediction files are committed and pushed and the data hash matches. Full runs. Commit and push.
4. `blind_primary.py score`: CPU only. Commit and push.

## Scoring

Per row: ratio = measured / predicted (relative rows: measured speedup / predicted speedup). Reported per version and per mapping: rows inside 0.5 to 2, rms of the natural log ratio (and its exponential),
median absolute log ratio (the F236 figure), worst ratio, missing rows counted as missing. While `model/blind/PASS.md` has STATUS: UNSET the scorer prints no verdict. The pass mark is Jack's and must be committed
before the first calibration file exists.

## Dry runs

A run of the whole chain on a dataset that was measured before (everything we have today) is a DRY RUN: files go to `model/blind/dry_run_NAME/`, every file says DRY RUN, never reported as a blind result.
A dry run tests the tools and the procedures, not the model.

## Revision 2.2 (2026-11-24, after the dry run, before any new dataset): RTNN mode and sample size

The dry run (`model/blind/dry_run_u100000`, files `diag_sample_size_knn_*` and `diag_sample_size_radius_*`) showed two errors in version 2.0 of this rule, both on the known dry run dataset only:
1. The stub build only stubs the radius mode result write (`notes/rebuild/patches/rtnn_stub_radius_tethys.diff`): with `-sm knn` the real and the stub build differ by less than 1.5 percent at every sample size
   including all 100,000 queries, so the callback cost came out as zero. F422 and F424 used radius mode. The RTNN rows (calibration and full measurement, primary and secondary) are therefore radius mode.
2. A 2 percent query sample (2,000 queries) is a launch of about 0.31 ms against a fixed floor of the same size: the real minus stub difference was 1.5 percent of the launch (run to run spread 1.2 percent).
   Difference over real, radius mode, K 50, r 0.064, medians of 5: 2,000 queries 1.5 percent, 10,000 1.7, 25,000 4.0, 50,000 20.7 (spread 2.9), 100,000 21.7 (spread 3.8). The sample is now at least 50,000 queries (all queries if N is
   smaller): the smallest size in that set where the difference is several times the spread. The seed (11) is unchanged. This is chosen from launch sizes, not from any prediction score.
Known limit, not changed (it is the frozen definition of variant S): cb is divided by 1.10 x the mean cube count, not by the candidates that are actually called, so when the sphere holds more than K points the cost per candidate is
under estimated by about C / Ce (here 192 / 93, a factor 2). The dry run's cb is 2.6 to 3.5 cycles per candidate against F424's 9.4 at r 0.05, for that reason and because the radius differs.

## Revision 2.3 (2026-11-25, before any new dataset): variant S denominator, BVH dump allowed, reordering row R3 restored

1. Variant S now divides the callback cost by the called candidates (FROZEN.md, "Revision 1 of variant S"). The dry run had shown a factor 2 under estimate of the cost per candidate from dividing by the cube count.
2. Building and dumping the BVH on the new dataset is allowed before the predictions, without timing (allowed list, item 4). RTNN cannot dump without launching, so the process is killed as soon as the query file is complete and everything it prints is discarded.
3. Row R3 (RTNN unsorted over sorted) is restored, predicted from the two dumps. The measured quantity is the program's `KERNEL queries` event time, `-qs 0 -ps 0` over the defaults, both with `-p 0 -s 0 -m 0 -d 0` in radius mode, medians of 3.
Constants of R3 are frozen in FROZEN.md (F123 and F127 on the 4070 Ti); R3 is therefore also a test of whether that fit transfers to the 4090.

