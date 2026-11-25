# Model v1, frozen for the blind test (written 2026-11-24, revised 2026-11-24 to make the calibrated version primary; tethys; before any new data exists)

This file records exactly which model is under test. Nothing listed here may change before the blind test is scored. If a file below changes, the
prediction script refuses to run (it checks the hashes in the block at the end of this file), and the test is void until a new freeze is committed and
announced as a different model version.

## Versions under test (revision: the first version of this file made the uncalibrated form the only test, which tests what F245 already showed fails)

- **PRIMARY: model v1-S, the calibrated model of F422 and F424.** The v1 form below with the callback cost measured through one fixed procedure on a
  small sample of the new dataset (procedures and sample sizes in `model/blind/RULE.md`, fixed there before any new data). Mappings: RTNN, TrueKNN, Arkade.
  This is the version the paper's claim is about (F422: 15 of 15 rows and 7 of 7 headline rows inside a factor 2 on the 5090 with the callback calibrated on
  the same input; F424: 1.27 and 1.09 of the 4090 kernel; both calibrated on the same input, so not blind). The first blind use of it is this test.
- **SECONDARY: model v1 uncalibrated** (cb - cbref = 0), RTNN only, kept as a row so the known failure stays on record (see "Known behaviour").
- **RELATIVE rows** (configuration speedups with frozen predictors): Arkade pinned over device output buffer, kNN register heap over original heap.

## What is frozen

- Name: model v1, the combined form of Finding 236 (frozen 2026-11-17, form commit `eb22f98`, text in `model/priorworks/FORM_combined_v1.md`).
- Repository state when this file was written: `d46e93f` (main, 2026-11-24). The tracked model files below have not
  changed since their commits: `model_v1_rtnn.py` last changed in `5ced603` (2026-11-18), `FORM_combined_v1.md` in `eb22f98` (2026-11-17).
- Form: T(launch) = max(T_throughput, T_chain), per launch.
- The part used in the blind test is the RTNN search launch, as in `model/priorworks/model_v1_rtnn.py` (`predict`, unchanged):
  - T_throughput = (110 + (4.9 + cb - cbref) q_mean) N / (SM x CLK), q = 1.10 Ce + 1, Ce = K (C + 1) / (S + 1) if S > K else C.
  - T_chain = (largest q) x 575 / CLK.
  - C = points in the query's Chebyshev ball of radius r (self included), S = points in its Euclidean ball (self included), both exact on the CPU with a
    kd tree (`replay_counts`). The radius is the F245 rule: the smallest radius at which 99 percent of 3,000 sampled queries have K other points inside
    (`radius99` in `blind_pipeline.py`, seed 5). K is a parameter (v1 was written with K = 50; generalizing K is the only change ever made, F245).
  - cb - cbref = 0 in a blind prediction (see `model/blind/RULE.md`): the callback cost is the uniform reference's, so cbref cancels and the prediction
    uses only the constants below.
  - Output: milliseconds for the search launch only (RTNN's "time search compute"); no query sort, no geometry build, no result copy.
- RTNN primary: the same form with cb - cbref = d_S, where d_S = max(cbS - cbref, -4.9) and cbS = max(0, (T_real - T_stub) x SM x CLK x 1e-3 / (nq x 1.10 x mean cube count of the
  sampled queries)): T_real and T_stub are RTNN's "time search compute" (ms, median of 3) on the sample with the real build and the stub build (intersection program
  without distance check and without result write, the F216 definition, `notes/rebuild/patches/rtnn_stub_radius_tethys.diff`). This is variant S of Phase 13
  (`blind_pipeline.calibrate`, frozen hash below) and the definition of F422 and F424; the only change is that it is run on a sample, as Phase 13 did.
- TrueKNN primary (F216 (b), `score_model_v1.py`): per round B = f0_cycles x 1.485 / (SM x 12.8) / CLK x 1e3 ms (f0_cycles = callback cycles of the round from the recorder
  of the port), chain = (largest box count of an active query in the round) x 5,800 / CLK x 1e3 ms, launch time = sum over rounds of max(B, chain). Rounds, active sets and
  box counts come from the CPU replay (F158 stop rule: a query stops when its box count - 1 >= k, radius doubles from r0). For a new dataset f0_cycles of a round is
  not measured on the full run: it is cycles per replay box hit, measured on the sample run, times the replay box hits of the round (the one extension of the F216 form;
  it is stated in RULE.md before any data).
- Arkade primary (F216 (b), F151, `score_model_v1.py`): T = max(A, B), A = T_ref x F_d / F_ref (T_ref and F_ref: event time and callback cycles f0 of the reference launch,
  100,000 points and 100,000 queries uniform 2D, radius 0.005, device output, float distance, measured on the same GPU in the same session; F_d: callback cycles of the
  new dataset, measured on the sample queries and scaled by the query count), B = (largest per lane count of a warp, 1.10 x exact cube count) x 21,000 / CLK x 1e3 x 0.09
  (float distance). Output buffer: device (the fixed build with DEVOUT).
- Not predicted by v1: hubs (CLOVER) and the winner call. The hardware BVH replay with real GAS bytes (`model/replay`) needs a GPU dump of the structure and is not part
  of the frozen form: "replay" means the kd tree replay counts that v1 uses.

## Constants (every one with its source; none is refitted)

| constant | value | source |
|---|---|---|
| fixed cycles per query per SM | 110 | F127 refit on the 4070 Ti, 12 configurations (`notes/q12/rtnn_refit.txt`) |
| cycles per hardware candidate (cq) | 4.9 | same |
| hardware candidates over exact cube count | 1.10 | F125 (recorder builds, 4070 Ti) |
| ray ends after K sphere hits: Ce = K (C+1)/(S+1) when S > K | rule | F116 |
| chain latency per candidate, L | 575 cycles | F133 (unloaded call, sorted RTNN, one warp per SM) |
| clock constant CLK | 2.5e9 Hz, for every GPU | F236; F255 kept it for the 4090 |
| SM count | 142 (RTX 6000 Ada), 128 (RTX 4090), 60 (RTX 4070 Ti) | chip specification, the only quantity that changes with the GPU |
| TrueKNN callback to wall factors | 1.485 and 12.8 | F160 (`score_model_v1.py`) |
| TrueKNN chain latency per candidate | 5,800 cycles | F133 (kNN round 4, one warp per SM, k 16) |
| Arkade chain latency per call | 21,000 cycles | F133, F207 (double pow, unloaded) |
| Arkade float over double | 0.09 | F151 (0.18 ms against 1.97 ms, device output) |
| Arkade pinned output cost | 0.33 ms per MB of result buffer | F145, F151 (triton) |
| kNN register heap speedup | 4.02 | F138 (rounds 1 to 4 total; 4.45 and 4.34 to 4.48 on the 4090, F255, F424, F425) |
| callback cost of the uniform reference, cbref | 1.0110154 cycles per candidate | `uni3_ref.csv` (uniform 3D, radius 0.0232, K 50) full minus stub runs, `notes/q34/raw/rtnn_uni3_ref_ref_*_20261117T161003_*`; cancels in a blind prediction |

Calibration data files behind these constants live in `notes/` (not in git); their hashes are in the private list below so a later change is visible.
The input `uni3_ref.csv` itself was on triton's /dev/shm and is not available on this machine; its hash was not recorded in the repository.

## Known behaviour before the test (uncalibrated secondary version; the calibrated primary is known only on the same input)

Calibrated on the same input, not blind: F422 15 of 15 rows inside a factor 2 (5090), F424 within 1.09 to 1.27 of the 4090 kernel (both on uniform inputs, callback from the
real minus stub difference of the full launch); F216 (b) with measured callback costs on the same inputs scored 6 of 14 inside a factor 2 over RTNN, TrueKNN and Arkade (model v1,
F236, median factor 2.1; Arkade 1 of 6, all under predicted 1.7 to 3.4).

**The sample calibration of this test has already been scored once, and it failed.** Phase 13 (variant S, RTNN callback from real and stub runs on a random 2 percent of the queries,
F245's 24 cells, `model/priorworks/results_blind_cal.md`): measured / predicted 1.94 to 3,360, 1 of 24 cells inside a factor 2, rms of ln ratio 4.4, median |ln ratio| 3.1; the
uncalibrated form had 1.84 to 3,130 and 1 of 24, so on those datasets the sample callback calibration changes almost nothing: the misses are not (mainly) the callback cost
(non uniform inputs: Porto 1M, Gowalla, Glove 3D, Manuscript, 3DRoad). Variant D of the same table (callback fitted to make the subset launch time exact) over predicts by 3 to 30 times
(0 of 24). So the expected outcome of the primary RTNN row on non uniform data is a miss of the same kind, recorded before the test; uniform data is where the calibration was shown
to work (F422, F424).

Uncalibrated secondary version (recorded so the result cannot be spun afterwards):

- On the 24 F245 cells (8 datasets x K 5, 50, 128) this uncalibrated form under predicts every cell: measured / predicted ranges from 1.8 to 3,130,
  median 61, rms of the natural log ratio 4.7, and 1 of 24 cells is inside a factor 2 (`model/priorworks/results_blind_cal.md`, column "ratio uncal").
  The uniform inputs are closest (1.8 to 10), surface and clustered inputs are the far misses (Gowalla, Glove 3D, Porto 1M: 100 to 3,000 times).
- On the 4090 with synthetic uniform inputs the form transfers within a small factor only (F255); the constants were measured on other chips.
- So the expected outcome of the blind test is: predicted is a lower bound, measured above it, by a data dependent factor. A prediction inside a factor 2
  on a non uniform dataset would be a surprise, not the baseline.

## Hashes (checked by `model/blind_tools/blind_predict.py` before every prediction)

Tracked files, first 12 hex digits of the sha256 (the push scan refuses longer hex runs), taken at the repository state above. Lines in this exact form are read by the script.

```
37b91d86e0a4  model/priorworks/model_v1_rtnn.py
c18941b6fb3f  model/priorworks/FORM_combined_v1.md
37f5204d0555  model/priorworks/score_model_v1.py
cd12dd7900e2  model/priorworks/predict_critical_path.py
13a0a3ae162d  model/priorworks/predict_priorworks.py
976e7d8cee3b  model/priorworks/blind_pipeline.py
ae4a9fb9c53f  model/priorworks/results_model_v1.md
026d053c487b  model/priorworks/results_blind_cal.md
```

Calibration data files (private, in `notes/`, not tracked; recorded here for later comparison):

```
c00577e6736c  notes/q12/rtnn_refit.txt
6a2142b63739  notes/q34/raw/rtnn_uni3_ref_ref_full_20261117T161003_1.txt
b56b4450a9c0  notes/q34/raw/rtnn_uni3_ref_ref_full_20261117T161003_2.txt
7d4d062c8731  notes/q34/raw/rtnn_uni3_ref_ref_stub_20261117T161003_1.txt
51dfe0203b76  notes/q34/raw/rtnn_uni3_ref_ref_stub_20261117T161003_2.txt
```

(The private block uses the same line form in a second code block; the script reads only the first block and checks tracked paths.)

## Tools added after the freeze (not part of the model)

`model/blind_tools/` (prediction, measurement and scoring scripts) and `model/blind/RULE.md` are written after this file. They contain no model
constants of their own: the prediction script imports `model_v1_rtnn.py` and only passes SM through.
