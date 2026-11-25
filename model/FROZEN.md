# Model v1, frozen for the blind test (written 2026-11-24, tethys, before any new data exists)

This file records exactly which model is under test. Nothing listed here may change before the blind test is scored. If a file below changes, the
prediction script refuses to run (it checks the hashes in the block at the end of this file), and the test is void until a new freeze is committed and
announced as a different model version.

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
- Not part of the blind test: TrueKNN and Arkade. Their v1 callback costs are measured on the same input (probe warp cycles, a reference launch scaled by
  measured cycles), which the rule forbids on new data, and the count only forms that exist for them (F158, F207) are earlier forms, not v1. Hubs (CLOVER)
  and the winner call are not predicted by v1. The hardware BVH replay with real GAS bytes (`model/replay`) needs a GPU dump of the structure and is not part
  of the frozen form either: "BVH replay" here means the kd tree replay counts C and S that v1 uses.

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
| callback cost of the uniform reference, cbref | 1.0110154 cycles per candidate | `uni3_ref.csv` (uniform 3D, radius 0.0232, K 50) full minus stub runs, `notes/q34/raw/rtnn_uni3_ref_ref_*_20261117T161003_*`; cancels in a blind prediction |

Calibration data files behind these constants live in `notes/` (not in git); their hashes are in the private list below so a later change is visible.
The input `uni3_ref.csv` itself was on triton's /dev/shm and is not available on this machine; its hash was not recorded in the repository.

## Known behaviour before the test (recorded so the result cannot be spun afterwards)

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
