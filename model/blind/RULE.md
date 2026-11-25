# Blind test rule for model v1 (written 2026-11-24, before any new dataset exists)

Model under test: `model/FROZEN.md` (model v1, RTNN search launch). This file fixes what may and may not enter a blind prediction. It is committed before
the tools that follow it and before any new data are touched. Changing it after a blind prediction exists voids that prediction.

## The calibration rule

**The callback cost of a blind prediction comes only from inputs we already have. Nothing from the new datasets enters the model.**

Concretely:

1. The callback cost is the uniform reference's (cb = cbref, so cb - cbref = 0), exactly as `model_v1_rtnn.predict` is called with its default.
   The only constants are those in `model/FROZEN.md`.
2. Allowed inputs from a new dataset: the dataset file read on the CPU, and quantities computed from it on the CPU alone: the number of points N,
   the radius from the F245 rule (99 percent of 3,000 sampled queries have K other points inside), and the replay counts C and S (kd tree).
   K is chosen by us before looking at anything, from the list the test announces.
3. Allowed inputs about the machine: the SM count of the GPU from its specification (table in `model/FROZEN.md`). The clock constant is not changed.
4. Not allowed, in any form, before the predictions are committed: any run of RTNN (real build or stub build), TrueKNN, Arkade, kNN, CLOVER or any other
   program on the new dataset or on a subset or sample of it; any timing, counter (ncu), probe or recorder reading taken on it; any "calibration
   fraction" (the Phase 13 variants S and D, which measure the callback cost on 1 to 10 percent of the queries, are retired for blind use); any
   constant adjusted after looking at a measurement of the new data.
5. After the measurement, a fitted or recalibrated variant may be computed, but it is labelled POST HOC and is never scored as a blind prediction.
6. A dataset is "new" only if no program of ours has been run on it before. The datasets already measured, which can only be used for dry runs and are
   never blind: the F245 set (3DRoad, Porto 1M, uniform 1M, bunny, dragon, Gowalla, Glove 3D, Manuscript), the synthetic uniform files
   (u100000, u200000, u400000, uni3_ref, ark_a1e5, ark2e5) and the Arkade/TrueKNN/Mochi/BarnesHut inputs of the 2026-11-24 baselines.

## Order of events (each step written to disk and committed before the next one; the commit history is the proof)

1. Prediction (`model/blind_tools/blind_predict.py`): checks the frozen hashes, reads the dataset, runs the CPU replay and the model, writes
   `model/blind/NAME/predictions_<ts>.{json,md}` (data hash, N, K, radius, SM, constants, predicted ms, code versions), commits and pushes. No GPU is used.
2. Measurement (`model/blind_tools/blind_measure.py`): refuses to start unless the prediction file is committed and unchanged and the data hash matches.
   Idle GPU check by nvidia-smi PIDs and utilization (never by process name), a warmup process before each timed run, three runs, the SM clock recorded
   per run, `timeout`. Measures RTNN's "time search compute" (the quantity v1 predicts), flags `-sm knn -p 0 -s 0 -m 0 -d 0`, the radius from the prediction
   file. Writes `measured_<ts>.json` (never overwritten).
3. Scoring (`model/blind_tools/blind_score.py`, CPU only): reads the two files, writes `scores_<ts>.{json,md}`.

## Scoring (fixed now)

Per row (dataset, K): ratio = measured / predicted. A row is inside the factor when 0.5 <= ratio <= 2. Reported: rows inside the factor, the rms of the
natural log ratio over all rows (and exp of it as a factor), the median absolute natural log ratio (the figure F236 used: 0.74 on its 14 non blind rows),
and the worst ratio. No pass mark is set here: the numbers are reported as they come and compared with `model/FROZEN.md` "known behaviour". A missing or
failed measurement counts as a missing row, never as a pass.

## Dry runs

A run of the whole chain on an already measured dataset is a DRY RUN. Its files go to `model/blind/dry_run_NAME/`, carry "DRY RUN" in every file, and are
never reported as blind results. A dry run tests the tools, not the model.

## What this rule does not do

It does not make the model good: the frozen form is a lower bound that missed every non uniform cell of F245 (see `model/FROZEN.md`). It makes the test
honest: if the prediction is wrong, the log shows that nothing from the new data was used to make it.
