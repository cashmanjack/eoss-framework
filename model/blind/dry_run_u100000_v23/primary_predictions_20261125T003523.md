# DRY RUN: PRIMARY (calibrated v1-S) predictions for u100000_v23, 20261125T003523, before any full run

**DRY RUN: the dataset was measured before. Tests the tools and procedures, not the model. Not a blind result.**

Data u100000.csv (100000 points, sha256 prefix 847eacd4769b), SM 128, clock 2.5e9, calibration file model/blind/dry_run_u100000_v23/calibration_20261125T003418.json (commit 3c417c0), rule hash fa2adf1713d6.

- RTNN K 50: primary 0.389587 ms (d_S 6.19467, radius 0.0640415525); secondary uncalibrated 0.191255 ms
- TrueKNN k 16: primary 1.5356 ms over 5 rounds
- Arkade: primary 0.291187 ms (A 0.291187, B 0.135551), radius 0.0170893
- Relative R1 (Arkade pinned over device kernel time): 5.53317; R2 (kNN register heap over original): 4.02; R3 (RTNN unsorted over sorted kernel time) by K: {'50': 1.29406}
