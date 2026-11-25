# DRY RUN: PRIMARY (calibrated v1-S) predictions for u100000_v22, 20261124T220112, before any full run

**DRY RUN: the dataset was measured before. Tests the tools and procedures, not the model. Not a blind result.**

Data u100000.csv (100000 points, sha256 prefix 847eacd4769b), SM 128, clock 2.5e9, calibration file model/blind/dry_run_u100000_v22/calibration_20261124T220018.json (commit 2378059), rule hash b24b6a256cb4.

- RTNN K 50: primary 0.266338 ms (d_S 2.34514, radius 0.0640416); secondary uncalibrated 0.191255 ms
- TrueKNN k 16: primary 1.53578 ms over 5 rounds
- Arkade: primary 0.291225 ms (A 0.291225, B 0.135551), radius 0.0170893
- Relative R1 (Arkade pinned over device kernel time): 5.53257; R2 (kNN register heap over original): 4.02
