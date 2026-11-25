# DRY RUN: PRIMARY (calibrated v1-S) predictions for u100000, 20261124T214948, before any full run

**DRY RUN: the dataset was measured before. Tests the tools and procedures, not the model. Not a blind result.**

Data u100000.csv (100000 points, sha256 prefix 847eacd4769b), SM 128, clock 2.5e9, calibration file model/blind/dry_run_u100000/calibration_20261124T214845.json (commit 2c0d693), rule hash ffb629155f19.

- RTNN K 50: primary 0.271553 ms (d_S 2.508, radius 0.0640416); secondary uncalibrated 0.191255 ms
- TrueKNN k 16: primary 1.53542 ms over 5 rounds
- Arkade: primary 0.291369 ms (A 0.291369, B 0.135551), radius 0.0170893
- Relative R1 (Arkade pinned over device kernel time): 5.53034; R2 (kNN register heap over original): 4.02
