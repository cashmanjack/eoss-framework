# Calibrated blind pipeline on the 24 F245 cells (Phase 13, triton GPU 1, 2026-11-18)

Predictions: predictions_phase13.md (commit 915e927), pipeline code commit 3899e81; each dataset went through blind_pipeline.py (commit order: predictions with the calibration numbers, then hubs and the call, then the full RTNN runs; the commits are in git, files in model/blind/f245_*). NOT blind: the F245 numbers are known. Ratio = measured RTNN search compute (median of 3) over predicted launch; inside 0.5 to 2 is a hit. 24 of 24 cells have complete data.

| dataset | K | measured ms | uncal | S 2% | D 2% | S 10% | D 10% | L (post hoc) | ratio uncal | ratio S | ratio D | ratio D 10% | ratio L | cal fixed / meas fixed | hubs | RTNN total | winner | call (S) | right |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3DRoad | 5 | 1.828 | 0.18 | 0.192 | 53.6 | 0.171 | 10.5 | 0.861 | 10.1 | 9.52 | 0.0341 | 0.174 | 2.12 | 41.9 / 33.4 | 34.2 | 35.3 | hubs | hubs | yes |
| 3DRoad | 50 | 10.602 | 0.545 | 0.499 | 267 | 0.478 | 58.5 | 11.5 | 19.5 | 21.2 | 0.0397 | 0.181 | 0.923 | 40.5 / 33.3 | 32.7 | 43.9 | hubs | hubs | yes |
| 3DRoad | 128 | 40.040 | 1.17 | 6.17 | 824 | 0.959 | 179 | 34.6 | 34.1 | 6.49 | 0.0486 | 0.223 | 1.16 | 40.5 / 34.2 | 36.0 | 73.1 | hubs | hubs | yes |
| Porto 1M | 5 | 42.494 | 0.412 | 0.391 | 564 | 0.391 | 127 | 28.1 | 103 | 109 | 0.0753 | 0.336 | 1.51 | 43.8 / 44.4 | 92.6 | 86.9 | RTNN | RTNN | yes |
| Porto 1M | 50 | 783.803 | 1.26 | 1.24 | 5.87e+03 | 1.07 | 1.62e+03 | 663 | 621 | 634 | 0.134 | 0.484 | 1.18 | 43.8 / 43.4 | 96.4 | 827.3 | hubs | RTNN | no |
| Porto 1M | 128 | 3914.580 | 2.72 | 2.34 | 1.43e+04 | 2.3 | 5.16e+03 | 3.1e+03 | 1.44e+03 | 1.67e+03 | 0.274 | 0.759 | 1.26 | 43.9 / 42.7 | 119.7 | 3957.2 | hubs | RTNN | no |
| uniform 1M | 5 | 1.212 | 0.462 | 0.431 | 6.36 | 0.44 | 6.16 | 6.12 | 2.62 | 2.81 | 0.191 | 0.197 | 0.198 | 43.3 / 35.7 | 89.5 | 36.9 | RTNN | RTNN | yes |
| uniform 1M | 50 | 9.218 | 1.75 | 1.53 | 47.7 | 1.45 | 18.5 | 11.9 | 5.28 | 6.01 | 0.193 | 0.499 | 0.774 | 43.2 / 35.7 | 102.5 | 44.9 | RTNN | RTNN | yes |
| uniform 1M | 128 | 41.102 | 3.96 | 3.21 | 211 | 3.21 | 85.7 | 57.5 | 10.4 | 12.8 | 0.195 | 0.48 | 0.715 | 43.6 / 35.8 | 136.7 | 77.0 | RTNN | RTNN | yes |
| bunny | 5 | 0.526 | 0.157 | 0.154 | 6.71 | 0.151 | 1.43 | 0.24 | 3.34 | 3.42 | 0.0784 | 0.368 | 2.19 | 40.6 / 33.4 | 29.4 | 33.9 | hubs | hubs | yes |
| bunny | 50 | 5.428 | 0.539 | 0.507 | 75.6 | 0.453 | 16.1 | 2.65 | 10.1 | 10.7 | 0.0718 | 0.338 | 2.05 | 40.3 / 33.3 | 30.6 | 38.8 | hubs | hubs | yes |
| bunny | 128 | 28.434 | 1.21 | 1.18 | 329 | 3.11 | 79.1 | 22.8 | 23.4 | 24.1 | 0.0863 | 0.359 | 1.25 | 40.3 / 33.7 | 32.8 | 61.7 | hubs | hubs | yes |
| dragon | 5 | 2.887 | 1.57 | 1.49 | 64.2 | 1.5 | 14.2 | 3.01 | 1.84 | 1.94 | 0.045 | 0.203 | 0.959 | 49.3 / 49.3 | 313.3 | 52.2 | RTNN | RTNN | yes |
| dragon | 50 | 26.180 | 5.32 | 4.72 | 119 | 4.61 | 47 | 30.8 | 4.92 | 5.55 | 0.22 | 0.557 | 0.85 | 48.6 / 49.2 | 392.3 | 75.4 | RTNN | RTNN | yes |
| dragon | 128 | 136.138 | 11.7 | 9.55 | 503 | 9.55 | 185 | 114 | 11.6 | 14.3 | 0.27 | 0.735 | 1.2 | 48.5 / 43.3 | 479.7 | 179.7 | RTNN | RTNN | yes |
| Gowalla | 5 | 210.065 | 0.548 | 0.705 | 2e+03 | 0.517 | 501 | 165 | 383 | 298 | 0.105 | 0.419 | 1.27 | 44.7 / 42.0 | 104.9 | 252.1 | hubs | RTNN | no |
| Gowalla | 50 | 781.174 | 1.73 | 2.5 | 6.05e+03 | 2.06 | 1.73e+03 | 770 | 451 | 312 | 0.129 | 0.451 | 1.01 | 44.7 / 42.3 | 112.3 | 823.4 | hubs | RTNN | no |
| Gowalla | 128 | 2398.700 | 3.97 | 11.4 | 1.05e+04 | 3.24 | 4.13e+03 | 2.7e+03 | 604 | 211 | 0.229 | 0.581 | 0.889 | 44.7 / 42.3 | 129.9 | 2441.0 | hubs | RTNN | no |
| Glove 3D | 5 | 1744.900 | 0.558 | 0.519 | 9.79e+03 | 0.958 | 2.45e+03 | 791 | 3.13e+03 | 3.36e+03 | 0.178 | 0.713 | 2.2 | 38.4 / 45.1 | 125.7 | 1789.3 | hubs | RTNN | no |
| Glove 3D | 50 | 4063.590 | 2.1 | 15.3 | 2.23e+04 | 2.86 | 5.76e+03 | 2.03e+03 | 1.94e+03 | 266 | 0.182 | 0.706 | 2 | 38.1 / 44.8 | 144.0 | 4108.4 | hubs | RTNN | no |
| Glove 3D | 128 | 8127.040 | 4.74 | 3.84 | 3.04e+04 | 7.77 | 1.01e+04 | 5.51e+03 | 1.71e+03 | 2.12e+03 | 0.268 | 0.807 | 1.48 | 38.2 / 44.1 | 204.0 | 8171.1 | hubs | RTNN | no |
| Manuscript | 5 | 76.380 | 0.862 | 0.822 | 255 | 1.28 | 122 | 91.9 | 88.6 | 92.9 | 0.3 | 0.627 | 0.831 | 45.9 / 43.5 | 163.8 | 119.7 | RTNN | RTNN | yes |
| Manuscript | 50 | 305.765 | 2.9 | 2.44 | 1.05e+03 | 2.44 | 467 | 335 | 105 | 125 | 0.29 | 0.655 | 0.913 | 47.9 / 45.1 | 204.6 | 350.9 | hubs | RTNN | no |
| Manuscript | 128 | 978.222 | 6.08 | 112 | 2.76e+03 | 24.2 | 1.47e+03 | 1.19e+03 | 161 | 8.7 | 0.355 | 0.663 | 0.825 | 47.4 / 44.1 | 244.5 | 1022.4 | hubs | RTNN | no |

Within a factor 2 (hits of cells, median absolute log error, median ratio measured / predicted):

- uncalibrated v1 (F250 form): 1 of 24 inside, median |log| 4.01 (factor 55.0), median ratio 61.36
- S, 2 percent (pre registered primary): 1 of 24 inside, median |log| 3.12 (factor 22.6), median ratio 22.66
- D, 2 percent: 0 of 24 inside, median |log| 1.71 (factor 5.6), median ratio 0.18
- S, 10 percent (diagnostic): 1 of 24 inside, median |log| 3.72 (factor 41.1), median ratio 41.09
- D, 10 percent (diagnostic): 10 of 24 inside, median |log| 0.73 (factor 2.1), median ratio 0.48
- L two point extrapolation (post hoc, not pre registered): 18 of 24 inside, median |log| 0.23 (factor 1.3), median ratio 1.17

Calibrated fixed part (query sort + point sort + geometry of the 2 percent run) against the measured query sort + geometry of the full run: within 1.5 times in 24 of 24 cells (ratio range 0.85 to 1.25).
S floor (real build not slower than the stub on the 2 percent subset, cb set to 0) binds in 8 of 24 cells.

Winner (RTNN with sort + geometry + search against hubs, measured in this run): RTNN wins 8 cells, hubs 16; always hubs is right in 16 of 24.
- call uncalibrated launch + calibrated fixed: right in 14 of 24 cells (wrong: Porto 1M K 50 called RTNN; Porto 1M K 128 called RTNN; Gowalla K 5 called RTNN; Gowalla K 50 called RTNN; Gowalla K 128 called RTNN; Glove 3D K 5 called RTNN; Glove 3D K 50 called RTNN; Glove 3D K 128 called RTNN; Manuscript K 50 called RTNN; Manuscript K 128 called RTNN)
- call S + calibrated fixed (the pipeline call): right in 14 of 24 cells (wrong: Porto 1M K 50 called RTNN; Porto 1M K 128 called RTNN; Gowalla K 5 called RTNN; Gowalla K 50 called RTNN; Gowalla K 128 called RTNN; Glove 3D K 5 called RTNN; Glove 3D K 50 called RTNN; Glove 3D K 128 called RTNN; Manuscript K 50 called RTNN; Manuscript K 128 called RTNN)
- call D 2 percent + calibrated fixed: right in 20 of 24 cells (wrong: Porto 1M K 5 called hubs; uniform 1M K 128 called hubs; dragon K 128 called hubs; Manuscript K 5 called hubs)
- call D 10 percent + calibrated fixed: right in 22 of 24 cells (wrong: Porto 1M K 5 called hubs; Manuscript K 5 called hubs)
- call L + calibrated fixed (post hoc): right in 24 of 24 cells (wrong: none)
- call A (S launch only, the old pipeline rule): right in 8 of 24 cells.
