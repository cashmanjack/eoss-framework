# Held out scenes part 2: camera and bounce rays (2026-11-17, tethys RTX 4090 GPU 1; predictions model/predictions_scenes.md committed (6fa9f5f) before any ordered launch was timed; runner ttu/bench/Morton/scene_check.py, scorer model/score_scenes.py; raw: notes/hs2/measure_*_<timestamp>.json, private; OptiX only, no TTU code, no fault, GPU idle after)

Sponza, Conference, Rungholt; camera rays (4096 x 2048, pixel order = natural) and the bounce rays from their closest hits; 8,388,608 rays per set, 100 percent hits; seven orders against the natural (pixel) order; 10 launches per order, median; 0 wrong results in all 48 launch sets. F123 form (no refit), Memory_4 form of the predict step, and the baseline "predict 1.0" (no change). Speedup over the natural order:
| set | quantity | shuffled | morton | oracle d1 | oracle d2 | oracle d3 | oracle d4 | oracle full | rms log error |
|---|---|---|---|---|---|---|---|---|---|
| conference_camera | measured | 0.789 | 0.956 | 0.993 | 0.992 | 0.993 | 0.990 | 0.992 | (wrong 0, natural 0.672 ms) |
| | F123 | 0.987 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.086 |
| | Memory_4 | 0.748 | 1.048 | 1.071 | 1.071 | 1.071 | 1.071 | 1.071 | 0.076 |
| | predict 1.0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.091 |
| | F123 log error | +0.223 | +0.045 | +0.007 | +0.008 | +0.007 | +0.010 | +0.008 | |
| conference_bounce | measured | 0.975 | 1.009 | 1.064 | 1.070 | 1.076 | 1.079 | 1.080 | (wrong 0, natural 0.908 ms) |
| | F123 | 0.945 | 0.999 | 1.009 | 1.009 | 1.009 | 1.009 | 1.009 | 0.055 |
| | Memory_4 | 0.811 | 1.042 | 1.106 | 1.106 | 1.106 | 1.106 | 1.106 | 0.075 |
| | predict 1.0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.061 |
| | F123 log error | -0.031 | -0.010 | -0.053 | -0.059 | -0.065 | -0.067 | -0.069 | |
| sponza_camera | measured | 0.843 | 1.000 | 1.011 | 1.010 | 1.011 | 1.010 | 1.008 | (wrong 0, natural 0.752 ms) |
| | F123 | 0.964 | 1.000 | 0.999 | 0.999 | 0.999 | 0.999 | 0.999 | 0.051 |
| | Memory_4 | 0.699 | 1.027 | 0.987 | 0.987 | 0.987 | 0.987 | 0.987 | 0.074 |
| | predict 1.0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.065 |
| | F123 log error | +0.134 | -0.000 | -0.012 | -0.011 | -0.011 | -0.011 | -0.009 | |
| sponza_bounce | measured | 0.962 | 0.998 | 1.018 | 1.018 | 1.017 | 1.017 | 1.018 | (wrong 0, natural 0.882 ms) |
| | F123 | 0.947 | 1.046 | 1.048 | 1.048 | 1.048 | 1.048 | 1.048 | 0.031 |
| | Memory_4 | 0.901 | 1.152 | 1.162 | 1.162 | 1.162 | 1.162 | 1.162 | 0.127 |
| | predict 1.0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.021 |
| | F123 log error | -0.016 | +0.047 | +0.029 | +0.029 | +0.030 | +0.030 | +0.029 | |
| rungholt_camera | measured | 0.839 | 0.959 | 1.001 | 0.988 | 0.983 | 0.976 | 0.975 | (wrong 0, natural 0.728 ms) |
| | F123 | 0.954 | 0.999 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.053 |
| | Memory_4 | 0.666 | 0.960 | 1.014 | 1.014 | 1.014 | 1.012 | 1.011 | 0.091 |
| | predict 1.0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.070 |
| | F123 log error | +0.128 | +0.041 | -0.001 | +0.012 | +0.017 | +0.024 | +0.026 | |
| rungholt_bounce | measured | 0.808 | 1.027 | 1.050 | 1.092 | 1.173 | 1.204 | 1.243 | (wrong 0, natural 1.065 ms) |
| | F123 | 0.817 | 0.970 | 1.002 | 1.036 | 1.085 | 1.093 | 1.097 | 0.075 |
| | Memory_4 | 0.756 | 0.958 | 1.028 | 1.035 | 1.182 | 1.309 | 1.428 | 0.075 |
| | predict 1.0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.153 |
| | F123 log error | +0.010 | -0.058 | -0.047 | -0.053 | -0.078 | -0.097 | -0.125 | |
pooled rms log error over 42 points: F123 0.061  Memory_4 0.088  predict 1.0 0.087
pooled camera (21 points): F123 0.066  Memory_4 0.081  predict 1.0 0.076
pooled bounce (21 points): F123 0.057  Memory_4 0.096  predict 1.0 0.096

Reading (rules of the predictions file): every set passes the 0.12 rule for the F123 form (0.031 to 0.086). Against predict 1.0 on the set error: F123 is better on 5 of 6 sets but by little on three of them (Conference camera 0.086 against 0.091, Conference bounce 0.055 against 0.061, Sponza camera 0.051 against 0.065); it is WORSE on Sponza bounce (0.031 against 0.021: the measured gains of 1.00 to 1.02 are smaller than the predicted 1.05); clearly better only on Rungholt bounce (0.075 against 0.153, the one set where sorting pays: measured 1.03 to 1.24). Pooled over 42 points: F123 0.061, Memory_4 form 0.088, predict 1.0 0.087 (camera 0.066 / 0.081 / 0.076; bounce 0.057 / 0.096 / 0.096). So on these rays the F123 form beats no change by 30 percent pooled and the Memory_4 form does not beat it at all; but most of that comes from two kinds of points, the shuffled order and Rungholt bounce; on the 30 sorted order points of the five other sets the measured speedups are 0.96 to 1.08 and a constant 1.0 is nearly as good.
Where it is wrong: (1) shuffling a coherent camera order: measured slowdown 0.79 (Conference), 0.84 (Sponza), 0.84 (Rungholt) against F123 0.987, 0.964, 0.954 (log error +0.22, +0.13, +0.13): the form under predicts the cost of destroying coherence by 0.13 to 0.22, the largest errors of the experiment; the Memory_4 form (0.75, 0.70, 0.67) is closer only on Conference and over predicts the others. Bounce shuffles were right (0.98, 0.96, 0.81 measured; F123 0.945, 0.947, 0.817). (2) Rungholt bounce under predicts every deep oracle order, the error growing with depth (-0.047 at depth 1 to -0.125 at the full order, measured 1.24 against 1.10), the same sign as the random rays of F205 (-0.119 to -0.168) but smaller, and the order of the orders is right. (3) Morton of the direction for camera rays loses 4 percent (0.956, 1.000, 0.959) where the model says 1.00. (4) Rungholt camera deep oracle orders are 2 to 3 percent slower than pixel order (0.975), predicted 1.00.
Written expectations scored: (1) held in direction (F123 under predicts the shuffle cost on all three scenes) but the clause "the Memory_4 form is nearer" was wrong on Sponza and Rungholt; (2) mostly held (oracle orders 0.975 to 1.011), Morton 0.956 and 0.959 outside my 1.00 to 1.04; (3) Sponza and Conference within 0.12 held; Rungholt bounce under predicted held in direction, but its rms error 0.075 is below my 0.10 to 0.25; (4) pooled errors 0.061 and 0.087 are both below my 0.08 to 0.15 and 0.10 to 0.20, the order (model better than baseline) held.
Caveats: the speedup is the only quantity scored (not absolute time); camera and bounce rays are from one camera per scene and one bounce (no deeper bounces, no shadow rays); the oracle orders need the answer; one measured run per cell (10 launches, median); the camera position was picked automatically (view quantile 0.7) and not varied; parameters fitted on the 4070 Ti and applied to the 4090 through SM count and L2 size only; the pixel order of camera rays is already very coherent so the sorted orders have little to gain, which makes the camera sets weak tests of the gain side.
