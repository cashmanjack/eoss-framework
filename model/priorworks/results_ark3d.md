# Arkade Glove 3D in real 3D against the 2D build (2026-11-19, triton GPU 1, session 20261119T101720)

Queries = the 10,000 appended rows of ark_glove3d.txt; replay = Chebyshev box count of the 10,000 queries on the N = 1193514 points, x 1.10 (F207 convention); recorder = counters / 2 (two launches).

## Launch time (CUDA event, fixed build, no probes)

| build | runs ms | median ms | F207 prediction (model B) | measured / predicted |
|---|---|---|---|---|
| 2D (F207 build) | 848.2, 834.7, 848.0 | 848.0 | 110.98 | 7.64 |
| 3D (z read from the file) | 397.9, 382.1, 396.3 | 396.3 | 110.98 | 3.57 |

3D / 2D launch time: 0.467 (A3 prediction 0.45 to 0.65)

## Counts (recorder from the probe build, 2 runs each)

| build | run | recorder mean | 3D replay x1.1 mean | recorder / 3D replay | 2D replay x1.1 mean | recorder / 2D replay | Spearman vs 3D | Spearman vs 2D | recorder mean / exact 3D count (no 1.10) |
|---|---|---|---|---|---|---|---|---|---|
| rec2d | 1 | 113154.2 | 44172.6 | 2.562 | 121899.2 | 0.928 | 0.7940 | 1.0000 | 2.818 |
| rec2d | 2 | 113154.2 | 44172.6 | 2.562 | 121899.2 | 0.928 | 0.7940 | 1.0000 | 2.818 |
| rec3d | 1 | 42897.4 | 44172.6 | 0.971 | 121899.2 | 0.352 | 1.0000 | 0.7947 | 1.068 |
| rec3d | 2 | 42897.4 | 44172.6 | 0.971 | 121899.2 | 0.352 | 1.0000 | 0.7947 | 1.068 |

## Slowest warp (probe runs; definitions of F220: calls per step = warp calls / its largest lane count; life cycles per call over the model's 21,000 x 0.09; its largest lane count / the replay's largest warp count)

| build | run | probe launch ms | slowest warp life ms | slowest / launch | warp calls / its max lane count | life cycles per call / model | its max lane count / replay max (3D) | product | callback share | sms used | callback cycles per call |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rec2d | 1 | 977.2 | 977.2 | 1.00 | 2.03 | 2.60 | 1.67 | 8.80 | 0.80 | 78 | 3981 |
| rec2d | 2 | 970.4 | 970.4 | 1.00 | 2.03 | 2.58 | 1.67 | 8.74 | 0.80 | 78 | 3985 |
| rec3d | 1 | 456.4 | 456.3 | 1.00 | 1.86 | 2.36 | 0.94 | 4.11 | 0.76 | 78 | 3467 |
| rec3d | 2 | 463.0 | 462.9 | 1.00 | 1.83 | 2.41 | 0.94 | 4.17 | 0.76 | 78 | 3466 |

Lane statistics: per warp max count, replay 3D: mean over warps 128637, max 146802; replay 2D: mean 255917, max 266054; recorder 3D run 1: mean 123556, max 140578.
