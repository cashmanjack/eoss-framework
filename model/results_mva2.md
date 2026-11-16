# MVA with traversal unit and hand off stations against measurement (no refit; predictions committed first in model/predictions_mva.md)
Error = log(predicted / measured). L = loaded callback latency per warp call (probe), T = launch time (kNN: probes off, CUDA events; Arkade: global timer span).

| launch | L measured | L issue only | L full | err issue only | err full | T measured ms | T issue only | T full | err issue only | err full |
|---|---|---|---|---|---|---|---|---|---|---|
| knn reg round 1 | 8637 | 4892 | 4880 | -0.57 | -0.57 | 0.165 | 0.077 | 0.078 | -0.76 | -0.75 |
| knn reg round 2 | 5868 | 3835 | 3818 | -0.43 | -0.43 | 0.225 | 0.131 | 0.132 | -0.54 | -0.53 |
| knn reg round 3 | 5994 | 4286 | 4195 | -0.34 | -0.36 | 0.692 | 0.468 | 0.475 | -0.39 | -0.38 |
| knn reg round 4 | 4995 | 3765 | 3399 | -0.28 | -0.38 | 2.937 | 2.175 | 2.373 | -0.30 | -0.21 |
| knn orig round 1 | 63746 | 31023 | 31008 | -0.72 | -0.72 | 0.661 | 0.255 | 0.255 | -0.95 | -0.95 |
| knn orig round 2 | 42599 | 20809 | 20791 | -0.72 | -0.72 | 0.979 | 0.424 | 0.424 | -0.84 | -0.84 |
| knn orig round 3 | 42544 | 20422 | 20393 | -0.73 | -0.74 | 3.502 | 1.593 | 1.593 | -0.79 | -0.79 |
| knn orig round 4 | 24085 | 12743 | 12656 | -0.64 | -0.64 | 11.016 | 5.658 | 5.666 | -0.67 | -0.66 |
| arkade device buffer, io (60 SMs) | 86859 | 75983 | 75974 | -0.13 | -0.13 | 3.700 | 3.764 | 3.764 | +0.02 | +0.02 |
| arkade device buffer, triton (142 SMs) | 84000 | 75983 | 75974 | -0.10 | -0.10 | 1.963 | 1.590 | 1.590 | -0.21 | -0.21 |

rms log error by kind: | kind | n | L issue only | L full | T issue only | T full |
|---|---|---|---|---|---|
| knn_reg | 4 | 0.42 | 0.44 | 0.53 | 0.51 |
| knn_orig | 4 | 0.70 | 0.71 | 0.82 | 0.82 |
| ark | 2 | 0.12 | 0.12 | 0.15 | 0.15 |

Speedup of the register heap over the original (launch time ratio), rounds 1 to 4:
| round | measured | issue only | full |
|---|---|---|---|
| 1 | 4.01 | 3.30 | 3.28 |
| 2 | 4.34 | 3.24 | 3.21 |
| 3 | 5.06 | 3.40 | 3.35 |
| 4 | 3.75 | 2.60 | 2.39 |
