# MVA with issue, L1, L2, DRAM stations against measurement (no refit; predictions committed first in model/predictions_mva3.md)
Error = log(predicted / measured). L = loaded callback latency per warp call (probe), T = launch time (probes off, CUDA events).

| launch | L measured | split L | err | F149 L | err | T measured ms | split T | err | F149 T | err |
|---|---|---|---|---|---|---|---|---|---|---|
| knn reg round 1 | 8637 | 7352 | -0.16 | 4880 | -0.57 | 0.165 | 0.095 | -0.55 | 0.078 | -0.75 |
| knn reg round 2 | 5868 | 4173 | -0.34 | 3818 | -0.43 | 0.225 | 0.138 | -0.49 | 0.132 | -0.53 |
| knn reg round 3 | 5994 | 3516 | -0.53 | 4195 | -0.36 | 0.692 | 0.430 | -0.48 | 0.475 | -0.38 |
| knn reg round 4 | 4995 | 2847 | -0.56 | 3399 | -0.38 | 2.937 | 2.267 | -0.26 | 2.373 | -0.21 |
| knn orig round 1 | 63746 | 21025 | -1.11 | 31008 | -0.72 | 0.661 | 0.185 | -1.27 | 0.255 | -0.95 |
| knn orig round 2 | 42599 | 14146 | -1.10 | 20791 | -0.72 | 0.979 | 0.308 | -1.16 | 0.424 | -0.84 |
| knn orig round 3 | 42544 | 13232 | -1.17 | 20393 | -0.74 | 3.502 | 1.093 | -1.16 | 1.593 | -0.79 |
| knn orig round 4 | 24085 | 8132 | -1.09 | 12656 | -0.64 | 11.016 | 3.937 | -1.03 | 5.666 | -0.66 |

rms log error over the 8 kNN launches: split T 0.880 (kill rule: dropped if above 0.3), F149 form T 0.681; split L 0.846, F149 form L 0.589.
KILL RULE: FIRED, MVA dropped

Speedup of the register heap over the original (launch time ratio), rounds 1 to 4: | round | measured | split | F149 form |
|---|---|---|---|
| 1 | 4.01 | 1.94 | 3.28 |
| 2 | 4.34 | 2.22 | 3.21 |
| 3 | 5.06 | 2.54 | 3.35 |
| 4 | 3.75 | 1.74 | 2.39 |
