# TrueKNN predictions on triton GPU 1 (142 SMs), written BEFORE the 200,000 and 400,000 point runs (2026-11-16)

Program: OWLRayTracing s01-trueknn, patched copy (frame buffer in device memory, per round probes of Finding 133 type, global timer span of each launch). Data: uniform random 3D points in the unit cube, k = 16, start radius 0.004, doubling, one launch per round, refit between rounds.

Constants from the calibration set A (100,000 points, medians of 5 runs): hardware warp calls / exact warp max box hits rho = 1.031, callback cycles per warp call c = 14815, total warp cycles / callback cycles g = 1.287, warps in flight per SM W = 12.03.

Calibration check on A (these constants were taken from A, so only the SHAPE per round is a test): the rule predicts active queries per round and the time per round pooled-constant.

| A round | radius | active (predicted) | active (measured) | warp calls (measured) | launch ms (measured) | warp calls (predicted) | launch ms (predicted) |
|---|---|---|---|---|---|---|---|
| 1 | 0.004 | 100000 | 100000 | 7436 | 0.1311 | 5879 | 0.0263 |
| 2 | 0.008 | 100000 | 100000 | 12202 | 0.0655 | 10132 | 0.0452 |
| 3 | 0.016 | 100000 | 100000 | 33056 | 0.2662 | 27269 | 0.1218 |
| 4 | 0.032 | 100000 | 100000 | 155530 | 1.3844 | 123298 | 0.5505 |
| 5 | 0.064 | 6111 | 5290 | 357748 | 0.6799 | 399394 | 1.7833 |
| total | | | | | 2.5271 | | 2.5271 |

## 200000 points (held out)

| B round | radius | active (predicted) | warp calls (predicted) | launch ms (predicted) |
|---|---|---|---|---|
| 1 | 0.004 | 200000 | 13604 | 0.0607 |
| 2 | 0.008 | 200000 | 26504 | 0.1183 |
| 3 | 0.016 | 200000 | 85528 | 0.3819 |
| 4 | 0.032 | 199790 | 439274 | 1.9614 |
| 5 | 0.064 | 164 | 22550 | 0.1007 |
| total | | | | 2.6230 |

## 400000 points (held out)

| C round | radius | active (predicted) | warp calls (predicted) | launch ms (predicted) |
|---|---|---|---|---|
| 1 | 0.004 | 400000 | 31917 | 0.1425 |
| 2 | 0.008 | 400000 | 74034 | 0.3306 |
| 3 | 0.016 | 400000 | 284399 | 1.2699 |
| 4 | 0.032 | 307525 | 1589427 | 7.0969 |
| total | | | | 8.8398 |

What can fail: rounds where the per call latency differs from the pooled c (Finding 133 saw 4 to 5 times between rounds of kNN); the active rule (hardware candidates beyond the exact box hits could stop queries earlier, Finding 132); the warp max model for lane efficiency. Success rule (set before): total launch time within a factor 1.5 and the number of rounds equal; per round error reported as log(predicted / measured). No refitting after the comparison.
