# Prior works, variant (a): per warp slowest lane (post hoc recomputation, no refit)

Same ingredients as predictions_priorworks.md; only the count that enters the model changes. Units ms.

## RTNN

| dataset | r | mean form ms (F207) | warp max form ms |
|---|---|---|---|
| bunny | 0.00030682 | 0.123 | 0.123 |
| bunny | 0.0012273 | 0.191 | 0.214 |
| dragon | 0.15333 | 1.737 | 1.883 |
| dragon | 0.6133 | 5.297 | 5.763 |

## TrueKNN (fixed = device buffer)

| dataset | k | mean form ms | warp max form ms | rounds |
|---|---|---|---|---|
| road | 5 | 5.63 | 12.69 | 13 |
| road | 659 | 538.20 | 815.73 | 13 |
| porto1m | 5 | 11.78 | 34.25 | 20 |
| uni1m | 5 | 16.60 | 26.01 | 5 |

## Arkade (fixed = device buffer + float square; original pinned)

Calibration warp max sum 1985315 (mean form total 1202595).

| dataset | mean fixed | warp fixed | mean original pinned | warp original pinned | A mean | A warp | B |
|---|---|---|---|---|---|---|---|
| gowalla | 39.606 | 43.639 | 441.384 | 486.202 | 223.736 | 484.882 | 440.064 |
| glove3d | 110.982 | 114.879 | 1234.453 | 1277.755 | 723.602 | 1276.435 | 1233.133 |
| manuscript | 0.240 | 0.151 | 3.984 | 2.992 | 2.664 | 1.657 | 1.672 |

Success rule (fixed before): a row is closed if it is within a factor 2 of the F207 measurement; the variant helps a program if the median absolute log error over its rows falls. Not blind (see the header).
