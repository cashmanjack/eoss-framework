# Prior works: critical path bound, predictions BEFORE scoring (2026-11-17, triton GPU 1)

T = max(throughput, slowest warp), per launch. Constants: RTNN L = 575, TrueKNN L = 5,800, Arkade L = 21,000 cycles per warp call (smallest unloaded values of F133; none fitted here). Units ms.

## RTNN

| dataset | r | F207 throughput | max lane q | slowest warp | prediction | chain binds? |
|---|---|---|---|---|---|---|
| bunny | 0.00030682 | 0.123 | 4.3 | 0.001 | 0.123 | no |
| bunny | 0.0012273 | 0.191 | 51.6 | 0.012 | 0.191 | no |
| dragon | 0.15333 | 1.737 | 74.7 | 0.017 | 1.737 | no |
| dragon | 0.6133 | 5.297 | 249.8 | 0.057 | 5.297 | no |

## TrueKNN (fixed = device buffer; per round max, summed)

| dataset | k | F207 throughput total | slowest warp per round (ms) | prediction total | rounds where chain binds |
|---|---|---|---|---|---|
| road | 5 | 5.63 | 0.030 0.046 0.063 0.116 0.130 0.269 0.378 0.626 0.689 1.341 0.682 0.007 18.140 | 26.34 | 6 of 13 |
| road | 659 | 538.20 | 0.030 0.067 0.086 0.130 0.262 0.397 0.965 1.821 4.236 8.340 14.152 24.228 25.383 | 560.75 | 1 of 13 |
| porto1m | 5 | 11.78 | 0.193 0.135 0.193 0.353 0.735 2.039 3.963 4.459 4.960 4.768 1.255 0.582 0.072 0.002 0.002 0.002 0.002 0.002 0.002 2308.249 | 2340.49 | 15 of 20 |
| uni1m | 5 | 16.60 | 0.009 0.016 0.037 0.116 0.401 | 17.00 | 1 of 5 |

## Arkade (event time of the launch; F207 already was max(A, B) with B the slowest warp x 21,000 cycles, so these rows equal F207 by construction)

| dataset | fixed | original pinned |
|---|---|---|
| gowalla | 39.606 | 441.384 |
| glove3d | 110.982 | 1234.453 |
| manuscript | 0.240 | 3.984 |

What can fail: L is a lower bound from small launches of other data (k = 16 for TrueKNN), so rows where one lane has many candidates may stay under; k = 659 has no k term; the chain ignores traversal latency (not in the count data).
