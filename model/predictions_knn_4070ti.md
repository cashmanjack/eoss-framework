# kNN (growing radius) predictions, rand order, 200,000 points = queries, K 16, initial radius 0.004, doubling; written before measuring

fixed part fit on the 12 RTNN configurations: c0 110, cq 4.9 cycles; per point set fits: c0 120 cq 5.9, c0 93 cq 4.2

| round | active | radius | nodes / query | leaf slots / query | prims in slots / query | lane efficiency (replay) | share of fetches beyond 480 lines | predicted launch ms (all fit; range of the two fits) |
|---|---|---|---|---|---|---|---|---|
| 1 | 200000 | 0.004 | 6.16 | 1.09 | 1.134 | 0.514 | 0.712 | 0.187 (0.162 to 0.204) |
| 2 | 200000 | 0.008 | 7.35 | 1.75 | 1.919 | 0.427 | 0.749 | 0.203 (0.178 to 0.221) |
| 3 | 200000 | 0.016 | 11.13 | 7.05 | 8.145 | 0.553 | 0.829 | 0.284 (0.252 to 0.309) |
| 4 | 199998 | 0.032 | 23.73 | 44.50 | 56.026 | 0.744 | 0.896 | 0.734 (0.658 to 0.822) |
| 5 | 5411 | 0.064 | 40.36 | 173.62 | 248.239 | 0.539 | 0.906 | 0.063 (0.056 to 0.072) |

Total predicted search launch time over the rounds: 1.47 ms (sum of the launches only; host work between rounds, BVH builds and copies excluded).
What can fail: candidates per query beyond 1 x 1.006 (the hardware count is for entered leaf slots, here many slots hold 1 point and queries rarely fall in any box in the first rounds, so the fixed cost dominates: the c0 range 50 to 420 cycles is the whole uncertainty of the early rounds); lane efficiency within 0.1; the active counts per round (CPU exact k-th neighbour distance, the program's own count would follow after the heap reset I patched in).

# kNN (growing radius) predictions, morton order, 200,000 points = queries, K 16, initial radius 0.004, doubling; written before measuring

fixed part fit on the 12 RTNN configurations: c0 110, cq 4.9 cycles; per point set fits: c0 120 cq 5.9, c0 93 cq 4.2

| round | active | radius | nodes / query | leaf slots / query | prims in slots / query | lane efficiency (replay) | share of fetches beyond 480 lines | predicted launch ms (all fit; range of the two fits) |
|---|---|---|---|---|---|---|---|---|
| 1 | 200000 | 0.004 | 6.16 | 1.09 | 1.134 | 0.545 | 0.482 | 0.162 (0.137 to 0.178) |
| 2 | 200000 | 0.008 | 7.35 | 1.75 | 1.919 | 0.453 | 0.378 | 0.168 (0.143 to 0.186) |
| 3 | 200000 | 0.016 | 11.13 | 7.05 | 8.145 | 0.588 | 0.249 | 0.213 (0.182 to 0.239) |
| 4 | 199998 | 0.032 | 23.73 | 44.50 | 56.026 | 0.786 | 0.135 | 0.545 (0.469 to 0.633) |
| 5 | 5411 | 0.064 | 40.36 | 173.62 | 248.239 | 0.592 | 0.235 | 0.051 (0.045 to 0.061) |

Total predicted search launch time over the rounds: 1.14 ms (sum of the launches only; host work between rounds, BVH builds and copies excluded).
What can fail: candidates per query beyond 1 x 1.006 (the hardware count is for entered leaf slots, here many slots hold 1 point and queries rarely fall in any box in the first rounds, so the fixed cost dominates: the c0 range 50 to 420 cycles is the whole uncertainty of the early rounds); lane efficiency within 0.1; the active counts per round (CPU exact k-th neighbour distance, the program's own count would follow after the heap reset I patched in).

