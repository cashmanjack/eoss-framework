# Mochi count gap diagnostic (F166): true primitives per leaf slot against box matching

2026-11-16. Question: is the under prediction of intersection calls from frame 5 (results_mochi.md) caused by the replay's box matching (primitives put into leaf slots by the smallest containing slot box)? Test: the slot type byte of a custom primitive leaf slot gives the primitives in that slot in its low 3 bits (0x41 = 1, 0x42 = 2; checked on frames 0, 5, 9 of run B: 92,466 slots of 1 and 3,767 of 2, sum exactly 100,000 = the particle count, and the slot types and the tree topology are identical in every frame, only the boxes are refit). The replay table was rebuilt with these counts (model/mochi_leafcount.py) and the same rays and boxes; run B's own dumps and counts (no new GPU run).

| frame | measured calls / query | box matched replay | true count replay | box matched error | true count error | per query corr (true) | frac queries true < measured | frac true > measured | mean nodes (true) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 1.075 | 1.082 | 1.082 | +0.6% | +0.6% | 1.000 | 0.000 | 0.000 | 6.3 |
| 1 | 1.075 | 1.082 | 1.082 | +0.6% | +0.6% | 1.000 | 0.000 | 0.000 | 6.3 |
| 2 | 1.075 | 1.082 | 1.082 | +0.6% | +0.6% | 1.000 | 0.000 | 0.000 | 6.3 |
| 3 | 1.075 | 1.082 | 1.082 | +0.6% | +0.6% | 1.000 | 0.000 | 0.000 | 6.3 |
| 4 | 1.075 | 1.082 | 1.082 | +0.6% | +0.6% | 1.000 | 0.000 | 0.000 | 6.3 |
| 5 | 2.651 | 2.422 | 2.614 | -8.6% | -1.4% | 0.986 | 0.046 | 0.000 | 13.5 |
| 6 | 3.713 | 3.107 | 3.619 | -16.3% | -2.5% | 0.985 | 0.096 | 0.000 | 49.9 |
| 7 | 5.764 | 4.210 | 5.598 | -27.0% | -2.9% | 0.986 | 0.155 | 0.000 | 113.0 |
| 8 | 8.464 | 5.280 | 8.231 | -37.6% | -2.8% | 0.990 | 0.207 | 0.000 | 187.0 |
| 9 | 12.941 | 6.427 | 12.612 | -50.3% | -2.5% | 0.994 | 0.275 | 0.000 | 253.8 |

Result: the box matching IS the cause. With the true count per slot the mean calls per query are within 1.4 to 2.9 percent of the measurement in frames 5 to 9 (box matched: -8.6 to -50 percent), per query correlation 0.985 to 0.994 (frame 0 1.000), the replay never exceeds the measured count by more than half a call for any query (0 percent of queries), and 4.6 to 27.5 percent of the queries are slightly higher in hardware than in the replay (residual: the hardware enters somewhat more leaf slots than the exact slab test of the replay, or rounds boxes outward; not separated). The success rule set before (5 percent) is met in every frame. Node counts per query do not depend on the slot assignment (6.3 to 253.8, same as before). Reason for the failure of box matching: refit boxes of sibling leaf slots overlap more and more as the particles pile up, so the smallest containing slot is often not the slot the primitive lives in; at frames 0 to 4 (a lattice) it is unambiguous.

Lane efficiency with the true counts (rule 0.05, set before the run; the box matched version failed at frames 7 and 8):
- frame 5 measured 0.446 true count replay 0.444 diff -0.002
- frame 6 measured 0.460 true count replay 0.455 diff -0.004
- frame 7 measured 0.517 true count replay 0.511 diff -0.006
- frame 8 measured 0.536 true count replay 0.532 diff -0.005
- frame 9 measured 0.544 true count replay 0.539 diff -0.004
All within 0.006: the lane efficiency rule is met in every frame with the true counts. The earlier lane efficiency verdict in results_mochi.md stays as it was reported (it describes the box matched replay).
