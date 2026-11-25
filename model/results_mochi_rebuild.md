# Mochi_1 results: periodic BVH rebuild (tethys, RTX 4090 GPU 1, driver 580.178.04), 2026-11-25
Predictions and rule: predictions_mochi_rebuild.md (6f6c63a, amendment 4eba6f8, both before the runs of this table). Raw files (not overwritten): notes/mochi1/mochi_rebuild_time_20261125T094122.json (5 runs per arm, rounds interleaved, warmup process each, 40 of 40 runs at full clock), mochi_rebuild_check_20261125T094421.json, and a 1 run smoke file mochi_rebuild_time_20261125T093741.json (kept, seen before amendment 1).

Premise correction: stock Mochi already refits at every sub step (4,000 times) and never rebuilds; the "build" term of F453 is refit time. The change here is the reverse: a full rebuild (both groups plus owlBuildSBT) every N sub steps, refit in between. N counts sub steps globally; the first timed sub step is a rebuild for every N above 0.

## Time (ms, median of 5, 100,000 particles, 10 frames x 400 sub steps)
| arm | rebuilds | build term | collision | update | total | total / stock | spread of total |
|---|---|---|---|---|---|---|---|
| stock binary | 0 | 1,192 | 299 | 61 | 1,552 | 1.000 | 35 |
| patched N 0 | 0 | 1,187 | 299 | 61 | 1,549 | 0.998 | 11 |
| N 4000 | 1 | 1,183 | 299 | 61 | 1,543 | 0.995 | 26 |
| N 400 | 10 | 1,198 | 149 | 61 | 1,408 | 0.907 | 29 |
| N 100 | 40 | 1,208 | 146 | 61 | 1,414 | 0.911 | 71 |
| N 25 | 160 | 1,265 | 144 | 61 | 1,470 | 0.947 | 12 |
| N 10 | 400 | 1,389 | 146 | 62 | 1,596 | 1.029 | 68 |
| N 1 | 4,000 | 3,071 | 147 | 64 | 3,283 | 2.115 | 24 |
One extra rebuild costs about 0.47 ms over a refit (N 1 against stock: 1,879 ms over 4,000), so a rebuild with the SBT is about 1.6 times a refit pair (0.30 ms).

## Accuracy (every frame, 25 dumps per frame, 250 per interval, 1,750 in all; last sub step of every frame included)
Program count against the CPU pair count on the dumped state, exact equality required:
| N | dumps differing | largest difference | frames with a difference |
|---|---|---|---|
| 0 | 10 | 1 | 7, 8, 9 |
| 4000 | 7 | 1 | 7, 8, 9 |
| 400 | 9 | 1 | 7, 8, 9 |
| 100 | 6 | 2 | 7, 8, 9 |
| 25 | 9 | 1 | 8, 9 |
| 10 | 9 | 1 | 6, 7, 8, 9 |
| 1 | 12 | 2 | 7, 8, 9 |
Frames 0 to 5 are exact in every dump of all seven intervals. Differences are 1 or 2 collisions out of about 10^4 in the late, dense frames, at the same rate (6 to 12 of 250) for the arm that never rebuilds (N 0) as for N 1. So rebuilding does not change the answer beyond what the program already does without it. The cause of the 1 or 2 count differences is NOT examined (dumps were deleted after checking; a guess is a borderline pair decided differently by the GPU float test and the CPU float64 test). Earlier F452 checked frames 0, 5 and 9 once, and they were exact then.

## Predictions scored
P1 (all dumps exact): NOT held as written (62 of 1,750 off by 1 or 2, none that depends on N). P2 (collision 0.6 at N 1, 0.95 at N 400): refuted, 0.49 at every N from 400 to 1. P3 (build term): N 400, 100, 1 inside the ranges, N 25 (1,400 predicted, 1,265) and N 10 (1,800, 1,389) too high; a rebuild costs 1.6 refits, not 4. P4 (no gain of 5 percent): refuted, N 400 and N 100 give 0.91, N 25 0.95. P5 (patched N 0 equals stock): held, 0.998. P6 (one rebuild at the start gives the whole gain): refuted, N 4000 is 0.995.
Rule: N 400 (cheapest rebuilds) and N 100 win (0.907 and 0.911 of stock, with counts as in the accuracy table); N 25 is borderline (0.947); N 10 and N 1 lose. A periodic rule beats the single start rebuild (N 4000) by 9 percent of total.

## Open
Why a rebuild at the first timed step does not help (N 4000) while the same rebuild every 400 steps halves the collision time, and why N 1 gains nothing over N 400, is not explained; program timers are totals, no per frame split was taken, nothing was tested. Not done: other particle counts, per frame times, an adaptive rule, padded boxes, the cause of the 1 or 2 count differences.
