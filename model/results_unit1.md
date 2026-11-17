# Unit_1 results (2026-11-17, tethys GPU 1, RTX 4090; predictions model/predictions_unit1.md, committed first)

Runs: stage 0 (no TTU) on par:4096:s and par:1024:o with 32 warps, one smallest TTU launch (tri1, 1 warp, 1 launch), then 22 stage 1 runs (11 scenes x 1 warp and 32 warps, 30 launches each, first dropped). All exit 0, exactly one block each, no fault, GPU idle after. The sequence is the checked remap build of Throughput_1 (no new TTU sequence). Raw JSON notes/unit1 (private). Runner ttu/bench/Unit_1/run.py.
c = launch span / (32 warps x 32 lanes), cycles per ray per SM (one SM, every lane the same ray, hits nothing). The "gap between warp ends" statistic of Throughput_1 is not usable for long queries (warps then end within 2 to 9 cycles of each other, all progress together): not used. L1w = one warp latency.

| scene | nodes / slots / tris | c measured | H_L | H_T | L1w | F67 line | L1w / c |
|---|---|---|---|---|---|---|---|
| par:1:s | 1 / 1 / 1 | 8.8 | 11 | 8 | 1,587 | 1,442 | 181 |
| par:4:s | 1 / 4 / 4 | 10.7 | 14 | 10 | 2,076 | 1,937 | 193 |
| par:16:s | 1 / 12 / 16 | 18.7 | 25 | 19 | 3,571 | 3,465 | 190 |
| par:64:s | 3 / 33 / 64 | 56.0 | 64 | 56 | 8,747 | 9,020 | 156 |
| par:256:s | 9 / 100 / 256 | 192.8 | 202 | 202 | 28,655 | 28,633 | 149 |
| par:1024:s | 33 / 356 / 1,024 | 761 | 745 | 785 | 100,559 | 105,729 | 132 |
| par:4096:s | 140 / 1,399 / 4,096 | 3,122 | 2,958 | 3,120 | 363,859 | 420,033 | 117 |
| par:16:o | 1 / 6 / 16 | 14.5 | 20 | 19 | 2,454 | 2,787 | 169 |
| par:64:o | 7 / 38 / 64 | 57.0 | 78 | 56 | 9,058 | 10,957 | 159 |
| par:256:o | 27 / 152 / 256 | 222 | 287 | 202 | 34,295 | 40,683 | 154 |
| par:1024:o | 108 / 603 / 1,024 | 876 | 1,123 | 785 | 131,684 | 159,365 | 150 |

Fit over the 10 L1 warm scenes (N up to 1,024): c = 4.7 (+-1.0) + 1.34 (+-0.69) nodes + 0.06 (+-0.21) leaf slots + 0.674 (+-0.051) triangles, largest residual 23 percent (par:16:o). Triangles only: 5.5 + 0.794 per triangle, largest residual 28 percent. Ratio node : slot : triangle in rate 2.0 : 0.1 : 1 (latency steps of F67: 6.6 : 2.2 : 1).

## Against the predictions
1. H_L (latency steps over a constant in flight capacity) within 25 percent on all: MISSED (-28, -27, -23, -22 percent on the four o scenes; -24 percent at par:4:s). H_T (triangle count alone, 0.76 per triangle) within 25 percent on all 11: HELD (smallest margin -24 percent at par:16:o), p was 0.15.
2. c(o) / c(s) between 1.2 and 1.6 at N 256 and 1024: MISSED (1.15 and 1.15; the extra nodes and slots of the o scenes cost 15 percent, not 40).
3. Fit: intercept 3 to 12 HELD (4.7); triangle cost 0.15 to 0.8 HELD (0.67); slot cost 0.4 to 2.5 MISSED (0.06, consistent with zero); ratios within a factor 2 of F67's MISSED (slots and nodes are far cheaper in rate than in latency).
4. One warp latency within 15 percent of the F67 line for s scenes up to N 256: HELD (0.97 to 1.10). Beyond: 0.95 (1,024), 0.87 (4,096); the o scenes 0.83 to 0.88 (F67's node cost is too high for a warm tree with many nodes).
5. Written ambiguously ("L1 / (32 c)" as typed is 3 to 6, outside 100 to 250: MISSED as typed). The quantity I meant, L1w / c (lanes in flight at saturation), is 117 to 193 for N >= 16 (inside 100 to 250), falling with the size of the query (193 at 4 triangles, 117 at 4,096).
6. par:4,096:s (269 KB, past the 130 KB node state): c within 15 percent of H_L (+6 percent) and equal to H_T (0 percent): HELD for "no memory bound regime"; the extra cost of L2 fetches is not visible for coherent same ray queries.

## Reading
- At saturation one SM (its TPC's unit alone) tests a triangle every 0.67 to 0.79 cycles (about 1.3 to 1.5 per cycle) and a node costs about 1.3 cycles; leaf slots are free in rate. The unit's rate is a triangle rate: step latencies of F67 (343 / 113 / 52 per node / slot / triangle, cold) say nothing about rate except that the overlap differs per element: triangles overlap by 52 / 0.67 = 77, nodes by 343 / 1.3 = 260, slots without limit. A single in flight capacity (H_L, 142) does not describe it; 117 to 193 lanes in flight (L1w / c) is a total, not a per element constant.
- Whole GPU model use: for coherent queries of many triangles the cost per ray at saturation is about 5 + 0.7 per triangle tested plus 1.3 per node (cycles per SM), replacing latency over capacity.
- Not covered: random rays (rays not equal across lanes: the unit may not dedupe but memory traffic then limits, F62), queries that hit (candidate returns and resumes), triangles per slot above 2.9 (all scenes here have 1 to 2.9), the second SM of the TPC (F50: no gain), misses in flight (no scene reached a memory bound regime, so the number is not determined; a real Unit_3 needs distinct lines per lane and wide nodes).
- Caveat: every lane runs the same ray, so lanes may share work inside the unit. A mixed ray version of this sweep (rays on different triangles in the same scene) is the next check of "triangle rate".
