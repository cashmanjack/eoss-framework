# Struct_1 results: build flags and compaction, instance levels, box quantization (2026-11-17, tethys RTX 4090 GPU 1, OptiX only, no TTU code)

Predictions: model/predictions_struct_flags.md, predictions_struct_inst.md, predictions_struct_quant.md (committed adcfa7b, addendum 9fd3828, before any run). Code ttu/bench/Struct_1, model/struct_analyze.py, struct_quant.py, struct_quant_tri.py, replay/replay_quant.cpp. Raw files notes/struct1 (private; every file timestamped, opened with mode x). GPU 1 idle before every run, no fault. Timing noise: the 262,144 ray launch (0.13 ms) has a run to run spread (max minus min over 4 process runs) up to 12 percent, so differences below about 5 percent at 100,000 triangles are not resolved; the instance item was repeated with 2,097,152 rays (spread under 3 percent).

## 1. Build flags and compaction (soup 100,000 and 1,000,000 triangles, random rays; custom 100,000 boxes, 262,144 point queries)
Table (median of the process runs for time; ncu on one launch for requests and instructions; nodes and leaf slots by walking the dumped GAS, 1M soup from the header). Reference for the "rel" columns is bf none, upd 0, comp 0.

| config | buffer B | final B | header used B | nodes (walk) | leaf slots | refs (hdr +0x60) | refs / prim | depth | ns / ray | rel time | run spread | unit req / ray | rel req | SM sectors / ray | warp inst / ray | cold cyc | warm cyc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| soup 100k build_u0_c0 | 5616128 | 5616128 | 5616128 | 3493 | 38135 | 100000 |  | 6 | 0.5703 | 1.112 | 0.103 | 92.1 | 1.141 | 21.7 | 24 | 27136 | 10432 |
| soup 100k build_u0_c1 | 5625088 | 4830208 | 4830208 | 3493 | 38135 | 100000 |  | 6 | 0.5681 | 1.108 | 0.056 | 92.2 | 1.142 | 21.7 | 24 | 26969 | 9741 |
| soup 100k build_u1_c0 | 6497152 | 6497152 | 6497152 | 3493 | 38135 | 100000 |  | 6 | 0.5830 | 1.137 | 0.107 | 92.1 | 1.140 | 21.7 | 24 | 27162 | 9895 |
| soup 100k build_u1_c1 | 6506112 | 5554688 | 5554688 | 3493 | 38135 | 100000 |  | 6 | 0.5684 | 1.109 | 0.048 | 91.9 | 1.138 | 21.7 | 24 | 27130 | 9630 |
| soup 100k none_u0_c0 | 6513024 | 6513024 | 6513024 | 7064 | 70390 | 115000 |  | 6 | 0.5127 | 1.000 | 0.122 | 80.7 | 1.000 | 21.7 | 24 | 24442 | 9122 |
| soup 100k none_u0_c1 | 6523008 | 5115520 | 5115520 | 7064 | 70390 | 115000 |  | 6 | 0.5027 | 0.981 | 0.047 | 80.8 | 1.001 | 21.7 | 24 | 24288 | 9838 |
| soup 100k none_u1_c0 | 6533760 | 6533760 | 6533760 | 6426 | 63924 | 100000 |  | 6 | 0.5176 | 1.010 | 0.057 | 82.4 | 1.021 | 21.7 | 24 | 26362 | 9838 |
| soup 100k none_u1_c1 | 6533760 | 5983488 | 5983488 | 6426 | 63924 | 100000 |  | 6 | 0.5078 | 0.990 | 0.062 | 82.4 | 1.021 | 21.7 | 24 | 26306 | 9636 |
| soup 100k trace_u0_c0 | 7000448 | 7000448 | 7000448 | 7958 | 78059 | 125000 |  | 7 | 0.5039 | 0.983 | 0.008 | 79.5 | 0.985 | 21.7 | 24 | 25266 | 9360 |
| soup 100k trace_u0_c1 | 7011200 | 5718144 | 5718144 | 7958 | 78059 | 125000 |  | 7 | 0.5186 | 1.011 | 0.064 | 79.5 | 0.985 | 21.7 | 24 | 25634 | 9616 |
| soup 100k trace_u1_c0 | 6329088 | 6329088 | 6329088 | 6426 | 63924 | 100000 |  | 6 | 0.5203 | 1.015 | 0.094 | 82.7 | 1.025 | 21.7 | 24 | 26670 | 10132 |
| soup 100k trace_u1_c1 | 6329088 | 5778816 | 5778816 | 6426 | 63924 | 100000 |  | 6 | 0.5332 | 1.040 | 0.015 | 82.7 | 1.025 | 21.7 | 24 | 26342 | 9666 |
| soup 1M build_c0 | 56147072 | 56147072 | 56147072 | 34053 |  | 1000000 | 1.000 |  | 0.7891 | 1.114 | 0.079 | 113.5 | 1.130 | 21.7 | 23 |  |  |
| soup 1M build_c1 | 56235904 | 48168960 | 48168960 | 34053 |  | 1000000 | 1.000 |  | 0.7979 | 1.127 | 0.039 | 113.6 | 1.130 | 21.7 | 23 |  |  |
| soup 1M none_c0 | 65112576 | 65112576 | 65112576 | 72580 |  | 1150000 | 1.150 |  | 0.7080 | 1.000 | 0.086 | 100.5 | 1.000 | 21.7 | 23 |  |  |
| soup 1M none_c1 | 65211648 | 48989696 | 48989696 | 72580 |  | 1150000 | 1.150 |  | 0.7100 | 1.003 | 0.107 | 100.5 | 1.000 | 21.7 | 23 |  |  |
| soup 1M trace_c0 | 69949696 | 69949696 | 69949696 | 81597 |  | 1250000 | 1.250 |  | 0.6934 | 0.979 | 0.085 | 98.7 | 0.982 | 21.7 | 23 |  |  |
| soup 1M trace_c1 | 70055680 | 53668096 | 53668096 | 81597 |  | 1250000 | 1.250 |  | 0.6934 | 0.979 | 0.039 | 98.7 | 0.982 | 21.7 | 23 |  |  |
| custom 100k pts build_c0 calls/q 40.98 | 2510848 | 2510848 | 2510848 | 6511 | 66807 | 0 | 1.000 | 7 | 1.1004 | 1.002 | 0.057 | 99.8 | 0.857 | 69.3 | 538 |  |  |
| custom 100k pts build_c1 calls/q 40.98 | 2510848 | 1633920 | 1633920 | 6511 | 66807 | 0 | 1.000 | 7 | 1.0977 | 1.000 | 0.060 | 99.8 | 0.858 | 69.3 | 538 |  |  |
| custom 100k pts none_c0 calls/q 36.14 | 3743104 | 3743104 | 3743104 | 9040 | 83855 | 0 | 1.000 | 7 | 1.0977 | 1.000 | 0.057 | 116.4 | 1.000 | 63.1 | 479 |  |  |
| custom 100k pts none_c1 calls/q 36.14 | 3743104 | 1957760 | 1957760 | 9040 | 83855 | 0 | 1.000 | 7 | 1.0996 | 1.002 | 0.084 | 116.5 | 1.000 | 63.1 | 479 |  |  |
| custom 100k pts trace_c0 calls/q 36.14 | 3743360 | 3743360 | 3743360 | 9040 | 83855 | 0 | 1.000 | 7 | 1.0986 | 1.001 | 0.032 | 116.4 | 1.000 | 63.1 | 479 |  |  |
| custom 100k pts trace_c1 calls/q 36.14 | 3743360 | 1958016 | 1958016 | 9040 | 83855 | 0 | 1.000 | 7 | 1.0977 | 1.000 | 0.057 | 116.4 | 1.000 | 63.1 | 479 |  |  |

Scoring (predictions_struct_flags.md):
1. Compaction: tree identical (nodes, leaf slots, references, requests within 0.2 percent), time within 3 percent: HELD. Final size 5 to 25 percent below the buffer on soup (none 21.5, trace 18.3, build 14.0 percent at 100,000; 24.8, 23.3, 14.2 at 1M; with the update flag 8.4 to 8.7): HELD. "Equals the used size in the header": MISSED, the header size word of an uncompacted build equals the buffer OptiX asked for (it includes the slack), not a used size; after compaction it equals the compacted size. Custom boxes compact much more (none 3.74 to 1.96 MB, minus 47.7 percent; fast build minus 34.9 percent): outside the 5 to 25 range, which I had written for soups only.
2. ALLOW_UPDATE alone: MISSED. It removes the spatial splits: references 1.00 per triangle instead of 1.15 (soup 100,000), nodes minus 9.0 percent (7,064 to 6,426), unit requests plus 2.1 percent, time plus 1 to 4 percent (inside the noise). With the update flag fast trace gives the same tree as none (6,426 nodes, 100,000 references, requests 82.4 and 82.7), buffer sizes differ (6.53 and 6.33 MB). Fast build with update is the same tree as fast build without (3,493 nodes) but a 16 percent larger buffer.
3. Fast trace against none (no update): HELD. References 1.25 against 1.15 (more splits), size plus 7.5 percent, nodes plus 12.7 percent, depth 7 against 6, requests 0.985, time 0.98 to 1.01. At 1M the same (references 1.25, size plus 7.4 percent, time 0.979).
4. Fast build against none: HELD for references (1.00), size (minus 13.8 percent at 100,000, minus 13.8 at 1M), requests (plus 14.1 percent, plus 13.0 at 1M), time (1.11 to 1.14 at 100,000, 1.11 to 1.13 at 1M). The part "the ratio at 1M is at least the 100,000 value" is NOT resolved: 1.114 and 1.127 at 1M against 1.112 and 1.108 at 100,000, a tie inside the noise. A fast build tree has half the nodes (3,493 against 7,064) and 14 percent more requests.
5. Custom boxes: fast build calls per query plus 13.4 percent (40.98 against 36.14; HELD, range 5 to 50), nodes minus 28 percent (6,511 against 9,040), unit requests minus 14 percent, time unchanged (1.100 against 1.098 ns per query: the intersection program calls, not the node requests, set the time here). Fast trace is bit for bit the same tree as none (9,040 nodes, 83,855 leaf slots, same calls), compaction changes nothing but size: HELD. Custom primitive references are never split (1.00 in every mode).
6. One warp cold latency (cycles, probe of 100 launches): none 24,442, trace 25,266, build 27,136 (plus 3.4 and plus 11 percent); requests minus 1.5 and plus 14 percent: HELD within the stated margin for build; for trace the sign is opposite (inside the probe noise, not distinguishable).
Added: the SM sector count per ray (21.7) and warp instructions per ray (24) do not depend on any build flag for triangles.
Not covered: real build times (not measured), other geometry types, refit (F170), IAS flags.

## 2. Transforms and two instance levels (soup 100,000 triangles and the custom box cloud, one instance per level)
ncu on a 262,144 ray launch (one run), timing on 2,097,152 rays (median of 3 process runs, ns per ray; custom = point queries):

| config | soup ns / ray | custom ns / query | unit req / ray (soup) | SM sectors / ray (soup) | warp instructions / ray (soup) | custom warp inst / query | instance id returned | hit fraction (soup) |
|---|---|---|---|---|---|---|---|---|
| none (single GAS) | 0.4272 | 0.9548 | 80.8 | 21.75 | 23.5 | 479 | 2147483647 | 0.8291 |
| id1 (one identity instance, SINGLE_LEVEL pipeline) | 0.4341 | 1.2422 | 80.9 | 21.03 | 16.0 | 306 | 7 | 0.8291 |
| id1any (id1 under ALLOW_ANY, the control) | 0.4302 | 0.9614 | 80.9 | 21.91 | 24.6 | n.a. | 7 | 0.8291 |
| r90_1 (rotation about z by 90 degrees) | 0.4229 | 1.2422 | 80.8 | 21.03 | 16.0 | n.a. | 7 | 0.8304 |
| r37_1 (rotation 37 degrees plus translation) | 0.3950 | 1.1528 | 75.5 | 21.03 | 16.0 | n.a. | 7 | 0.7777 |
| id2 (two identity levels) | 0.4463 | 1.0049 | 81.1 | 22.81 | 38.6 | 523 | 5 | 0.8291 |
| r90_2 (rotations about z and x) | 0.4233 | 1.0049 | 80.9 | 22.81 | 38.6 | 522 | 5 | 0.8291 |

Per ray results of none, id1 and id2 (hit, t, primitive): identical on all 262,144 rays. Intersection program calls per query (custom): 36.14 in none, id1 and id2, 36.10 in r90_2. One warp cold latency (cycles, probe of 100 launches of the first 32 rays): none 24,480, id1 23,814, id1any 25,123, r90_1 27,074, r37_1 15,795, id2 28,903, r90_2 21,678; the rotated cases trace different rays through the tree, so only id1, id1any and id2 are comparable (id2 against id1any plus 3,780 cold, plus 2,263 warm).

Scoring (predictions_struct_inst.md):
1. Results: HELD for hit, t, primitive (identical) and r90 hit fractions (0.8304 and 0.8291 against 0.8291). MISSED for the instance id of id2: OptiX returns the innermost instance (5), not the outer one (7); a single GAS returns 2147483647 (not instanced).
2. One level: HELD. r90_1 against id1: time 0.974 (within 5 percent), warp instructions equal, unit requests 80.8 against 80.9. r37_1 0.91 in time (within 10 percent, borderline; its work is 6.7 percent lower in requests, 75.5, because the rotated geometry meets the rays differently). id1 against none in time 1.016 (HELD); warp instructions NOT within 5 percent (id1 16.0 against none 23.5, a pipeline path effect, see 3).
3. Control: time HELD (0.991), warp instructions MISSED: id1 under ALLOW_ANY executes 24.6 warp instructions per ray against 16.0 under SINGLE_LEVEL_INSTANCING (plus 54 percent). So the pipeline flag changes the OptiX code path on its own. For custom primitives the effect is big in time: id1 is 30 percent slower than the same instance under ALLOW_ANY (1.242 against 0.961) and than no instancing (0.955), and r90_1 the same; id2 (which needs ALLOW_ANY) takes 1.005. One configuration (uniform boxes, point queries), cause not found (the SM instruction count is LOWER under SINGLE_LEVEL, 306 against 479).
4. Second level: by the rule as written (id2 against id1 under ALLOW_ANY, the control failed, so this is the comparison it asks for): warp instructions 38.6 against 24.6 per ray (1.57, over the 1.5 threshold), SM load sectors per ray 22.81 against 21.91 (rise), time per ray 1.037 (needs at least 1.15). Not all three, and not all within 10 percent: NOT DETERMINED by my rule (mixed). Against id1 itself the instruction ratio is 2.41. Numeric predictions under the SM hypothesis: warp instructions 1.5 to 5 times: held; time 1.2 to 3 times: MISSED (1.03); cold latency plus 500 to plus 3,000: missed by a little (plus 3,780 against id1any, plus 5,089 against id1; probe noise at least 3,000 by the r90 versus id1 difference). Reading beyond the rule (labeled as reading): the unit's request count is unchanged (81.1 against 80.9) while the SM executes about 14 more warp instructions per ray (about 450 per 32 ray warp) and loads about 0.9 more sectors per ray, so the second level is not free in the unit: the SM does extra work for it. At saturation that work is hidden behind the unit (time plus 3 percent); one warp latency rises. This is an inference from counters, not read from the OptiX code or confirmed with our own sequence.
5. r90_2 against id2: HELD (time 0.949, warp instructions equal 38.6): a non identity transform costs nothing extra at either depth; no identity shortcut and no per transform cost.
6. Unit requests of id2 within 15 percent of id1: HELD (81.1 against 80.9).
Not covered: many instances per level, instance flags, motion, hardware confirmation (OptiX code read or our own sequence with a non identity matrix: a new TTU sequence class, not done).

## 3. Box quantization false positives (custom boxes, 100,000, fresh tree, point queries and segments)
H = hardware intersection program calls per query, Q = replay on the decoded quantized boxes, E1 = replay on exact tight boxes (same topology), E0 = exact candidates (the query meets the primitive box).

| scene | H | Q | E1 | E0 | H / Q | Q / E1 | E1 / E0 | H / E0 | queries H = Q | nodes Q / E1 | leaf slots Q / E1 | queries with extra node | queries with extra call |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| u 0.035 points | 4.724 | 4.679 | 4.522 | 4.173 | 1.0097 | 1.0346 | 1.0837 | 1.1321 | 0.962 | 1.0273 | 1.0344 | 0.199 | 0.127 |
| u 0.07 points | 36.139 | 35.889 | 35.008 | 32.524 | 1.0070 | 1.0251 | 1.0764 | 1.1112 | 0.820 | 1.0235 | 1.0246 | 0.297 | 0.500 |
| u 0.07 segments (tmax 0.25) | 121.129 | 120.489 | 118.230 | 111.930 | 1.0053 | 1.0191 | 1.0563 | 1.0822 | 0.607 | 1.0197 | 1.0187 | 0.488 | 0.815 |
| c uniform points (clusters, mostly empty) | 0.111 | 0.110 | 0.106 | 0.098 | 1.0106 | 1.0381 | 1.0782 | 1.1312 | 0.999 | 1.0241 | 1.0378 | 0.063 | 0.003 |
| c centre points | 4.727 | 4.684 | 4.571 | 4.299 | 1.0091 | 1.0248 | 1.0633 | 1.0996 | 0.963 | 1.0238 | 1.0245 | 0.172 | 0.092 |
| t uniform points (thin rods) | 3.308 | 3.243 | 3.042 | 2.761 | 1.0198 | 1.0662 | 1.1016 | 1.1978 | 0.943 | 1.0273 | 1.0664 | 0.428 | 0.167 |
| t centre points | 4.346 | 4.283 | 4.085 | 3.765 | 1.0148 | 1.0484 | 1.0851 | 1.1544 | 0.944 | 1.0265 | 1.0485 | 0.420 | 0.165 |
| t segments | 38.956 | 38.554 | 37.133 | 35.270 | 1.0104 | 1.0383 | 1.0528 | 1.1045 | 0.695 | 1.0226 | 1.0384 | 0.658 | 0.714 |
| u 0.07 centre points | 37.131 | 36.879 | 36.009 | 33.530 | 1.0068 | 1.0242 | 1.0739 | 1.1074 | 0.818 | 1.0235 | 1.0238 | 0.299 | 0.495 |
| o (u 0.07 shifted by 1000) points | 36.170 | 35.947 | 35.104 | 32.576 | 1.0062 | 1.0240 | 1.0776 | 1.1103 | 0.839 | 1.0230 | 1.0235 | 0.291 | 0.484 |
Triangles (100,000 soup, 262,144 random rays, replay only, mode 1 assignment, E1 = triangle box clipped to the quantized slot box): nodes 12.73 (Q) against 12.37 (E1), leaf slots 5.11 against 4.93, weighted requests (2.2 per node, 9.9 per leaf slot, F187) 78.56 against 76.02, measured 80.8; Q over E1 1.030 (nodes), 1.036 (leaf slots), 1.033 (requests); 28.3 percent of rays visit an extra node, 14.9 percent an extra leaf slot; hit decision and nearest primitive identical on all rays.

Scoring (predictions_struct_quant.md):
1. Quantized box contains the exact box: 99.90 to 99.998 percent of slots (MISSED the literal 100 percent; the exceptions, 0.002 to 0.1 percent of slots, coincide with the 0.00 to 0.22 percent of leaf slots whose assigned primitive count differs from the type byte count, so they are assignment errors, not narrower boxes; far below the 1 percent that would invalidate E1).
2. H within 3 percent of Q: HELD on every row (0.5 to 2.0 percent); H closer to Q than to E1: HELD on every row. H = Q for at least 90 percent of the point queries: MISSED on the two dense rows (u 0.07: 82.0 percent, o: 83.9); held on u 0.035 (96.2), c (99.9 and 96.3), t (94.3 and 94.4). The hardware makes slightly MORE calls than the decoded box replay (0.5 to 2 percent, in 4 to 39 percent of queries, never fewer than 0.03 percent): a hardware rounding rule or test form that the replay does not have; cause not determined.
3. Uniform: Q over E1 1.035 and 1.025 (range 1.02 to 1.08: HELD), E1 over E0 1.084 and 1.076 (1.03 to 1.12: HELD), H over E0 1.132 and 1.111 (1.05 to 1.20: HELD; F116 had 1.10 on a different cloud).
4. Thin rods have the largest quantization share (6.6 percent on uniform points, 4.8 on centre points, 3.8 on segments; uniform boxes 2.5 and 1.9): HELD for "largest, at least twice the uniform excess" (2.6 and 2.0 times), MISSED for "at least 1.10" (1.066 at most).
5. Clusters: Q over E1 1.038 on uniform queries (HELD; but only 0.11 calls per query), 1.025 on centre queries (addendum range 1.03 to 1.15: MISSED by a little).
6. Offset by 1000: H 36.170 against 36.139 (0.1 percent, HELD).
7. Nodes plus 2.0 to 2.7 percent (range 2 to 12: HELD on all ten rows); leaf slots plus 1.9 to 6.6 percent (range 3 to 15: held on u 0.035, c uniform and the three t rows; MISSED on the rows with 2.4 to 2.5 or less: u 0.07 points 2.46, u 0.07 segments 1.87, u 0.07 centre 2.38, o 2.35, c centre 2.45); queries with an extra leaf slot or call 50 percent on u 0.07 points (30 to 80: HELD), 12.7 on u 0.035 (below 50: HELD), extra node 29.7 percent (just below the 30).
8. Segments: ordering held (H closer to Q, Q over E1 above 1); the Q over E1 value on u 0.07 segments 1.019 is just below the 1.02 lower bound.
9. Triangles: Q requests 78.56 against 80.8 (2.8 percent below, HELD within 10), E1 3.3 percent below Q (range 3 to 15: HELD at the edge) and 5.9 percent below the measurement (further than Q: HELD).
Addendum predictions (centre queries): H within 3 percent of Q on all: HELD; Q over E1 c 1.025 (range 1.03 to 1.15 MISSED), t 1.048 (HELD), u 1.024 (HELD); H = Q for at least 80 percent: HELD (96.3, 94.4, 81.8).
Not covered: the F166 Mochi residual was not tested (no Mochi dumps used here); refit trees; instance level boxes; node precision beyond what the decoded boxes say (the replay with the decoded boxes already misses 0.5 to 2 percent of the hardware calls).
