# HG_1 results: one warp, our harness against OptiX, 2026-11-16 (tethys, GPU 1, RTX 4090)

Predictions: model/predictions_hg1.md (committed 4c91dee before any run; addendum A 54d19d3 before the second round of runs). Raw data stays private (notes/hg1/). Script: ttu/bench/HG_1/run.py, table: model/hg1_analyze.py. Every harness lane of every run equals the OptiX result of the same ray (hit, primitive, t). No fault, no hang, GPU idle after every run, stage 0 first for every scene.

## What was timed
- Harness: build_patch --lclock on the checked loop build (clock64 before the first group, after the resume loop ended; group bytes unchanged, check_seq ALL PASS). Whole ray, all rounds, one warp.
- Harness group only: the existing --clock build (one query, the pure traversal group).
- OptiX: the setup3 raygen plus probes (hg1_dev.cu, then hg1b_dev.cu which traces the same ray up to 3 times). TRACE = latest lane return from optixTrace minus earliest start (all 32 lanes return at the same clock). 100 launches per scene, median, spread under 0.5 percent in every scene. L2 warm, L1 cold at the start of each launch.

## Table (cycles per warp, median)
| scene | harness group (1 query) | harness loop (whole ray) | loop minus group | OptiX TRACE, first call | OptiX 2nd and 3rd call | OptiX minus harness loop |
|---|---|---|---|---|---|---|
| tri1 | 1630 | 2211 | 581 | 6856 | 4750 | 4645 |
| grid 12 | 1618 | 2200 | 582 | 6831 | 4750 | 4630 |
| grid 144 | 1937 | 2516 | 579 | 7072 | 4795 | 4554 |
| grid 200 | 1906 | 2559 | 653 | 7098 | 4795 | 4538 |
| grid 2500 | 2280 | 2866 | 586 | 7450 | 4837 | 4584 |
| grid 4096 | 2280 | 2854 | 574 | 7385 | 4837 | 4528 |
| grid 20736 | 2556 | 3149 | 593 | 7743 | 4877 | 4582 |
| soup, 32 random rays (1.91 rounds mean, 3 max) | 13469 | 19596 | 6127 | 24215 | 11279 | 4663 |
Four independent series of the first call (first run, control with nrep 1, the 3 trace run, the rotated run) agree within 1 percent (tri1 6796 / 6853 / 6860 / 6859; soup 24205 / 24215 / 24303 / 24345).

## Reading rule outcome (as written before the run)
Gap = OptiX TRACE / harness loop: tri1 3.1, grids 2.5 to 3.1, soup 1.23 (four series 1.233 to 1.236). Every scene is over 20 percent: by the rule, the one warp difference is in the query path, not concurrency, and the term must be found by the per term comparison. The harness constants do NOT represent OptiX at one warp as a whole ray; they represent its query part exactly (below).

## Per term comparison
1. Query part (nodes, levels, leaves, triangles): equal. The harness group costs 926 cycles more at 20,736 triangles than at one (1630 to 2556); OptiX TRACE costs 924 more (6796 to 7720, first run). The warm increments are the same to one cycle: harness second session 782 / 782 / 827 / 826 / 867 / 867 / 908 (F176), OptiX warm trace 4750 / 4750 / 4795 / 4795 / 4837 / 4837 / 4877: the difference is 3968 / 3968 / 3968 / 3969 / 3970 / 3970 / 3969 in every one round scene. A fit of the OptiX excess over the harness loop against the group cost has slope -0.07 (excess 4528 to 4645 for group 1618 to 2556): no node term.
2. Per optixTrace call, warm (not in the model, not in the harness): a constant of about 3,969 cycles on top of the warm query, the same in every scene (1 to 3 rounds). ncu check (done, same process type, stage 0 binary, kernel optixLaunch): the HG_1 raygen launch executes 1,143 warp instructions (36,576 thread instructions for 32 rays) for tri1 and for grid 2500 alike, 1,173 in the first launch; the harness loop and group are about 100 instructions. The single warp cannot hide the dependent issue latency of the other roughly 1,050: 3,969 cycles for about 1,050 extra instructions is about 3.8 cycles per instruction. A second program (the Flags raygen with 4 payload values, 709 instructions per warp) has a warm trace of about 3,400, i.e. 2,620 above the 782 query: 434 fewer instructions, 1,350 fewer cycles, 3.1 cycles per instruction. So the constant is a property of the instruction count of OptiX's trace path and of the program around it, not of the query. This is the "issue and instruction overhead" term. With 13 warps per SM the other warps hide it, which is why it does not slow OptiX in the whole GPU rate.
3. First call of a launch: about 1,200 cycles more than the second call after removing the cold L1 data cost of the query (first minus second 2110 / 2081 / 2277 / 2302 / 2613 / 2548 / 2866 less the harness cold minus warm 848 / 836 / 1110 / 1080 / 1413 / 1413 / 1648: 1262 / 1245 / 1167 / 1222 / 1200 / 1135 / 1218). Mean 1,207, spread 40. Cold code (instruction fetch of OptiX's traversal code) is the likely cause (not separated; nothing in the harness has it because its code is a few hundred bytes).
4. Harness loop: 580 cycles per round of its own (the loop reloads the 14 words the group overwrote, one L2 round trip, plus the pending logic), which OptiX does not pay as a separate item. Not in the OptiX numbers.
5. Restart rounds: soup OptiX minus harness loop 4663 against 4580 for one round scenes: the per round difference is +40 per extra round with 2 extra rounds (max 3) and one scene: no per round term separated from the node term (one multi round scene only).
Totals for one round scenes: OptiX TRACE = harness group (cold L1) + 3,969 (per call) + 1,207 (first call) with residual under 60 cycles in every scene; harness loop = group + 580.

## Predictions against measured
| item | predicted | measured |
|---|---|---|
| harness tri1 loop span | 1640 (1560 to 1780) | 2211 (miss: the loop reload term, 581, not counted) |
| harness soup, 32 random rays | 7500 (5000 to 11000) | 19596 (miss, 2.6 times: slowest lane has 3 rounds) |
| OptiX TRACE tri1 | 1900 (1700 to 2300) | 6856 (miss, 3.6 times) |
| OptiX excess is one additive constant, no slope in nodes or queries | form predicted | held (4528 to 4663), 18 times larger than the 250 predicted |
| gap tri1 / soup | 1.16 / 1.04 | 3.1 / 1.23 |
| OptiX raygen minus trace | 350 | 1038 (pre 808 plus post 235) |
| queries per ray tri1, grids 1; soup 1.9 | | 1.0 and 1.91 (harness rounds); OptiX side not counted (no replay of the OptiX side run; indirect only, the constant difference) |
| addendum: trace 2 below 3000 with 70 percent | | MISS: 4750 for tri1 (not first call cost only) |
| addendum: soup repeat 6000 to 14000 | | 11279 (inside) |
| addendum: soup rotated 16000 to 22000 | | 11239: the rotated rays had already been traced by other lanes in trace 1, their nodes were in L1, so the test did not have unseen data (design flaw, found after the run) |
| addendum B group only | 1630 / 1608 / 1934 / 1982 / 2281 / 2270 / 2584, within 3 percent | 1630 / 1618 / 1937 / 1906 / 2280 / 2280 / 2556: all within 3.8 percent (grid 200 3.8 percent low) |
| addendum B soup first query | 6000 (4000 to 9000) | 13469 (miss) |
| addendum B loop overhead | 590 (400 to 800) | 574 to 653 (held) |

## What was not done or not separated
- 4, 13 and 28 warps on one SM: not possible for OptiX (its warps spread over the GPU).
- The cold L2 variant (256 MB flush) was not run.
- OptiX queries per ray were not counted (no replay of the OptiX side, no ray counter on this GPU); the equal excess in 1 and 3 round scenes is the only evidence they agree.
- The probe cost was not measured separately (8 clock reads and 8 stores per lane); the repeated calls of the same raygen (trace 2 and 3 equal to the cycle) show the probes do not drift, not that they cost nothing. Stated bound: 8 clock reads and stores are under 100 cycles of 3,400 to 6,900.
- Cold code versus cold setup in the 1,207 first call term is not separated.
- Different payload counts change the constant (3,969 here, about 2,600 with 4 payload values in the Flags raygen): the number belongs to this program.
- Instructions per ray at the whole GPU rate for the HG_1 raygen: not measured (the ncu run was the one warp launch). The 4090 ncu sectors: l1tex__t_sectors minus the load store pipe is about 0 on coherent scenes (the RT unit requests are not visible there), so fills (xbar to L1 read sectors) are used instead.
