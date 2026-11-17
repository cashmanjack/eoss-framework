# HG_1 predictions: one warp, our harness against OptiX, written 2026-11-16 BEFORE any run of this test

Pre-registration. The test: one block of one warp (32 rays), one SM, idle GPU 1 of tethys (RTX 4090), the same 32 rays and the same in process GAS for both. Nothing of HG_1 has been run or built when this file is committed. The reading rule is the one in MASTER_PLAN.md (HG_1 paragraph, Phase 10); this file makes its quantities exact and adds the cases the rule left open. It is not edited after the run; results go in model/results_hg1.md.

## What is timed (fixed now)
- Harness (build_patch, new flag --lclock on the existing --slim --remap --loop build): clock64 read right before the first TTU group and right after the resume loop has ended and the final hit is in place. Includes every query of the ray, the reloads and the pending logic of the loop. Excludes the 8 ray word loads at the start and the result stores. Per warp span = end minus start, lane 0 value (all lanes equal).
- OptiX (own raygen in a new hg1_dev.cu, the setup3_dev.cu program plus probes): ta at raygen entry, tb right before optixTrace, tc right after it returns (payload in registers, closest hit program already run), td after the result struct is built and stored. Reported: TRACE = tc - tb (the query path, the quantity the reading rule compares), RAYGEN = td - ta. Lane 0 of the warp.
- Launch: 1 block, 1 warp (OptiX launch size 32). 200 launches after one cold first launch, median of the launches. "Launch to launch" state = L2 warm, L1 cold (as Latency_1: tri1 1630 on this GPU). A second state, cold L2 after a 256 MB flush, for the single triangle only.
- Scenes (same as Latency_2 plus the case of F73, F75 and the ablation): tri1 (coherent, 32 identical rays), grid 12, 144, 200, 2500, 4096, 20736 triangles (32 identical rays on the triangle N/2), soup:100000 with 32 random rays (the first 32 of the ray set make_rays(…, seed 2026)). Every harness lane is checked against the OptiX reference of the same ray (hit, primitive, t) before any number is used.
- Queries per ray: harness side from the rounds counter of the loop; OptiX side from the CPU replay of F81 (no ray counter on this GPU). If the two differ for a scene, that scene is reported but not used for the rule.

## Reading rule (as written in MASTER_PLAN, made exact, with the middle case added before the run)
Gap = OptiX TRACE median / harness span median, per scene.
- Gap within 10 percent (0.90 to 1.10): the harness constants represent OptiX at one warp; the 1.6 times whole GPU gap is concurrency (a shared per SM resource), not the query path.
- Gap over 20 percent (above 1.20 or below 0.80): the query path differs; the term that carries it is found by the per term comparison below.
- Between 10 and 20 percent: the rule says nothing (it left this case open). Decided only by the per term comparison: an additive fixed excess below about 400 cycles that does not grow with queries or nodes is called "small fixed term, constants represent OptiX"; an excess that grows with queries or nodes is called "query path differs".
- Headline scenes are soup:100000 (the case of the 1.6 times gap) and tri1 (the case of F73, F75, 4 to 10 times). The grid scenes decide which term.
- A harness lane that disagrees with OptiX, a timeout, a fault or a probe that moves the span of the harness by more than 5 percent (checked by a run without the probe on one scene) stops the test.

## Predictions (cycles per warp, median; range in brackets)
1. Harness span, launch to launch: tri1 1640 (1560 to 1780: Latency_1 1630 plus the loop's few instructions); grid 12 / 144 / 200 / 2500 / 4096 / 20736: 1610 / 1940 / 1990 / 2290 / 2280 / 2600 (Latency_2 values of F176, plus about 10). Soup, 32 random rays: 7500 (5000 to 11000; one chain of about 13 nodes, 5 leaves, 9 triangles per ray from F81 at F67 costs, plus 0.9 restarts, the warp waits for its slowest lane).
2. OptiX TRACE: tri1 1900 (1700 to 2300); the excess over the harness is a fixed part: 250 (100 to 600) cycles for the code OptiX runs around the query (instance node load, ray and payload setup, closest hit call, return), the same for every scene. So grid scenes: harness + 250, and soup 7800 (5300 to 11500).
3. Gaps from 1 and 2: tri1 1.16 (1.04 to 1.40), grid 12 1.16, grid 20736 1.10, soup 1.04 (0.90 to 1.20).
4. OptiX RAYGEN minus TRACE (ray load from the params buffer, result build): 350 (200 to 600), dominated by one L2 round trip for the ray words.
5. Queries per ray: tri1 and every grid scene 1 on both sides; soup 1.9 per ray on average (F81: 1.85 to 2.0), the warp maximum 3 to 4.
6. Reading rule outcome predicted: soup within 10 percent (concurrency reading), tri1 in the 10 to 20 percent middle case, called "small fixed term" by the per term comparison. If tri1 comes out above 1.40 or soup above 1.20 that is a prediction miss and the query path differs.
7. Cold L2 (256 MB flush), tri1: harness 2630 (F176 cold), OptiX TRACE 2900 (2650 to 3400), the same fixed excess.

## Per term model comparison (fixed now, applied after the run)
Terms per warp, from the model (performance_model.md, F67, F25): fixed cost per query (queries x fixed, 740 L1 warm or about 934 with the instance line when L1 is cold), node fetches (343 per node, L1 cold), leaves (113), triangles (52), then the terms that the harness has no constants for: issue and instruction overhead (OptiX non TTU instructions between queries) and the input and result memory term (ray loads, result stores, both measured by the probes).
Procedure:
1. Model value M per scene from the replay counts (queries, nodes, leaves, triangles of the warp's slowest lane) and the F67 costs.
2. Residuals R_h = harness span - M and R_o = OptiX TRACE - M. The harness residual says how well the model describes the harness; the OptiX residual how well it describes OptiX.
3. D = OptiX TRACE - harness span per scene, regressed on [1, queries, nodes] over the eight scenes (least squares, intercept free). Reading: intercept = fixed excess (issue, instruction and setup overhead), queries slope = restart cost difference, nodes slope = node fetch difference.
4. Prediction: intercept 250 (100 to 600), queries slope 0 (within +-150 per query), nodes slope 0 (within +-30 per node). A nodes slope above 60 per node or a queries slope above 300 per query would name the node fetch or the restart term as the difference.
5. The ncu counters (instructions, L1 and L2 sectors per ray of the OptiX launch) are used only if ncu runs on tethys (Memory_2, below in the queue); otherwise the issue term is the intercept by elimination and is said to be unmeasured.

## What each outcome would mean
- Soup within 10 percent and tri1 gap an additive 250: the one warp harness constants hold for OptiX; the 1.6 times at 142 SMs is not in the query path. Next test: Sched_1, Sched_2 and Unit_1 (shared per SM resource).
- Soup gap over 20 percent: OptiX queries are faster or slower per query than ours at one warp; the nodes and queries slopes say whether it is the fetch or the restart.
- Tri1 gap far above 1.4: the 4 to 10 times of F73 and F75 is partly in the query path (instance node, closest hit call), not only our loads.

## Not covered (stated now)
- 4, 13 and 28 warps on one SM: OptiX cannot be placed on one SM (its warps spread over the GPU), so the added gap at more warps is not run here. The whole GPU comparison at 13 warps per SM is the ablation of 2026-11-11.
- A second pass over the same rays with warm L1 inside one OptiX raygen is not compared to the harness loop (the loop build has no second session); the single query builds (--twice) give 782 for the harness and are compared in the single triangle and grid scenes only if time allows.


---
## Addendum A, written 2026-11-16 AFTER the first HG_1 run (the numbers below under "First run" were seen before this addendum; the predictions above are not edited)

### First run (one warp, 100 launches each, stage 0 first, every harness lane equals OptiX, no fault, GPU idle after; raw data notes/hg1, private)
Median cycles per warp, L2 warm, L1 cold per launch:
| scene | harness loop span | OptiX TRACE | OptiX minus harness | gap |
|---|---|---|---|---|
| tri1 | 2225 | 6796 | 4571 | 3.05 |
| grid 12 | 2192 | 6767 | 4575 | 3.09 |
| grid 144 | 2519 | 7042 | 4523 | 2.80 |
| grid 200 | 2491 | 7006 | 4515 | 2.81 |
| grid 2500 | 2865 | 7433 | 4568 | 2.59 |
| grid 4096 | 2865 | 7356 | 4491 | 2.57 |
| grid 20736 | 3146 | 7720 | 4574 | 2.45 |
| soup (1.9 rounds per ray, max 3) | 19637 | 24205 | 4568 | 1.23 |
Per launch spread under 0.5 percent. OptiX: the span from raygen entry to optixTrace is 773 cycles and from the return to the end 150 in every scene; all 32 lanes return from optixTrace at the same clock.
Against the predictions above: harness tri1 predicted 1640, measured 2225 (miss; the loop reloads the 14 words the group overwrote after every query, one L2 round trip, about 590, which I did not count); soup harness predicted 7500, measured 19637 (miss, 2.6 times: the warp waits for its slowest lane, 3 rounds, each round a full chain); OptiX TRACE predicted 1900, measured 6796 (miss, 3.6 times); predicted gaps 1.16 / 1.04 missed (3.05 / 1.23). What did hold: the OptiX excess is one additive constant, the same 4500 to 4575 cycles in all eight scenes, with no slope in nodes or queries (predicted form: intercept only, slopes 0), but 18 times larger than the 250 predicted.
By the rule as written: tri1 over 20 percent, soup over 20 percent (1.23): "the query path differs". But a constant that does not depend on the scene or on the number of rounds is not a property of the query path of a scene; it looks like a cost of the first optixTrace of a launch (cold instruction fetch of OptiX's large traversal code, or OptiX's per call setup), which a persistent kernel never pays again.

### Test A (HG_1b): trace the same ray again inside the same raygen
Probe variant hg1b_dev.cu: after the first optixTrace the raygen traces again, 3 traces in all, tb_k and tc_k read around each; call site, code and payload registers identical for all three; only trace 0 stores the result. Variant R (rotated): trace k uses the ray of launch index (i + k) mod 32 instead of its own (code warm, data not seen before for the soup; identical to the plain repeat for tri1 and the grid scenes, whose 32 rays are the same).
Predictions (cycles, the warp, median of 100 launches):
1. If the 4.5k is first call cost (code or setup that is warm after one call): trace 2 and 3 of tri1 in 1100 to 2500 (point 1500: the harness warm query 782 plus the loop reload term 590 plus about 150), and the gap of trace 2 to the harness first query (2225) 0.5 to 1.1. If trace 2 stays above 5000 the excess is paid on every optixTrace and is a property of the OptiX query path: then the reading rule's answer stands (query path differs, by about 4.5k per trace, additive).
2. The probability I give to "trace 2 below 3000": 70 percent.
3. Soup, plain repeat (same rays again, L1 warm for data): trace 2 about 6000 to 14000 (the second trace finds every node in L1: F25 gives 740 fixed plus 43 per level against 934 plus 343 per node cold; 3 rounds of about 2000 to 4000).
4. Soup, rotated (data not in L1, code warm): trace 1 + k about 16000 to 22000 (harness 19637 plus the OptiX per query additive term if it is per trace, minus it if it is first call cost); gap to the harness 0.85 to 1.15 if first call cost.
5. Trace 3 equal to trace 2 within 5 percent.

### Test B: the harness group only, the same scenes (clock around the group, one query, the existing --clock build)
Predicted (the Latency_2 numbers measured on this GPU in F176): 1630 / 1608 / 1934 / 1982 / 2281 / 2270 / 2584, each within 3 percent, soup first query only 6000 (4000 to 9000). This gives loop span minus group span: the harness loop overhead per ray (predicted 590, range 400 to 800, constant over the scenes with one round).

### Reading after A and B (written now)
- Trace 2 gap to the harness within 10 percent on soup (rotated and plain), tri1 at most 0.5 to 1.1: the harness constants represent OptiX at one warp in steady state; the 4.5k belongs to the first call of a launch (not in the model, not in a persistent kernel); the 1.6 times is concurrency.
- Trace 2 still 4000 or more above the harness: the OptiX per trace path adds a fixed cost per query that the harness lacks (candidate terms: closest hit call, payload and stack handling in the SM, the instance node and SBT loads); then the model needs that term and the whole GPU 1.6 times may partly be there.
