# Flags, masks and tmax: results, 2026-11-16 (tethys, GPU 1, RTX 4090, OptiX launches only)

Predictions: model/predictions_flags.md (committed 54d19d3 era, before any run of these tests, not edited). Program: ttu/bench/Flags_1 (flags.cu, flags_dev.cu, run.py), table: model/flags_analyze.py. Raw data private (notes/flags1/). No TTU sequence of ours was run in these tests. Every config: 32 ray probe launches (100 launches, 3 traces each: first call cold, 2nd and 3rd call warm), 20 timed launches of 262,144 rays, ncu on the 262,144 ray launch and on a 32 ray launch.

## Changes to the method found during the run (stated before the numbers)
- The RT unit L1 sector count (l1tex__t_sectors minus the load store pipe) is about 0 on this GPU for scenes where the rays share their nodes (layers, instances: 0.003 to 0.008 per ray); it works for random rays (soup: 81 per ray). For the coherent scenes the number that carries information is the L1 fill count of a one warp launch (xbar to L1 read sectors, "fills32"), plus instructions and the probe trace cycles.
- Raygen here has 4 payload values (HG_1 had 7): the per call constant is smaller (3,401 warm trace for a one triangle class scene against 4,750). Differences between configs are the quantity.
- CULL_BACK and CULL_FRONT together are documented as mutually exclusive in the OptiX header: my "both culls" configuration is invalid, its result (a hit returned, front flag 1) is undefined and not used. The prediction for it was my error.

## 1. Terminate on first hit
| scene | flag | trace cold | trace warm | ns per ray (big) | RT sectors per ray | thread instr per ray | hit fraction |
|---|---|---|---|---|---|---|---|
| layers 64 | none | 5357 | 3481 | 0.156 | 0.005 | 709 | 1.00 |
| layers 64 | terminate | 5268 | 3459 | 0.156 | 0.005 | 709 | 1.00 |
| soup 100,000 random rays | none | 24629 | 9126 | 0.508 | 80.9 | 711 | 0.829 |
| soup 100,000 random rays | terminate | 24188 | 8952 | 0.445 | 64.0 | 711 | 0.829 |
Layers: no effect at all (1.7 percent on the cold trace, same fills and instructions), and the first hit found was the nearest plane in every ray with and without the flag (the nearest first order already finds it). Soup: the unit's L1 requests fall 21 percent, kernel time per ray 12 percent, but one warp traces fall only 1.8 percent and the SM instruction count is identical with and without the flag. Predicted: trace fall 5 to 30 percent (miss on both scenes at one warp, 1.7 to 1.9 percent), sectors fall 20 to 60 percent (soup 21 percent: inside, at the low end), returned primitive nearest in under half of the rays with the flag (layers: no, all nearest; the prediction was wrong for this scene). Reading: the flag shortens the work in the unit (fewer node requests) but not the SM side loop in this OptiX build, and the saving shows in a loaded launch (12 percent) more than in a lone warp. This corrects the F56 sentence "terminate on first hit changes nothing inside the RT core": it changes the unit's L1 requests on random rays (21 percent).

## 2. Any hit program
Scene layers 16, an any hit program that ignores every hit (16 calls per ray).
| config | trace cold | trace warm | ns per ray | thread instr per ray | any hit calls per ray |
|---|---|---|---|---|---|
| no any hit program, geometry opaque (control) | 5222 | 3401 | 0.133 | 709 | 0 |
| any hit program, no flag | 31968 | 28439 | 0.812 | 4267 | 16 |
| ray flag DISABLE_ANYHIT | 5145 | 3401 | 0.156 | 709 | 0 |
| ray flag ENFORCE_ANYHIT | 31970 | 28439 | 0.820 | 4267 | 16 |
| geometry flag DISABLE_ANYHIT at build | 5145 | 3401 | 0.152 | 709 | 0 |
Per any hit call: (28439 - 3401) / 16 = 1,565 cycles warm (1,676 cold), 222 SM instructions per call (4267 - 709, 16 calls), 7.0 cycles per instruction. Predicted 850 (500 to 1,100): MISS (1.8 times above the range top), the F49 figure for a custom primitive does not carry over to a triangle any hit. At the whole GPU rate the 16 calls cost 0.656 ns per ray more (kernel time of 262,144 rays on 128 SMs, launch ramp included; the SM clock was not recorded, 2.5 to 2.8 GHz assumed), about 13 to 15 SM cycles per call per ray (420 to 470 per call per warp), so latency 1,565 against a throughput cost of about 450 per warp: the call is latency bound for one warp, throughput cheap when many warps overlap. DISABLE equals no program, ENFORCE equals no flag, the geometry flag equals the ray flag, all within 1.5 percent (as predicted, within 3).

## 3. Cull flags (layers 16, even planes front facing, odd back facing, nearest plane back facing)
| flag | trace cold | trace warm | primitive, front flag | fills (32 rays) |
|---|---|---|---|---|
| none | 5220 | 3401 | 13, back (nearest plane) | 49 |
| cull back | 5480 | 3458 | 12, front (second plane, nearest culled) | 53 |
| cull front | 5145 | 3401 | 13, back (as none) | 49 |
Predicted: cull back within -10 to +15 percent of none (+5.0 percent cold, +1.7 percent warm: inside), cull front within 5 percent (-1.4 percent: inside). A culled plane costs 4 more sectors and about 260 cycles cold, 57 warm for one skipped plane: it is skipped inside the unit without a return (consistent with F56). Both culls: invalid, see above.

## 4. Masked out instances (Feature_1)
64 instances of one one triangle GAS, ray through all, instance masks as in the predictions file.
| scene | ray mask | trace cold | trace warm | fills (32 rays) | thread instr per ray |
|---|---|---|---|---|---|
| C: only the farthest instance, nothing masked | 2 | 4632 | 2282 | 54 | 482 |
| A: 64 instances, 63 nearer ones masked out | 2 | 4889 | 2325 | 58 | 482 |
| D: 64 instances, none masked | 2 | 5616 | 2539 | 66 | 482 |
| B: 64 instances, mask 255, nearest hit | 255 | 5623 | 2539 | 66 | 482 |
Does a masked out instance still cost a node fetch? Almost none: 63 masked instances add 4 L1 fill sectors, 257 cycles cold and 43 warm over the one instance scene (predicted H1: +60 to +130 sectors, +4,000 to +10,000 cycles cold: both MISS, far below; H2 (a line per instance) is excluded by 60 times). Masked out instances are cheaper than visible ones (A below D by 727 cold cycles and 8 sectors): they are rejected before their subtree is read, so the rejection must happen at the interior IAS nodes (consistent with a mask summary per node; not decoded). Limit: in this layout the masked instances form a block by position, so a parent node can reject whole subtrees; a single masked out instance sitting between visible ones was not tested, and its cost may be a line.

## 5. tmax pruning on random rays (Feature_5), soup 100,000, 262,144 random rays
| tmax | RT sectors per ray | ns per ray | hit fraction (262,144 rays; corrected 2026-11-17: the first version of this column was the 32 ray probe launch) | trace cold (32 rays) | trace warm |
|---|---|---|---|---|---|
| 0 | 20.7 | 0.188 | 0.000 | 7455 | 3498 |
| 0.01 | 22.9 | 0.188 | 0.038 | 8273 | 4389 |
| 0.02 | 25.2 | 0.219 | 0.075 | 8754 | 4556 |
| 0.05 | 31.5 | 0.246 | 0.175 | 9464 | 4672 |
| 0.1 | 41.8 | 0.289 | 0.314 | 9752 | 4871 |
| 0.2 | 56.8 | 0.371 | 0.509 | 12626 | 5513 |
| 0.5 | 75.2 | 0.469 | 0.754 | 15676 | 6690 |
| 1 | 80.2 | 0.496 | 0.823 | 22456 | 8578 |
| 2 | 80.7 | 0.507 | 0.829 | 24080 | 9114 |
| unlimited | 80.8 | 0.508 | 0.829 | 24140 | 9130 |
Predicted sectors 12 / 20 / 30 / 45 / 65 / 80 / 90 (r 0 to 0.5) and equal to unlimited at r >= 1: measured inside the range at r = 0 to 0.02, then 30 to 36 percent below the prediction at 0.05 to 0.2 and 16 percent below at 0.5 (saturation 81, not 90; the rule "equal to unlimited at r >= 1 within 3 percent" held: 80.2 against 80.8). Hit fraction predicted 0.9 or more by r = 0.3: miss (0.81 at most; 0.53 at 0.2). Time per ray predicted 0.3 of unlimited at r = 0.01 and 0.75 at 0.2: 0.37 and 0.73 (inside). Saving at r = 0.05 against unlimited: sectors 61 percent, time 52 percent (predicted 40 to 65: inside). One warp trace at r = 0.05 against unlimited: 0.39 cold, 0.51 warm (predicted 0.5 to 0.8: cold below). A zero length ray still costs 20.7 sectors per ray and 37 percent of an unlimited ray's time (F44: an empty interval from inside the scene is not free; the lone warp warm trace at r = 0 is 38 percent of the unlimited one). At one warp the cold trace keeps growing after the per ray sectors have flattened (15676 at r = 0.5 to 22456 at r = 1 while sectors grow 6.6 percent); hypothesis, not tested: the warp waits for its slowest lane, whose chain is longer than the mean.

## Which of these change the traversal group itself
Not checked. F56 found that back face culling works through the ray flag words of the group, and Phase 9 says the store words must be read from a dump per flag before our own sequence uses them; I did not read any dump here. The any hit program is SM side only (a callback per hit, 222 instructions here). Our own sequence was not run for any of these tests.
