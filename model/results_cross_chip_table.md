# Cross chip table of the headline microbenchmark numbers (2026-11-18, from existing Findings; one new run, Phase 10g)

Chips: 6000 Ada = RTX 6000 Ada (triton, 142 SMs = 71 TPCs, 96 MB L2); 4070 Ti (io, 60 SMs = 30 TPCs, 48 MB L2); 4090 (tethys, 128 SMs = 64 TPCs, 72 MB L2). "n m" = not measured on that chip, "?" = the Finding text I read does not name the chip. Cycles are SM cycles. Every value is from the Finding named in the last column; nothing was re derived. The 6000 Ada and the 4090 are both AD102, the 4070 Ti is AD104 (so "second chip" is a second part of one family, not a second architecture).

| Group | Metric | 6000 Ada | 4070 Ti | 4090 | Findings |
|---|---|---|---|---|---|
| Cost | L1 hit, L2 hit, DRAM chase (cycles) | 41, 285 to 352 (ramp), n/a | 41, 285, about 655 | 41, 285 to 295 up to 70 MB, 620 | F104, F176, F202, F203 |
| Cost | second trace of one ray, 1 / 2 / 3 levels | 780 / 830 / 870 | 782 / 827 / 867 | 782 (1 level; 827, 867 with the sibling run) | F25, F104, F176 |
| Cost | one triangle, L2 warm / cold | 1,641 / 2,632 | 1,612 / 2,593 | 1,630 / 2,627 | F104, F176 |
| Cost | regression: 934 + 343 node + 113 leaf + 52 triangle (cold launch) | 934 / 343 / 113 / 52 | n m | n m | F67, F163 |
| Cost | saturated unit: triangle 0.67, node 1.3, leaf slot 0 (cycles) | n m | n m | 0.67 / 1.3 / 0 | F194 |
| Cost | one SM steady rate, short coherent queries (rays per cycle) | n m | 0.10 | 0.143 to 0.148 (core set steady state, F176); 0.100 tri1 (Phase 10g) | F105, F176, Phase 10g |
| TPC sharing | sibling pair over one SM, long coherent queries | 1.04 | 1.03 to 1.05 | 1.02 to 1.05 | F62, F104, F176 |
| TPC sharing | sibling pair over one SM, short (1 triangle) queries | 1.53 | 1.38 | GAP: first run invalid (overlap 0 of 5 at G 128); per block rate 0.91 of alone only bounds it; open | F105, F27, Phase 10g |
| TPC sharing | other TPC pair | 2.0 | 1.99 to 2.01 | 2.0 | F27, F104, F176 |
| TPC sharing | whole chip, long coherent queries (times one SM) | 70.6 at 71 SMs | 29.8 at 30, 31.1 at 60 | 63.5 at 64 (one per TPC), 64.6 at 128 | F62, F104, F176 |
| TPC sharing | whole chip, one triangle queries (times one SM) | n m | 30.0 at 30, 41.3 at 60 | 63.5 at 64; 128 invalid | F105, Phase 10g |
| Handoff | one warp, one custom primitive callback (cycles) | 850 to 900 (about 875) | n m | n m | F49 |
| Handoff | loaded callback latency over alone, four programs | n m | 1.3 to 16 times | n m | F133 |
| Handoff | register heap speedup of the any hit kNN | n m | 4.0 | 4.45 | F138, F255 |
| Handoff | built in sphere against custom primitive | n m | n m | same cost; tessellated 20 or 80 triangles no speedup | F204 |
| Handoff | opacity micromap: any hit calls kept at levels 2, 4, 6, 8 | n m | n m | 0.55, 0.14, 0.04, 0.01 | F228 |
| Handoff | partial warp cost per ray, 16 / 8 / 1 lanes over 32 | n m | n m | 1.03 to 1.05 / 1.15 / 5.4 | F230 |
| Handoff | idle lanes cost unit time (1 active lane runs 2.3 times as many warp queries) | 6000 Ada (F44, chip not stated in my excerpt, triton era) | n m | n m | F44 (F230 for the 4090 form) |
| Node state | second session saving, flat up to about 130 KB | 0.53 to 0.64 (0.62 at 68 KB, 0.54 at 132) | node cache half lost near 100 to 115 KB (F104) | 0.58 to 0.63, 0.54 at 132 | F150, F104, F256 |
| Node state | saving at 199 / 263 KB, then 4 MB | 0.23 / 0.23, 0.00 | n m | 0.23 / 0.23, 0.00 | F150, F256 |
| Node state | TTUCCTL between sessions | removes all, 22 sizes | n m | removes all, 11 sizes | F150, F256 |
| Node state | gpu fence on sibling / on own SM / both (68, 132 KB) | 0.26, 0.23 / 0.26, 0.25 / 0.00 | n m | 0.26, 0.22 / 0.26, 0.25 / 0.00 | F159, F256 |
| Node state | TTUCCTL alone on the sibling SM | 0.00 / 0.00 | n m | 0.00 / 0.00 (GPU 1; GPU 0 the same) | F161, F256 |
| Node state | TTUCCTL on the own SM | 0.00 / 0.00 | n m | 0.00 / 0.00 | F162, F256 |
| Node state | 1 MB own L1 stream against wait only | 0.29, 0.27 against 0.53, 0.42 | n m | 0.27, 0.27 against 0.57, 0.45 | F174, F256 |
| Caches | L1 about 90 KB falling to about 20 KB with the shared memory reservation | yes (F203) | yes (F104) | yes (F176, F202) | F104, F176, F202, F203 |
| Caches | L1 sector 32 B, tag line 128 B, L2 capacity ramp at 72 to 76 MB | repeats the 4090 within a few percent | n m | measured (the ramp is later and wider than modelled, 354 / 451 / 572 cycles at 72 / 74 / 76 MB) | F202, F203, F176 |
| Request merging | raw lane fetches per merged fetch, RT-BarnesHut | n m | 7.3 (sectors per traversal 3.17 and 4.39 measured) | n m | F135 |
| Request merging | model with held out error (reordering rays) | 0.08 rms log, chip: check F123 | n m | n m | F123 |
| Request merging | knee at about 10 distinct node lines per warp step | n m | n m | located on two new scene sizes, wrong on one regime | F201 |
| Flags and masks | ray flags through our own sequence | n m | n m | cull flags and enforce any hit as OptiX | F190 |
| Flags and masks | instance mask AND ray mask in the unit, zero mask rejected | n m | n m | yes (0F, FF, 01 hit; F0, 10, 00 miss against 0F) | F195, F192 |
| Flags and masks | instance transform from the leaf, 30 launches bit identical | n m | n m | 30 of 30 | F253 |
| Sibling slowdown | stream on siblings, random rays: RT speed against alone | 0.022 | n m | 0.022 (0.135 clock matched, other TPCs 0.147) | F101, F229, F244 |
| Sibling slowdown | coherent rays, siblings against other TPCs | n m | n m | 0.46 against 0.28 | F229 |
| Sibling slowdown | stream footprint 40 MB to 512 MB, siblings, unthrottled | n m | n m | 0.154 to 0.024 | F254 |
| Sibling slowdown | siblings, 64 MB footprint, all L2 hits, 1 / 2 / 4 / 8 / 16 / 32 pages per warp load (RT speed against alone) | n m | n m | 0.203, 0.192, 0.145, 0.121, 0.103, 0.091 | F262 |
| Sibling slowdown | stream hit rate alone by ncu at 40 / 72 / 96 / 120 / 512 / 1,024 MB | n m | n m | 1.00 / 0.98 / 0.60 / 0.49 / 0.22 / 0.23 | F262 |
| Request merging | cost follows distinct node paths: same ray moved by 1e-6 costs the identical lane floor at every D; distinct paths 85 cycles per ray at D 32 | n m | n m | floor 18 and 20.4 (100,000 and 1M triangles), 84 at D 32 | F263, F199 |
| Node state | wait without loads between sessions, 68 / 132 KB, saving after 10 / 70 / 300 us (no wait: 0.62 / 0.54) | n m | n m | 0.57, 0.58, 0.57 / 0.46, 0.46, 0.46 | F265 |
| Node state | 70 us wait loss, F175 (68 / 132 KB) | 0.08 / 0.15 (F175, F256 as quoted) | n m | 0.04 / 0.08 (F265; differs from F256) | F175, F256, F265 |
| Encodings | canonical TTU instructions byte identical SM89 and SM120 | 9 of 10 (TTULD.CLOSE close bit only) | n m | n m | F258 |
| BITTU | decode, 32 instances, counts exact; many warps and IAS group counts exact; timing correction fails with 2 or more warps | yes (1 warp checked group; many warps, IAS) | 4070 Ti (F119, F122) | n m | F260, F261 |
| Blind pipeline | calibrated launch time inside 2x: 1 of 24 cells, winner call 14 of 24 | 24 cells | n m | n m | F259 |
| TPC sharing | short query pair, same record loop build: one warp ran, final records disagree with OptiX; 64 and 128 block runs not done | n m | n m | GAP, still open (F264) | F264 |

## What is missing, and what I did about it
Counting the 36 rows: all three chips only for the cost (hits, one triangle, depth), TPC sharing (long queries, other TPC, whole chip) and cache rows; two chips for the node state rows (6000 Ada and 4090, which are the same die, plus F104's coarse 4070 Ti numbers) and the short query pair rows (4070 Ti and 6000 Ada); the callback, flag, mask, micromap, partial warp, sibling DRAM and instance rows are one chip each (mostly the 4090).
Filled on the 4090 this session: F161 (now 0.00 / 0.00 on the 4090, GPU 1) and the 4090 side of F150, F159, F162, F174 (F256) and F138 (F255).
Attempted and NOT filled: the short query sibling pair on the 4090 (Phase 10g): at G 64 63.5 times one SM held its prediction, at G 128 the blocks did not overlap, so the pair number is invalid; the persistent loop tool is memory bound and G 128 stopped at stage 0; no checked build with a small output exists. Open, needs a new checked loop build (my recommendation: build_patch --loop with the output store off, one hit counter per block).
Gaps not tried (cost beyond the plan): callback latency alone (875) on the 4090 and 4070 Ti; the cost regression (F67) on the 4090 and 4070 Ti; the request merging counters (F135) on the 6000 Ada and the 4090 (needs ncu source counters); the sibling DRAM stream on the 4070 Ti; flags and masks on the 4070 Ti and 6000 Ada; node state on the 4070 Ti. Each is one core set style run with predictions first; the callback latency and F67 are the two I would do first because they feed the model.

## Update 2026-11-19: F253 to F265 and the remaining gaps
Added rows for F258 to F265 above (F253 to F257 were already in the table). Counted from the table: 45 rows. F262, F263 and F265 are 4090 only; F259 to F261 are 6000 Ada only (triton); F258 is an encoding fact, not a chip row.
Remaining gaps, each one run on an idle GPU, predictions first:
1. Short query sibling pair on the 4090: the same record loop build ran one warp at stage 1 but its final record store disagrees with OptiX (F264); needs the store code read and one warp rerun, then G 64 and 128. Still the only gap that blocks a paper claim (TPC sharing).
2. 4070 Ti (io): node state rows, flags and masks, sibling DRAM stream, F262 page sweep, F263 node path test, F265 wait sweep, callback latency alone, cost regression F67, Cache_1 and Cache_2 (script ready), L2 points 40 to 56 MB, kNN clock sample.
3. 6000 Ada (triton): F262 to F265 (needs ncu for the hit rate, still admin only), the 4090 only rows F228, F230, F253 and the micromap and partial warp rows.
4. 4090: callback latency alone (875), cost regression F67, request merging counters F135, whole GPU short queries (blocked by gap 1).
5. F262 hit rate beside RT (measured alone), coherent rays in the page sweep.

