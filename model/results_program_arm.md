# Program arm: is the Mochi frame 0 excess the program or the instance level? (F171; form, predictions, rule and the two source checks: model/FORM_program_arm.md, committed before the run)

2026-11-16, triton GPU 1, OptiX only (ttu/bench/Load_1 load3.cu, load3_dev.cu; raw private in notes/q30). The real Mochi dataset in Mochi's Morton order, 100,000 point queries, GAS built with the update flag only, span from the raygen timers (launch only: the refit and the update kernel are outside it). Timer step 1.024 us.

| arm | program | instance level | span us (median of 20) | cold us | calls per query | warp cycles mean |
|---|---|---|---|---|---|---|
| A | trivial | no | 11.26 | 12.29 | 1.075 | 13015 |
| B | Mochi | no | 12.29 | 13.31 | 1.075 | 14056 |
| C | trivial | yes | 10.24 | 10.24 | 1.075 | 17177 |
| D | Mochi | yes | 11.26 | 12.29 | 1.075 | 19060 |

Mochi at frame 0 (F165, F168): 1.075 calls per query, 100.4 us span, 78,339 warp cycles per warp (loaded). The replica has the SAME calls per query (1.075).

Verdict by the rule set before the run: RULED OUT. D (Mochi program, instance level, the full replica) is 11.3 us, not at or above 70 and below the 30 us limit; the prediction for D (20 to 55) and for B (15 to 35) were too high, A (12 to 20) slightly low, C within range (A + 0 to 8: it came out 1 us lower). Shares: the Mochi program adds 1.0 us (A to B, equal to one timer step), the instance level removes 1.0 us (A to C), the interaction is zero (D = A). None of these is distinguishable from zero at the 1.024 us timer step. The instance level raises the warp latency by 32 percent (13,022 to 17,180 cycles) and the Mochi program by 8 percent (14,042), D 19,059 cycles against Mochi's loaded 78,339.

How Mochi's frame 0 tree is built (read from hostCode.cu and the OWL sources, not checked against a dumped GAS): particles sorted by Morton code, largest first; one custom primitive GAS with ALLOW_UPDATE only (replaces OWL's default PREFER_FAST_TRACE and ALLOW_COMPACTION; OWL may OR flags in, not confirmed), then an instance group (one identity instance, ALLOW_UPDATE) over it, traced against the instance group (single level instancing); then 200 warmup substeps (update kernel, refit of GAS and instance group, launch) before frame 0, and 400 substeps per frame with the same sequence.

What is ruled out for Mochi frames 0 to 4: the program (force gathers, payload pointer in local memory, particle loads, sqrt, atomics), the instance level, the update flag (F170), the Morton ordered dataset and its calls per query (identical at 1.075), tree quality of a fresh tree and of a lattice refit chain (F170), loaded traversal in general (F169). The 6.25 nodes per query of Mochi were not compared with the nodes of this tree (not counted here).

Still open (not run; cheapest first): (1) the launch sequence: Mochi runs an update kernel, two refits and a launch back to back with host synchronisation, so the traversal unit's and the caches' state at the start of every launch (F159 to F162: node state in L1, removed by fences and cache control) and the GPU clock state are different from 20 launches in a row; (2) the 200 warmup refits and the actual per frame tree (a refit chain on the real moving particles, not a lattice); (3) the query positions after the fall; (4) the clock (the loaded warp is 4 times slower in cycles, which is not a clock effect if the cycles are SM cycles).
