# RTX 5090 (pc14): model with chip measured inputs, and a correction to the Throughput_1 saturated rate (2026-11-19)

POST HOC. Both parts below use RT measurements that already existed (model/results_5090_core.md), so none of it is a pre registered prediction. Script: model/model_5090_chip_inputs.py (the form was written into the file before any score was computed; no variant was tried and dropped).

## 1. Latency model with the 5090's own plain CUDA latencies (no RT data as input)
Form. The Ada model has, per scene: S the in kernel second trace (unit and L1 only: 782 / 827 / 868 cycles at 1 / 2 / 3 levels, 910 assumed at 4 levels), W the warm launch and C the cold launch (after a 256 MB read). Per scene, with the Ada point values L1 41, L2 290, DRAM 650:
  k_L2 = (W_ada minus S) / (L2_ada minus L1_ada), k_D = (C_ada minus W_ada) / (DRAM_ada minus L2_ada), both from Ada numbers only.
  5090: W = S + k_L2 x (L2 minus L1), C = W + k_D x (DRAM minus L2), with the 5090 plain chase medians L1 45.8, L2 349.3, DRAM 862 (F403, one thread, no RT). S stays the Ada constant (the 5090's own second trace, 790, is not used).
The ratios that matter: (L2 minus L1) is 1.219 times the Ada one, (DRAM minus L2) 1.424 times.

| Scene | k_L2 | k_DRAM | Warm pred | Warm meas | err | in 10% | Cold pred | Cold meas | err | in 15% |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 triangle | 3.29 | 2.64 | 1,779 | 1,788 | +0.5% | yes | 3,132 | 3,306 | +5.6% | yes |
| grid 12 | 3.49 | 2.86 | 1,840 | 1,791 | -2.7% | yes | 3,307 | 3,385 | +2.4% | yes |
| grid 144 | 4.31 | 3.89 | 2,135 | 2,247 | +5.3% | yes | 4,129 | 4,229 | +2.4% | yes |
| grid 200 | 4.39 | 3.78 | 2,159 | 2,177 | +0.8% | yes | 4,096 | 4,141 | +1.1% | yes |
| grid 2,500 | 5.55 | 4.56 | 2,552 | 2,551 | -0.1% | yes | 4,888 | 5,137 | +5.1% | yes |
| grid 4,096 | 5.55 | 4.86 | 2,552 | 2,571 | +0.7% | yes | 5,045 | 5,289 | +4.8% | yes |
| grid 20,736 | 6.71 | 5.72 | 2,946 | 2,994 | +1.6% | yes | 5,879 | 6,009 | +2.2% | yes |
Warm 7 of 7 inside 10 percent (before: 1 of 7 inside plus the item 2 range). Cold 7 of 7 inside 15 percent (before: 0 of 7). Latency_6 (cold minus warm, one triangle): 1,381 predicted, 1,518 measured, +9.9 percent (before: 950 to 1,000, 55 percent off); inside 15 percent, just inside 10. Memory_5, trace on another TPC (equals the warm value of the scene): 1,779 / 2,135 / 2,552 predicted against 1,810 / 2,267 / 2,637 measured (+1.7, +6.2, +3.3 percent, all inside 10 percent; before: 2 of 3).
Do the misses close? Yes, for every row that has L2 or DRAM latency in it. Limits of this statement: (a) k_L2 and k_D come out between 2.6 and 6.7 per scene, so they are effective serial counts, not counts of fetches; the model is "the memory part of the latency scales with the chase latency", one ratio per level, with per scene weights from Ada. (b) The unit part is the Ada constant and it was already right (790 against 782). (c) The test is consistent with the two ratios alone explaining the earlier misses; it does not say why the L2 and DRAM chase latencies are 1.2 and 1.4 times the Ada ones. (d) Rows that are not latency (Throughput_1 steady rate, Throughput_6 long query rate, siblings) cannot be recomputed from L2 or DRAM latencies; see part 2 for the steady rate.

## 2. Correction: the Throughput_1 "saturated rate miss" was a metric mismatch
model/results_5090_core.md and F407 scored the steady state against 0.14 to 0.148 with the whole span rate (32 x warps over the span of the launch, ramp included): 0.0958, 0.1007, 0.1042 at 24, 28, 32 warps, "missed by 26 percent". The Ada number 0.14 to 0.148 is not a span rate. In model/results_4090.md (and the 4070 Ti results) it comes from warp completion gaps (rays per cycle = 32 divided by the gap between successive warp completions, over the second half of the completions of a launch: 216 to 225 cycles), while "the whole span rate, which includes the ramp, is 0.090 to 0.100". The same two statistics from the 5090 raw files (notes/pc14_core/ttu_20261119T113012, the same sweep_stats definitions):
| Warps | 20 | 24 | 28 | 32 |
|---|---|---|---|---|
| span rate (rays per cycle) | 0.0907 | 0.0958 | 0.1007 | 0.1042 |
| completion gap (cycles) | 200 to 203 | 197 | 201 | 203 |
| gap rate = 32 / gap | 0.158 to 0.160 | 0.162 | 0.159 | 0.158 |
Like for like: gap rate 0.158 to 0.162 on the 5090 against 0.142 to 0.148 on Ada (+7 to +14 percent, and inside 10 percent of the predicted range top 0.148: the prediction 0.14 to 0.148 held); span rate 0.091 to 0.104 against 0.090 to 0.100 on Ada (0 to +4 percent). The 5090 saturates 7 to 14 percent ABOVE Ada, not 26 percent below. The long query rate (0.0277 against 0.0178, +56 percent, F407) is a different scene (par:64:s) and a real difference: the new RT core is faster on queries that spend their time in the unit, and about 10 percent faster on one triangle queries.
Checks I made for the supposed mechanism (issue, occupancy, placement): placement: all warps are on one SM (SM filter, one block, 32 warps); occupancy: 32 warps resident at 64 registers is the full register file (1,024 x 64), and the gap does not change from 20 to 32 warps, so there is no occupancy cliff; issue: not measured (no counters), and not needed because the numbers agree once the same statistic is used. The ramp part (span minus steady) is the same shape on both chips. Open: the 7 to 14 percent is a ratio of two different harness builds on two chips; not tested whether it comes from the unit or from the SM side.
