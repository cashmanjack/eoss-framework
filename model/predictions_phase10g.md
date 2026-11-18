# Phase 10g (tethys, 2026-11-18): short-query sibling pair on the 4090, predicted before the run

Gap named in model/results_4090.md row 6 ("short query version: not run"). Run: ttu/bench/io_core/core_set.py WORK thr6 (scene tri1, coherent ray, 32 warps; blocks G = 1, 2, 4, 8, 16, 24, 63, 64, 65, 79, 128; same checked group, stage 0 first, run.py timeout 30, idle GPU), nothing else.
References: 4070 Ti G 30 / 60 = 30.0 / 41.3 times one SM (F105, pair adds 38 percent); RTX 6000 Ada pair 1.53 times (F105 text); 4090 long queries 64.6 times at G 128 and 63.5 at G 64 (F176).
Predictions (same die as the 6000 Ada, 128 SMs = 64 TPCs):
1. G 64 (one SM per TPC) about 64 times one SM (60 to 66).
2. G 128 (both SMs of every TPC) 1.3 to 1.6 times the G 64 value (center 1.45), i.e. 83 to 102 times one SM. Probability 70 percent. A pair adding under 1.15 would mean the 4090 unit saturates on short queries already (like long queries); over 1.7 would mean the sibling SM is nearly independent.
3. Rate of one SM about 0.10 rays per cycle (0.08 to 0.12) and of a TPC pair about 0.14 (0.12 to 0.17), as the 4070 Ti (F105).
No tuning after the result. Faults, a wrong result or a hang: stop, report, no retry.
