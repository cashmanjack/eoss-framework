# Phase 10g (tethys, 2026-11-18): short-query sibling pair on the 4090, predicted before the run

Gap named in model/results_4090.md row 6 ("short query version: not run"). Run: ttu/bench/io_core/core_set.py WORK thr6 (scene tri1, coherent ray, 32 warps; blocks G = 1, 2, 4, 8, 16, 24, 63, 64, 65, 79, 128; same checked group, stage 0 first, run.py timeout 30, idle GPU), nothing else.
References: 4070 Ti G 30 / 60 = 30.0 / 41.3 times one SM (F105, pair adds 38 percent); RTX 6000 Ada pair 1.53 times (F105 text); 4090 long queries 64.6 times at G 128 and 63.5 at G 64 (F176).
Predictions (same die as the 6000 Ada, 128 SMs = 64 TPCs):
1. G 64 (one SM per TPC) about 64 times one SM (60 to 66).
2. G 128 (both SMs of every TPC) 1.3 to 1.6 times the G 64 value (center 1.45), i.e. 83 to 102 times one SM. Probability 70 percent. A pair adding under 1.15 would mean the 4090 unit saturates on short queries already (like long queries); over 1.7 would mean the sibling SM is nearly independent.
3. Rate of one SM about 0.10 rays per cycle (0.08 to 0.12) and of a TPC pair about 0.14 (0.12 to 0.17), as the 4070 Ti (F105).
No tuning after the result. Faults, a wrong result or a hang: stop, report, no retry.

## Addendum 1 (written after the first run, before the second; GPU 1)
First run (core_set thr6, notes/p10g/thr6_gpu1_*): 0 wrong lanes, no fault. G 64 was 63.5 times one SM (prediction 1 held, 60 to 66; per SM rate 0.0997 vs 0.1005 alone, prediction 3 first part held). G 128 gave 116.5 times, a ratio of 1.83 to G 64, but the tool marks the aggregate invalid: the blocks overlapped in 0 of 5 launches (a 1,024 ray block ends before all 128 blocks have started; on the 4070 Ti G 60 overlapped 5 of 5). So prediction 2 is NOT scored from this run. Per block rate at G 128 was 0.91 of alone (a lower bound on the contention).
Second run: the persistent loop tool ttu/bench/Throughput_6L/long_run.py (checked builds, stage 0 first, results vs OptiX and a CPU reference), scene tri1, coherent ray, G = 1, 64, 128, R = 200 rounds, 6 launches (milliseconds long, so all blocks overlap), the ratio (rays per us at G 128) / (rays per us at G 64) is prediction 2 again with the same range 1.3 to 1.6 and the same decision rules. No other change.
