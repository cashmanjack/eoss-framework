# Bounce ray oracle test on the 4070 Ti (io): prediction written 2026-11-13 BEFORE any run

Question: does the oracle grouping's gap over a plain origin Morton sort grow with 48 MB of L2 (4070 Ti, 60 SMs) against 96 MB (RTX 6000 Ada, 142 SMs)? Same scenes as Findings 88 and 96: soup scenes of 1e5, 3e6, 8e6 and 16e6 triangles (OptiX, closest hit, about 11 million bounce rays, 10 launches median, every order checked equal to pixel order). Real scenes (Finding 102) are not in /dev/shm on io and are not part of this run.

Reasoning. Triton (Findings 88, 96): oracle gain over pixel order 1.27, 1.27, 1.28, 1.28 at 1e5, 3e6, 8e6, 16e6 (scene 0.07, 1.9, 5.4, 10.8 times the L2); origin Morton 1.10 to 1.17; pixel order faster than shuffled by 2.1 to 2.5 times; net of sort costs 0.45 to 0.46 of pixel order. The gain did not grow from 1.9 to 10.8 times the L2, so I expect the smaller L2 (the same scenes are now 3.9, 10.8, 21.6 times the L2 for 3e6, 8e6, 16e6) to stay on the same plateau: the locality that grouping buys is mostly in the L1 and the node cache, and the L1 per SM is the same (128 KB). The 4070 Ti has about as much L2 and DRAM bandwidth per SM as the 6000 Ada (0.8 vs 0.68 MB, 8.4 vs 6.8 GB/s per SM), so per SM conditions are similar.

Predictions (speedup over pixel order, trace only):
1. Oracle (full key) 1.2 to 1.35 on every scene of 3e6 and larger; 1e5 (6.5 MB, fits the L2) 1.2 to 1.3.
2. Origin Morton (4 or 5 bits): 1.08 to 1.20.
3. Gap oracle / Morton: 1.07 to 1.17, same as on triton (1.10 to 1.15). It does NOT grow by more than 0.05 with the smaller L2. (If it grew, the sign would be the L2 mattering for grouping: that is the alternative I rate at about 25%.)
4. Shuffled: pixel order faster than shuffled by 2.0 to 3.0 times (a smaller L2 may make the shuffled penalty a little larger).
5. Absolute: pixel order trace about 2.4 times slower than on triton (60 vs 142 SMs): 5 to 7 ms for 11 million rays.
6. Net of costs (GPU keygen, sort, gather, scatter), best 4 bit Morton: 0.45 to 0.6 of pixel order speed (sort work is bandwidth bound, 504 vs 960 GB/s: about 1.9 times slower than triton's 3 ms, the trace is about 2.4 times slower, so the ratio moves up a little). Never above 1.
