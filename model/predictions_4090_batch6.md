# RTX 4090 (tethys GPU 1), session 6 predictions, 2026-11-19. Written BEFORE any ncu run of the tests below. OptiX only (no TTU code of ours).

Follows the 5090's model/predictions_5090_batch6.md and model/results_5090_batch6.md (F416, F417). Same metrics, same builds as F255 (kNN original and register heap in /dev/shm/knnr/{orig,reg}, RTNN optixNSearch in ~/thirdparty/div1, same inputs and flags as Phase 10f). ncu with `--clock-control none --cache-control none`, TMPDIR in a directory we own, GPU 1 idle checked before every process, timeout -k 2 240, SM clock sampled beside every run.

## A. F416 on the 4090: RTNN split at r 0.05 and 0.025
Existing raw (tethys, notes/p10f/raw, read before this file, not a prediction): scored "time search compute" 0.7453 and 0.2355 ms (medians of 3); the program's own CUDA event (KERNEL queries line) is 0.638 and about 0.142 ms. So the host part outside the event is about 0.107 and 0.093 ms (larger than the 5090's 0.065 and 0.056). The GPU kernel duration (ncu, optixLaunch with 100,000 rays) is still unmeasured here.
Predictions (kernel in ms, ratio = kernel / form prediction 0.191 and 0.058):
1. Kernel below the event by 0.01 to 0.15 ms at both radii: 70 percent (5090: 0.033 and 0.034).
2. r 0.05 kernel 0.54 to 0.63 ms (ratio 2.8 to 3.3): 60 percent. r 0.025 kernel 0.085 to 0.125 ms (ratio 1.5 to 2.2): 60 percent.
3. Both ratios inside a factor 2: 30 percent (r 0.025 yes, r 0.05 no is the likely split: 55 percent).
4. Rescoring F255 against kernel time changes only the two RTNN rows (every other row has its own basis). Rule: rescored table reported next to the original; the original F255 count (10 of 15 inside a factor 2, headline 3 of 7) is not edited, the rescore gets its own line and a scope note. If r 0.025 enters the factor 2 then 11 of 15; the headline count stays 3 of 7 unless an RTNN row is a headline row (checked in model/score_phase10f.py when scoring).

## B. F417 on the 4090: kNN original against register heap, per round
Per round time x 128 SMs from F255 (SM microseconds orig / reg): 54.7 / 14.7, 78.1 / 17.0, 276.6 / 49.8, 851 / 201.7 (event speedups 3.72, 4.59, 5.55, 4.22). The 5090 had 3.0 to 4.0 in the kernel and in L2 bytes alike.
Metrics per round (launch of 200,000 active at round 1): gpu__time_duration.sum, lts__t_bytes.sum (L2 requested bytes), L2 request bandwidth = bytes / duration, dram__bytes.sum, smsp__inst_executed.sum, lts, l1tex, dram, sm throughput (pct of peak), lts__t_sector_hit_rate.pct, long scoreboard stall share.
Predictions (round 4 unless stated; ratio orig / register heap):
1. L2 byte ratio between 2.7 and 3.6 in every round (the byte counts are the program's, not the chip's): 75 percent. Same as the 5090 to within 10 percent: 65 percent.
2. Warp instruction ratio between 0.85 and 1.05 in every round (same as the 5090): 80 percent.
3. Kernel speedup (duration) per round: rounds 1 to 4 each between 3.4 and 5.8: 70 percent; total rounds 1 to 4 between 3.8 and 4.7: 65 percent (event speedup of F255: 4.02 to 4.45).
4. Kernel speedup EXCEEDS the L2 byte ratio by more than 15 percent in at least 3 of 4 rounds (the 5090 was 3 to 10 percent): 65 percent. This is the test of "the speedup matches the byte ratio here too"; matches (within 10 percent in every round): 20 percent.
5. L2 request bandwidth of the register heap is higher than that of the original in rounds 3 and 4 (original costs more per byte on Ada): 65 percent. The original's bandwidth (TB/s) is below the 5090's 3.07: 90 percent (the 4090 has fewer SMs and slices); lts pct of peak orig between 50 and 85: 60 percent.
6. DRAM bytes below 20 percent of the DRAM peak for both builds (heap and BVH fit the 72 MB L2): 70 percent. L2 sector hit rate of the original below that of the register heap: 55 percent.
7. Long scoreboard stall share: orig at least 50 percent and above the register heap's: 75 percent.
8. ncu round kernel durations within 25 percent of the event times (printed LAUNCH ms): 70 percent.
Reading rules: (a) if 4 holds and 5 holds, the original costs more per L2 byte on Ada (the 4.0 to 4.5 against 3.08 is then mostly the original, not the byte ratio); say which of L2 hit rate, throughput share or stalls differs. (b) If 4 fails (speedup within 10 percent of the byte ratio) then the byte ratio explains Ada too, and the difference to Blackwell is the byte ratio itself (compare 6a: the byte counts) or the chip's L2 limit. (c) Any claim about Blackwell's own cause is not made (no 5090 access here); the comparison is against the 5090 numbers in model/results_5090_batch6.md.
Why Ada's 4.0 to 4.5 against 3.08: no prediction on a single cause. The candidates to look at are the original's cost per L2 byte (hit rate, store traffic, long scoreboard) and the chip's L2 request limit, as listed above.
