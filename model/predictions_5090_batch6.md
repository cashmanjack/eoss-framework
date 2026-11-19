# RTX 5090 (pc14), session 6 predictions, 2026-11-19. Written BEFORE any run of the tests below (ncu on the RTNN and kNN builds, nothing else).

## A. RTNN: is the 4x miss (F255 4090 3.9 / 4.06, F415 5090 4.01 / 4.27) a scope mismatch?

Reading of existing raw files (done before this file, not a prediction): the scored number is "time search compute". On the 5090 the CUDA event interval printed beside it (KERNEL queries) is 0.065 ms shorter (0.577 against 0.512 at r 0.05; 0.188 against 0.132 at r 0.025): the host part that F220 found on triton (allocating and filling the output buffer, 50 slots x 4 bytes per query = 20 MB here, F220: 3.4 to 4.0 us per MB, this run 3.2). Not all host time is outside the event though: the event is recorded BEFORE launchSubframe, which prints, cudaMallocs the parameter block and copies it, so a GPU idle gap sits inside the event interval. The cleanest launch time is the GPU kernel duration (ncu gpu__time_duration of the traversal launch). The 4090 raw files are on tethys only: for the 4090 only an estimate is possible here (the same 20 MB output buffer: 0.06 to 0.08 ms subtracted from 0.7453 and 0.2355).

Test: ncu (non admin works here), gpu__time_duration.sum for every kernel of one RTNN run per radius (flags as the scored runs), picked by grid size (100,000 rays). GPU idle checked before every process, timeout -k 2 120 (ncu replays), OptiX only.
Predictions (ratio = measured kernel duration / prediction 0.144 ms (r 0.05) and 0.044 ms (r 0.025)):
1. The kernel duration is below the event time by 0.01 to 0.15 ms at both radii: 65 percent.
2. r 0.05 ratio between 2.5 and 4.0: 60 percent; r 0.025 ratio between 2.0 and 4.0: 60 percent; both inside a factor 2: 20 percent.
3. Rule: if both are inside a factor 2, F255 and F415 are corrected as a scope mismatch. Otherwise they stay, with the host part (0.065 to 0.07 ms, 11 to 35 percent of the scored time) added as a partial scope note, and the 4x is called launch time that the form misses.

## B. Register heap speedup 3.08 here against 4.0 to 4.45 on Ada: per round breakdown first (existing numbers, no GPU)
Rounds 1 to 4, time x SM count (SM microseconds per round, lower is faster per SM), orig / register heap:
- 4070 Ti (60 SMs, F138): 39.7 / 9.9, 58.7 / 13.5, 210 / 41.5, 661 / 176
- 4090 (128 SMs, F255): 54.7 / 14.7, 78.1 / 17.0, 276.6 / 49.8, 851 / 201.7
- 5090 (170 SMs, F415): 38.2 / 15.0, 51.8 / 18.3, 169 / 45.3, 534 / 179
Per round speedup orig / reg: 4070 Ti 4.0, 4.35, 5.06, 3.75; 4090 3.72, 4.59, 5.55, 4.22; 5090 2.55, 2.83, 3.73, 2.98.
Reading: the register heap costs about the same per SM on all three chips (rounds 3 and 4: 41 to 50 and 176 to 202); the ORIGINAL costs 1.3 to 1.6 times less per SM on the 5090 than on the 4090 (169 against 277, 534 against 851), and 0.8 times the 4070 Ti. So the lower speedup comes from the original getting faster, not the register heap getting slower. The original keeps its heap in global memory (25.6 MB, 128 B per query, loaded and stored in an insertion loop).
Hypothesis H1: the original is limited by the memory system (L2 and L1 traffic of the heap), the register heap by the SM instruction issue; the 5090's L2 and memory bandwidth per SM are higher than Ada's, so only the original gains.
Test: ncu on the 5090 of knn_orig and knn_reg (one run each, the 5 launches), per round: gpu__time_duration, lts__t_bytes (L2 traffic), dram__bytes (read plus write), smsp__inst_executed (warp instructions), lts__throughput and l1tex__throughput pct of peak, dominant warp stall (long scoreboard share). Round 4 is the main comparison (200,000 active).
Predictions (orig / register heap, round 4):
1. L2 bytes ratio orig / reg at least 2: 70 percent.
2. Warp instructions ratio orig / reg at least 1.3: 60 percent.
3. L2 throughput (pct of peak) of orig above that of the register heap: 70 percent; orig above 40 percent: 45 percent.
4. DRAM bytes below 20 percent of the DRAM peak for both (data fits the 96 MB L2): 75 percent.
5. Long scoreboard (global memory wait) share of the stalls: orig at least 50 percent: 65 percent; the register heap's share lower than the original's: 70 percent.
6. The round kernel durations under ncu match the event times of F415 within 25 percent: 70 percent.
Reading rule: 1 and 3 hold = the original's cost sits in the memory system (H1 supported); if the register heap's L2 throughput is as high as the original's, H1 fails and the cause is elsewhere (stated then). The Ada side has no ncu data on this machine, so the cross chip step stays a timing argument (per SM times), not a counter comparison; no claim about why the 4090 original was slow is made.
