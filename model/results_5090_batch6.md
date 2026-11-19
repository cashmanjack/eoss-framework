# RTX 5090 (pc14) session 6: RTNN scope (F416) and the register heap speedup (F417), scored against model/predictions_5090_batch6.md (commit 9083baa, before the runs), 2026-11-19

Runs: ncu (non admin works here) on the OptiX only builds in thirdparty/div1_sm120, GPU check before each process, timeout -k 2 240, GPU 0, `--clock-control none --cache-control none` (the first RTNN pass with the ncu defaults locked the clock to base and flushed caches: 0.602 ms and 0.127 ms for the traversal launch, kept in notes/pc14_workloads/ncu_rtnn_r05_20261119T150928.csv and ncu_rtnn_r025_20261119T150935.csv, not used). Raw (private): notes/pc14_workloads/ncu_*. Scorer: model/score_5090_batch6.py. Runner: ttu/bench/pc14/ncu_workloads_pc14.sh. No fault.

## A. RTNN scope (F416)
| | r 0.05 | r 0.025 |
|---|---|---|
| scored "time search compute" (F415, median of 3) | 0.577 | 0.188 |
| CUDA event interval printed by the program | 0.512 | 0.132 |
| GPU kernel duration of the traversal launch (ncu, optixLaunch) | 0.479 | 0.098 |
| prediction (form, 170 SMs) | 0.144 | 0.044 |
| scored / predicted | 4.01 | 4.27 |
| kernel / predicted | 3.33 | 2.22 |
Where the scored time goes: host part outside the event 0.065 / 0.056 ms (allocating and filling the 20 MB output buffer: 3.2 us per MB; F220 on triton 3.4 to 4.0), a gap inside the event interval before the launch 0.033 / 0.034 ms (the event is recorded before launchSubframe, which prints, allocates and copies the parameter block), the kernel 0.479 / 0.098. The scored number is 83 and 52 percent kernel. So F220's scope mismatch is present on this input too but is small here (the output buffer is 20 MB, not 72 to 722 MB) and does not explain the miss.
Predictions: 1 (kernel below event by 0.01 to 0.15 ms): HELD (0.033, 0.034). 2 (r 0.05 between 2.5 and 4.0, r 0.025 between 2.0 and 4.0): HELD (3.33, 2.22); both inside a factor 2: NO (20 percent predicted). Rule 3: F255 and F415 are NOT corrected as a scope mismatch; they get a partial scope note.
4090: the raw KERNEL lines are on tethys only. Estimate from the same 20 MB output buffer (0.06 to 0.08 ms host part) and the same 0.03 ms gap: 0.745 becomes 0.63 to 0.65 ms (3.3 to 3.4 times the prediction 0.191) and 0.2355 becomes 0.125 to 0.145 (2.2 to 2.5 times 0.058). An estimate, not a rescoring; a rescoring needs the KERNEL lines (or an ncu run) from the 4090 (tethys, ncu non admin was not checked there).
Not explained: the remaining 2.2 to 3.3 times in the kernel (the form's 110 + 4.9 q cycles, F220 had real data launches within 0.6 to 1.43 of the form on triton with the recorder builds; this input is the uniform cube with -p 0 -s 0).

## B. Register heap speedup (F417)
Per round breakdown (time x SM count, SM microseconds; orig / register heap), from the existing numbers and this chip's:
| round | 4070 Ti (60) | 4090 (128) | 5090 (170) |
|---|---|---|---|
| 1 | 39.7 / 9.9 | 54.7 / 14.7 | 38.2 / 15.0 |
| 2 | 58.7 / 13.5 | 78.1 / 17.0 | 51.8 / 18.3 |
| 3 | 210 / 41.5 | 276.6 / 49.8 | 169 / 45.3 |
| 4 | 661 / 176 | 851 / 201.7 | 534 / 179 |
| speedup per round | 4.0, 4.35, 5.06, 3.75 | 3.72, 4.59, 5.55, 4.22 | 2.55, 2.83, 3.73, 2.98 |
The register heap costs about the same per SM on all three chips (rounds 3, 4: 41 to 50 and 176 to 202); the original costs 0.6 times (against the 4090) and 0.8 times (against the 4070 Ti) as much per SM on the 5090. So the lower speedup is the ORIGINAL getting faster per SM, not the register heap getting slower. (The 4090's kNN points are another file than F138's, F257.)
ncu on the 5090, kernel durations (ms), L2 requested bytes (MB), L2 throughput (pct of peak), stalls:
| round | orig ms | reg ms | speedup (kernel) | L2 MB orig / reg | L2 byte ratio | ratio of speedup to byte ratio | lts pct orig / reg | warp inst M orig / reg |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.183 | 0.061 | 2.99 | 576 / 199 | 2.90 | 1.03 | 78.1 / 60.5 | 11.9 / 11.7 |
| 2 | 0.286 | 0.086 | 3.34 | 870 / 274 | 3.17 | 1.05 | 75.3 / 56.7 | 22.5 / 22.4 |
| 3 | 0.982 | 0.246 | 4.00 | 3,033 / 832 | 3.64 | 1.10 | 80.1 / 65.3 | 74.9 / 82.5 |
| 4 | 3.108 | 1.044 | 2.98 | 9,557 / 3,527 | 2.71 | 1.10 | 78.6 / 67.7 | 347 / 386 |
Round 4 L2 byte rate: orig 3.07 TB/s, register heap 3.38 TB/s. DRAM below 8 percent of peak in every round of both (heap and BVH sit in the 96 MB L2). Long scoreboard stall share (pct of warps): orig 79 to 82, register heap 60 to 62. Kernel totals rounds 1 to 4: 4.56 against 1.44 ms, speedup 3.17 (the event times of F415 gave 3.08), so the speedup is not an event artifact.
Predictions: 1 (L2 bytes orig / reg at least 2): HELD (2.7 to 3.6). 2 (warp instructions orig / reg at least 1.3): MISSED (0.9 to 1.0: the register heap executes as many or more instructions). 3 (orig lts above reg's, orig above 40 percent): HELD (78.6 against 67.7). 4 (DRAM below 20 percent): HELD. 5 (long scoreboard orig at least 50 percent and above the register heap's): HELD (81 against 62). 6 (ncu durations within 25 percent of the event times): HELD for the original (0.82 to 0.99) and register heap rounds 2 to 4 (0.79 to 0.99); round 1 0.70 (MISSED by 5 points).
Reading: the reading rule said H1 fails if the register heap's L2 throughput were as high as the original's; it is 68 against 79 percent, close. On the 5090 BOTH builds are running near the L2 request limit (3.1 to 3.4 TB/s), and the kernel speedup equals the L2 byte ratio within 3 to 10 percent in every round. The register heap removes 62 to 72 percent of the original's L2 bytes and nothing else (same instruction count), so the gain is bounded by the byte ratio, 2.7 to 3.6. This explains 3.0 to 3.2 on the 5090. It does not explain Ada's 4.0 to 4.5: that needs the original to cost more per L2 byte on Ada (L2 bandwidth per SM, the original's store traffic, or LSU limits), which I cannot test here (no Ada ncu on this machine). Suggested: run ttu/bench/pc14/ncu_workloads_pc14.sh's metric list on io (4070 Ti, ncu non admin per CLAUDE.md) with the same builds recompiled for sm_89; prediction: the L2 byte ratio is the same (2.7 to 3.6) and the kernel speedup is above it. Not tested: why the original gets cheaper per SM on the 5090 than on Ada (L2 bandwidth per SM is a candidate), any statement about the 4090's own counters.
