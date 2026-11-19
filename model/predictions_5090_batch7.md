# RTX 5090 (pc14), session 7 predictions, 2026-11-19. Written BEFORE any run of the items below (plain CUDA and OptiX only, no TTU code).

Disclosure: I have seen the session 6 5090 numbers (L2 requested bytes, lts pct of peak, round times) and the 4090 results (F418, F419, model/results_4090_batch6.md, the second ncu metric list in ttu/bench/Phase10f/ncu_4090_batch6b.sh). The predictions below use them; item 1 is the replication of the 4090's second ncu run, item 2 is new.

## 1. Write share of the L2 traffic of the kNN builds on the 5090 (tethys's second metric list, per round)
Same builds (thirdparty/div1_sm120/knn_orig, knn_reg), same input, ncu --clock-control none --cache-control none, the 4090 metric list unchanged (lts__t_sectors_op_read/write/atom/red, lts__t_requests_srcunit_tex_op_read/write, l1tex__t_sector_hit_rate, lts__t_sectors_srcunit_tex_lookup_hit/miss, lts__t_bytes, lts__throughput, gpu__time_duration). 4090 values (round 4; rounds 1 to 3 similar): original 71 percent writes (68, 70, 75), register heap 38.5 (27, 29, 39); sectors per microsecond per percent of L2 peak 466 to 507 (original) and 757 to 916 (register heap), ratio 1.57 to 1.84.
Predictions (5090):
1. Write share of the original at least 60 percent in every round and within 8 points of the 4090's per round value: 60 percent (the sector counts are mostly set by the code; the L1 differs a little).
2. Write share of the register heap between 20 and 45 percent in every round: 70 percent.
3. No atomics or reductions: 90 percent.
4. Sectors per microsecond per percent of peak, register heap over original, round 1 to 4 each between 1.15 and 1.45 (below the 4090's 1.57 to 1.84): 60 percent. (Disclosure: from the session 6 bytes the round 4 ratio is about 1.27; this is not independent for that round.)
5. L1 sector hit rate of the original above the register heap's (as on the 4090): 65 percent.
6. The two sector based rates agree with the byte based numbers of F417 (L2 MB = sectors x 32 within 3 percent): 85 percent.

## 2. L2 throughput for read only, write only and mixed streams (plain CUDA, new microbenchmark ttu/bench/pc14/l2stream.cu)
Design: grid of 3 blocks per SM x 512 threads, 16 byte vector accesses with L1 bypass (ld.global.cg, st.global.cg), 32 lanes cover 512 contiguous bytes (whole sectors), one untimed warm up launch then timed launches of 2 to 6 ms (passes over the footprint repeated inside the kernel). Modes: read only (R), write only (W), 1 : 1 copy (read A, write B, each half the footprint), 2 : 1 (two reads, one write). Footprints (total bytes touched) 2, 8, 24, 48, 72 MB (fit the 96 MB L2), control 256 MB (spills to DRAM). Reported: sectors (32 B) per SM clock cycle for the whole chip (cycles from clock64 against globaltimer per block: the real clock), TB/s, and with ncu (one footprint, 24 MB, timed launch only): lts__t_sectors_op_read/write, lts__throughput pct of peak.
Predictions:
1. Read only at 24 MB: between 3.3 and 5.8 TB/s requested (40 to 70 sectors per SM cycle at 2,595 MHz): 65 percent.
2. Write only rate in sectors per cycle at most 1.0 and at least 0.3 of read only: 70 percent; between 0.3 and 0.6 (the cost implied by the two kNN builds: peak percent per sector linear in the write share gives write about 2.3 times a read, i.e. 0.43): 45 percent; below 0.2 (the 4090 fit said 6.8 times): 12 percent.
3. 1 : 1 mixed between the write only and the read only rate: 80 percent; 2 : 1 between the 1 : 1 and the read only rate: 70 percent.
4. Rates within 15 percent of each other for footprints 8 to 72 MB (same mode): 60 percent; 2 MB lower than the 8 MB value by more than 15 percent (fewer slices touched or ramp): 30 percent.
5. Control 256 MB: read only at most 0.5 of the 24 MB read rate: 85 percent; write only at most 0.5: 70 percent.
6. ncu: the read only stream reaches at least 85 percent of the L2 peak (lts__throughput): 65 percent; write only at least 85: 50 percent. Sector counts equal bytes / 32 within 3 percent: 85 percent.

## 3. RTNN callback calibration (as the model defines it: F216 b, predictions_phase13 variant S)
Stub build: the one line of triton's rt_stub tree (the radius intersection program returns at once, same traversal and program invocations) applied to the sm_120 RTNN tree (thirdparty/div1_sm120/rtnn_stub), real and stub on u100000, -p 0 -s 0 -m 0 -k 50, r 0.05 and 0.025, 3 repetitions each, GPU kernel time from the program's CUDA event (KERNEL line; the pre launch gap is the same in both and cancels in the difference), then ncu kernel durations of the stub (one run per radius) for the final time. cb = (T_real - T_stub) x SM x CLK / (N x 1.10 x C_mean) with SM 170, CLK 2.5e9 (the frozen constant), N 100,000, C_mean the replay mean of points in the cube (recomputed on the CPU from the same file: 93.8 and 13.0 on the 4090's file; to be recomputed, not assumed), cb floored at 0, (4.9 + cb - 1.011) floored at 0; calibrated T = (110 + (4.9 + cb - cbref) q_mean) N / (SM x CLK) (and the chain term, q_max x 575 / CLK, as in v1 where it can bind). Calibrated on the same input: NOT blind, labelled calibrated.
Predictions:
1. Stub kernel time over real, r 0.05: between 0.35 and 0.8: 70 percent; r 0.025: between 0.4 and 0.9: 60 percent. Stub runs rc 0: 90 percent.
2. cb at r 0.05 between 3 and 15 cycles per candidate (F216 triton: 7 to 11 percent of time was callbacks on real data; here the miss is 3.3 times so I expect a larger cb): 60 percent.
3. Calibrated prediction against the kernel time: inside a factor 2 at r 0.05: 65 percent; at r 0.025: 45 percent; both: 35 percent. Measured above the calibrated prediction in both: 70 percent.
4. F415 rescored with the calibrated RTNN rows against kernel time: RTNN rows enter the factor 2 in both: 35 percent; headline then 7 of 7: 35 percent (the five other headline rows are as in F415; the calibration uses a measurement of this program on this input, so the rows are labelled calibrated).
