# RTX 5090 (tgrogers-pc14): the TTU part of the core set, scored against model/predictions_5090.md (2026-11-19)

Approval: Cesar approved TTU runs on this machine and Jack approved stage 1 on SM120 (2026-11-19). Every run: nvidia-smi checked immediately before (only Xorg, gnome-shell, the Steam helper), timeout -k 2 30, GPU 0 (the display GPU), the checked SM120 builds only (build_patch --sm 120, check_seq ALL PASS: close on bit 74, the group byte for byte from OptiX's own code), one block and one warp first, no fault, hang, timeout or wrong result in any run. SM clock sampled by nvidia-smi every 50 ms around every run: 2,595 MHz in every one of 9,005 samples (power 82 W median, 86 W max, 33 C), so cycle counts convert at 2,595 MHz (L1 hit 17.6 ns, L2 134 ns, DRAM 332 ns). No prediction was refitted. Raw files: notes/pc14_core (private).

## Gate: Latency_1, one warp, against OptiX on the same ray (passed)
clk_s1 and twice_s1, one block, one warp, one launch each. The result words matched OptiX and the CPU reference exactly: hit, t = 1 (bit exact), primitive 0, u = 0.25, v = 0.5, the back face flag (OptiX front = 0), the leaf block address = GAS + 0x300, the instance word echoed. The register layout was read from this first result (the SM120 destinations differ from the SM89 ones only in register numbers): R52 u (sign = back face), R53 v, R66 primitive with flag bits, R67 = -t, R39 hit flag, R62:R63 leaf address. So the roles were fitted to the first result and then used for every later check (every stage 1 run that stored result words was compared lane by lane with OptiX's hit, primitive and t: no difference). Second trace in the kernel: 790 cycles.

## Latency_1, 2, 6 (predictions items 2 to 4; warm = L2 warm, cold = after a 256 MB read; 30 launches, median of warp clock spans)
| Scene | Warm measured | Predicted | Score (10%) | Cold measured | Predicted | Score (15%) | Second trace |
|---|---|---|---|---|---|---|---|
| 1 triangle | 1,788 | 1,600 (item 3), 1,630 (item 2, range 1,500 to 1,900) | item 2 held, item 3 missed (+11.7%) | 3,306 | 2,550 | MISSED (+29.6%) | 790 (782 predicted, held) |
| grid 12 | 1,791 | 1,650 | held (+8.5%) | 3,385 | 2,680 | MISSED (+26.3%) | 790 |
| grid 144 | 2,247 | 1,900 | MISSED (+18.3%) | 4,229 | 3,300 | MISSED (+28.2%) | 841 |
| grid 200 | 2,177 | 1,920 | MISSED (+13.4%) | 4,141 | 3,280 | MISSED (+26.2%) | 838 |
| grid 2,500 | 2,551 | 2,250 | MISSED (+13.4%) | 5,137 | 3,890 | MISSED (+32.1%) | 884 |
| grid 4,096 | 2,571 | 2,250 | MISSED (+14.3%) | 5,289 | 4,000 | MISSED (+32.2%) | 884 |
| grid 20,736 | 2,994 | 2,580 | MISSED (+16.0%) | 6,009 | 4,640 | MISSED (+29.5%) | 939 |
Latency_6 (one triangle, cold minus warm): 1,518 measured against 950 to 1,000 predicted: MISSED. Reading: the second trace (everything cached, the unit and L1 only) matches the Ada value (790 against 782), the warm launch cost is 9 to 18 percent above it and the cold cost 26 to 32 percent above it: the part of the cost that is L2 and DRAM is larger on this chip (L2 349 and DRAM 862 cycles in the plain chase against the Ada 290 and 650, F403), the unit and L1 part is the same. A rough check: the cold minus warm difference over the predicted one, 1.55, is of the order of the DRAM chase ratio 862 / 650 = 1.33 plus the L2 ratio 349 / 290 = 1.20; not tested further.

## Throughput_1 (one SM, tri1 coherent, warm; rays per cycle), item 5

**CORRECTION 2026-11-19 (later, see model/results_5090_chip_inputs.md part 2):** the Throughput_1 steady state rows below compare the whole span rate (0.0958 to 0.1042) with the Ada number 0.14 to 0.148, which is a completion gap rate. Same statistic on the 5090: gap rate 0.158 to 0.162, so the prediction held (inside 10 percent of 0.148) and the 5090 saturates above Ada; the "MISSED" marks for 24 to 32 warps in this section and in the tally are withdrawn. The span rate 0.091 to 0.104 compares with Ada's 0.090 to 0.100 (held). Also: Latency_2/6 and Memory_5 misses are recomputed with the chip's own plain latencies in the same file (all rows close).
| Warps | 1 | 2 | 4 | 8 | 16 | 20 | 24 | 28 | 32 |
|---|---|---|---|---|---|---|---|---|---|
| Measured | 0.0179 | 0.0288 | 0.0440 | 0.0610 | 0.0854 | 0.0907 | 0.0958 | 0.1007 | 0.1042 |
| Predicted | 0.0195 | 0.031 | 0.0445 | 0.058 | 0.080 | 0.090 | 0.14 to 0.148 steady from 20 to 32 warps | | |
| Score | held (-8%) | held (-7%) | held (-1%) | held (+5%) | held (+7%) | held (+0.8%) | MISSED | MISSED | MISSED (-26% at 32) |
The curve up to 20 warps matches the Ada one; it does not reach the Ada steady state (0.14 to 0.148) but keeps rising slowly to 0.104 at 32 warps. The plain builds reach 20 warps; 24 to 32 warps use the remap layout (the two layouts agree at 20 warps: 0.0907 and 0.0907).

## Throughput_8 (TPC siblings, SM ids from the measured table; long coherent queries par:256:s, 20 warps, blocks pinned with the SM filter), item 6
| Pair of SMs | Rate A | Rate B | Pair over one SM (0.0060) | Overlap | Reading |
|---|---|---|---|---|---|
| 0,1 | 0.00329 | 0.00329 | 1.10 | 81 to 100% | siblings |
| 10,11 | 0.00330 | 0.00324 | 1.08 | 80 to 100% | siblings |
| 2,3 | 0.00325 | 0.00401 | 1.20 | 80 to 81% | siblings, overlap only 80%, not clean |
| 0,2 | 0.00602 | 0.00606 | 2.01 | 98 to 99% | different TPCs |
| 1,3 | 0.00602 | 0.00593 | 1.98 | 93 to 99% | different TPCs |
| 2,4 | 0.00604 | 0.00603 | 2.01 | 96 to 98% | different TPCs |
Predicted: siblings together 1.02 to 1.05 times one SM, different TPCs 2.0. Different TPCs held (1.98 to 2.01). Siblings 1.08 and 1.10 (the clean pairs) MISSED narrowly (above the range by 0.03 to 0.05; the 1.20 pair has only 80% overlap). The pairing of SM ids (2k, 2k+1) as one TPC is supported: those pairs share one unit's throughput, other pairs do not.

## Throughput_6 (long coherent queries par:64:s, 32 warps per SM, one block per TPC = SM 2k, then every SM), item 7
| G | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 85 (one per TPC) | 170 (every SM) |
|---|---|---|---|---|---|---|---|---|---|
| Rays per cycle per SM | 0.02772 | 0.02759 | 0.02763 | 0.02767 | 0.02766 | 0.02774 | 0.02771 | 0.02773 | 0.01450 |
| Aggregate over one SM | 1.0 | 1.99 | 3.99 | 7.98 | 15.96 | 32.0 | 63.75 | 84.61 | 88.02 |
Predicted: 84.5 at G = 85 (held, +0.1%) and 87.8 at G = 170 (held, +0.3%); per SM rate 0.0178 (MISSED, measured 0.0277, +56%) and 1.5 and 1.56 rays per cycle in total (MISSED: 2.35 at G = 85 and 2.44 at 170). Scaling is 99.5% efficient per TPC up to 85 TPCs, and the sibling SMs add 4% (88.02 / 84.61): the scaling prediction held, the absolute rate (the 4090 and 6000 Ada value 0.0178) missed: the 5090 does 0.0277 on this scene. Not refitted.

## Memory_5 (TPC sharing; second trace on the sibling right after its partner, SM pair 0,1; other TPC pair 0,2), item 10
| Scene | B on the sibling after A | B on another TPC after A | A alone |
|---|---|---|---|
| tri1 | 790 | 1,810 | 1,787 |
| grid 144 | 841 | 2,267 | 2,230 |
| grid 2,500 | 884 | 2,637 | 2,565 |
Predicted: a trace on the sibling right after its partner costs the same as a second trace on the partner (782 / 827 / 867): measured 790, 841, 884 equal the in kernel second trace of the same scenes to the cycle (790, 841, 884): held. On an SM in another TPC 1,570 to 2,385: 1,810 and 2,267 inside the range (held), 2,637 above it (MISSED). The other TPC value equals the alone value (no sharing across TPCs). The claim "about 50 percent of node fetches in the own L1 and 50 percent in the sibling's L1" needs counters and was not tested here.

## Memory_2 (node cache, refetch fraction by ncu; 16 warps, soup:100000, one ray pool; L1 about 90 KB), item 9
| Distinct rays | 1 | 4 | 16 | 32 | 48 | 64 | 96 | 128 | 192 | 384 |
|---|---|---|---|---|---|---|---|---|---|---|
| Footprint KB | 4.2 | 15.1 | 41.0 | 90.8 | 139.9 | 167.6 | 353 | 446 | 746 | 866 |
| Refetch (0 = all found again) | 0.00 | 0.00 | 0.09 | 0.45 | 0.88 | 0.77 | 0.76 | 0.81 | 0.89 | 1.06 |
Predicted: every node found again below about 60 KB, half lost at about 100 KB. Measured: nothing refetched up to 15 KB, 9 percent at 41 KB, half lost at about 99 KB (interpolated between 0.45 at 91 KB and 0.88 at 140 KB). The half point held; "every node" is nearly held (0.09 at 41 KB; no point between 41 and 91 KB). With the reserved shared memory (L1 about 20 KB) the refetch fraction is 0.56 at 4 KB and above 1 for every larger size (1.04 to 4.2; fills of the second session exceed the first session's). Second trace latency by rays (Memory_2 latency script, 16 warps): see notes (1,718 cycles at 1 ray to 34,160 at 512 rays, small L1 1.08 to 1.86 times the big one).
Memory_1 (L1 capacity, plain chase) is in model/results_5090_plain.md.

## Tally of this file's predictions (no refit)
Held: second trace 790, Latency_1 warm (item 2 range), grid 12 warm, Throughput_1 up to 20 warps (6 of 6), different TPCs 2.0, Throughput_6 scaling 84.5 and 87.8, Memory_5 sibling equals second trace (3 of 3) and two of three other TPC values, Memory_2 half point. Missed: all 7 cold latencies, 6 of 7 warm (Latency_2 list), Latency_6 difference, Throughput_1 steady state (3 points), siblings long queries (narrowly), Throughput_6 absolute rate (twice), Memory_5 other TPC at 2,500 triangles. The null hypothesis (Ada cycle constants inside one SM and its unit) holds for the unit and L1 behaviour and for the TPC structure, and fails for everything with L2 or DRAM in it and for the saturated unit rate.
