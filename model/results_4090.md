# Results on the RTX 4090 (tethys, GPU 1), 2026-11-16, against model/predictions_4090.md

The predictions file was committed before any run and is not edited. No refitting: every number below is the first and only run of the core set (one repeat of each step). Raw data stays private (notes/tethys/).

## Detected (inputs of the predictions table)
128 SMs, 64 TPCs, L2 72 MB (75,497,472 bytes), 384 bit memory, SM89, driver 580.178.04, max SM clock 3105 MHz. All as assumed. Placement rule (probe_placement.cu, plain CUDA): 0 mismatches in 5 launches. Driver 580.178.04 was already in validated_drivers (io); memdump on tethys (our own OptiX launch, no TTU code of ours): the GAS traversal group, its ISETP and all code from fnaddr-0x10000 to fnaddr-0xdca0 are byte identical to the earlier dump; 592 of 4096 instructions after that differ (data and pointer tables, as on io with 558). The IAS group was not rechecked (the core set uses the GAS).

## Item by item (match = within 10%; cold and DRAM within 15%)
| # | Prediction | Measured | Verdict |
|---|---|---|---|
| 1 | L1 hit 41 | 41.0 (16 KB), 40.2 (64 KB) | match |
| 1 | L2 hit 285 to 295, 1 to 64 MB, .cg, cross | 284.8 to 290.1; .cg 284.6; cross 283.8, 285.8 | match |
| 1 | ramp 68 MB 285 | 293.4 | match |
| 1 | ramp 70 MB 295 | 295.3 | match |
| 1 | ramp 72 MB 425 (340 to 520) | 354.3 (-17%) | inside the range, outside 10% |
| 1 | ramp 74 MB 554 | 450.5 (-19%) | miss |
| 1 | ramp 76 MB 658 | 572.4 (-13%) | miss |
| 1 | DRAM level 96, 128 MB: 658 | 620.5, 621.2 (-6%) | match |
| 1 | DRAM 512 MB 650 (560 to 780); cold after 256 MB flush | 623.4; cold 614 to 615 | match |
| 2 | warm one triangle 1600 (1500 to 1780) | 1630 | match |
| 2 | cold 2550 (2400 to 3000) | 2627 | match |
| 2 | second trace in the kernel 782 | 782 | match |
| 3 | warm 1 / 12 / 144 / 200 / 2500 / 4096 / 20736 tri: 1600 / 1650 / 1900 / 1920 / 2250 / 2250 / 2580 | 1630 / 1608 / 1934 / 1982 / 2281 / 2270 / 2584 | all within 3.3% |
| 3 | cold 2550 / 2680 / 3300 / 3280 / 3890 / 4000 / 4640 | 2627 / 2566 / 3210 / 3253 / 3904 / 3921 / 4664 | all within 4.3% |
| 3 | second trace 782 / 827 / 868 at 1, 2, 3 levels | 782 (1 and 12 tri), 827 (144), 826 (200), 867 (2500, 4096); 908 at 20,736 (not predicted) | match |
| 4 | warm 1600, cold 2550, difference 950 to 1000 | 1630, 2627, difference 997 | match |
| 5 | rays per cycle at 1 / 2 / 4 / 8 / 16 / 20 warps: 0.0195 / 0.031 / 0.0445 / 0.058 / 0.080 / 0.090 | 0.0196 / 0.0312 / 0.0446 / 0.0577 / 0.0807 / 0.0904 | match (all within 1%) |
| 5 | saturated about 0.14 from 20 to 32 warps | warp completion gaps give 0.148 (20 warps) to 0.143 (32 warps); the whole span rate, which includes the ramp, is 0.090 to 0.100 | match (upper end, as the 4070 Ti) |
| 6 | siblings 1.02 to 1.05 times one SM (long coherent) | 1.02 to 1.05 (pairs 0,1; 2,3; 10,11) | match |
| 6 | two SMs in different TPCs 2.0 times | 1.98 to 2.0 | match |
| 6 | short query version (60 to 80% each, pair 1.25 to 1.4) | not run (not in the core set) | open |
| 7 | par64 G = 64: 63.6 times one SM | 63.5 (99.2%) | match |
| 7 | par64 G = 128: 66.2 times | 64.6 (-2.4%); the sibling adds 1.7%, predicted about 4% | match |
| 7 | absolute: about 0.10 rays per cycle per SM, 6.4 at G = 64, 6.6 at G = 128 | 0.0178 per SM, 1.13 and 1.15 aggregate | MISS, 5.6 times |
| 8 | Memory_1: L1 from about 90 KB to about 20 KB (chase) | holds to 80 KB, half lost at 96 to 112 KB; with the reservation 16 KB hits, 24 KB mostly lost | match |
| 8 | TTU second trace slower with the small L1 (mechanism check) | ratio 1.3 to 3.2 (1.45 at R = 1, 3.24 at R = 32, 1.29 at R = 256) | consistent |
| 9 | Memory_2: every node found again below about 60 KB, half lost at about 100 KB | NOT MEASURED: the ncu refetch run (mem2n) stopped at its first launch, ncu cannot open /tmp/nsight-compute-lock (owned by another user). The latency view (mem2) ran but has no direct comparison | open |
| 10 | sibling right after its partner costs 782 / 827 / 867 | 782 (tri1), 827 (grid 144), 867 (grid 2500) | match |
| 10 | an SM in another TPC 1570 to 2385 | 1676, 1968, 2335 | match |
| 10 | 50% of node fetches in own L1 and 50% in the sibling's | not measured (needs ncu) | open |

## Notes on the misses
- Item 7 absolute: the prediction took the tri1 rate (about 0.10 per SM) for the par64 scene. Triton measured par64 at 0.0178 per SM and tethys gives the same, so this is an error in the written prediction, not a difference between the boards. The scaling part (the ratios) is right.
- Items 1 ramp points: the measured ramp starts later and is wider than the model. Hit fraction from the latencies (between 295 L2 and 621 DRAM): about 0.82 at 72 MB, 0.52 at 74 MB, 0.15 at 76 MB, against 0.63, 0.28, 0 predicted. Still partly hitting at 76 MB, 1.05 C. Not refitted.
- Item 7, tri1 (not predicted, as the prediction said): 0.100 rays per cycle per SM from 1 to 65 SMs (aggregate 64.0 times at G = 64), 0.0877 at G = 128 (112 times); the script marks G >= 63 as not overlapping (0 of 5 launches), so the aggregate is not valid there.
