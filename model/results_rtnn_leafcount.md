# RTNN replay recheck with true slot counts (F167)

2026-11-16, triton GPU 1 (OptiX only, no TTU code). Rebuilt the patched RTNN (notes/q11/rtnn_patches.diff plus one line: RTNN_DRAM_GB overrides the device memory it reads, set to 11.5958 so that the memory based grid sizing is the 4070 Ti's; without it the 48 GB of this card makes a 1850^3 grid and the run faults in a thrust scan; build recipe in notes/q27: host compiler wrapper selecting the CUDA 12.8 headers and -D__CUDA_ARCH_LIST__=890, cmake -DKNN=50). Same points (seeded generator of model/predict_rtnn.py), same 12 configurations as F127. Check that triton's trees equal io's: u1e5 r4 sorted per query hardware calls identical to the io recorder file for 99.7 percent of the queries (mean 54.24097 both). Then model/rtnn_leafcount.py: replay with box matching (as F125, F127) and with the true primitives per leaf slot (type byte low 3 bits) on the dumped GAS and queries; recorder counts from the same run.

| config | hardware calls / query (recorder) | box matched replay x1.006 | true count replay x1.006 | box matched error | true count error | exact box hits | per query replays equal | lane eff (recorder) | lane eff box matched | lane eff true | sum of slot counts / primitives | slots with count 8 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| u1e5_r2_sorted | 7.97 | 7.96 | 7.96 | -0.2% | -0.2% | 7.21 | 1.0000 | 0.585 | 0.584 | 0.584 | 100000 / 100000 | 0 |
| u1e5_r2_unsorted | 7.97 | 7.96 | 7.96 | -0.2% | -0.2% | 7.21 | 1.0000 | 0.552 | 0.551 | 0.551 | 100000 / 100000 | 0 |
| u1e5_r4_sorted | 54.24 | 54.22 | 54.22 | -0.0% | -0.0% | 49.18 | 1.0000 | 0.777 | 0.777 | 0.777 | 100000 / 100000 | 0 |
| u1e5_r4_unsorted | 54.24 | 54.22 | 54.22 | -0.0% | -0.0% | 49.18 | 1.0000 | 0.737 | 0.737 | 0.737 | 100000 / 100000 | 0 |
| u1e5_r6_sorted | 99.20 | 174.31 | 174.31 | +75.7% | +75.7% | 158.64 | 1.0000 | 0.760 | 0.839 | 0.839 | 100000 / 100000 | 0 |
| u1e5_r6_unsorted | 99.20 | 174.31 | 174.31 | +75.7% | +75.7% | 158.64 | 1.0000 | 0.721 | 0.796 | 0.796 | 100000 / 100000 | 0 |
| u1e6_r1_sorted | 9.84 | 9.82 | 9.82 | -0.2% | -0.2% | 8.87 | 1.0000 | 0.614 | 0.613 | 0.613 | 1000000 / 1000000 | 0 |
| u1e6_r1_unsorted | 9.84 | 9.82 | 9.82 | -0.2% | -0.2% | 8.87 | 1.0000 | 0.579 | 0.578 | 0.578 | 1000000 / 1000000 | 0 |
| u1e6_r2_sorted | 69.65 | 69.66 | 69.66 | +0.0% | +0.0% | 63.09 | 1.0000 | 0.815 | 0.814 | 0.814 | 1000000 / 1000000 | 0 |
| u1e6_r2_unsorted | 69.65 | 69.66 | 69.66 | +0.0% | +0.0% | 63.09 | 1.0000 | 0.776 | 0.773 | 0.773 | 1000000 / 1000000 | 0 |
| u1e6_r3_sorted | 104.98 | 228.26 | 228.26 | +117.4% | +117.4% | 207.37 | 0.9999 | 0.742 | 0.880 | 0.880 | 1000000 / 1000000 | 0 |
| u1e6_r3_unsorted | 104.98 | 228.26 | 228.26 | +117.4% | +117.4% | 207.37 | 0.9999 | 0.696 | 0.839 | 0.839 | 1000000 / 1000000 | 0 |

Result: nothing moves. The two replays give the same primitives in entered slots for every query (equal for 100 percent of queries in ten configurations and 99.99 percent in the two radius 3 configurations of 1,000,000 points), so the F125 and F127 numbers, figures and fits stand unchanged (they used the box matched replay; their time and recorder data were from the 4070 Ti, whose trees are the same as triton's). Calls versus recorder: within 0.2 percent in the eight configurations that do not end early; the radius 6 (100,000 points) and radius 3 (1,000,000 points) configurations are 76 and 117 percent above the recorder in both replays: the K = 50 early termination, already stated in F127, not a slot assignment error. In a freshly built RTNN tree every slot holds one box or 2 (no slot with 8), the sum of slot counts is the number of primitives, and box matching is unambiguous because sibling boxes of a fresh build do not overlap enough to confuse it; the Mochi error (F166) appears only on a refit tree whose boxes were stretched by motion.
