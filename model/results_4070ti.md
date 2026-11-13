# Core set on the RTX 4070 Ti (io, GPU 0): measured vs predictions written beforehand (model/predictions_4070ti.md, commit 886ea97)
Run 2026-11-13. "match" = within the stated range. All TTU runs exit 0, 0 wrong vs OptiX where results are stored, no fault.

| # | Test | Predicted | Measured | Verdict |
|---|---|---|---|---|
| 1 | Chase L1 (16, 64 KB) | 41 | 41.0, 40.2 | match |
| 1 | Chase L2 (1..40 MB, cg, cross) | 285 (260..310) | 283.5..286.3 | match |
| 1 | Chase 48 MB / 56 MB / 64 MB | 64 MB: about 450 (350..600) | 424 / 658 / 658 | MISS at 64 MB: a cyclic chase bigger than the L2 gets no hits (LRU thrash): DRAM level. L2 holds 40 MB, partly 48 MB, nothing at 56 MB |
| 1 | Chase DRAM (96..512 MB, cold) | 650 (560..780) | 652..661.5 | match |
| 2 | Latency_1 second trace (1 level) | 782 | 782 | match |
| 2 | Latency_1 warm, one triangle | 1641 (1500..1780) | 1612 | match (-1.8%) |
| 2 | Latency_1 cold (256 MB read) | 2650 (2400..3000) | 2593 | match (-2.2%) |
| 3 | Latency_2 warm 1/12/144/200/2500/4096/20736 tri | 1641/1652/1908/1920/2315/2315/2639 | 1612/1635/1930/1984/2234/2248/2564 | match (-3.5..+3.3%) |
| 3 | Latency_2 cold | 2648/2680/3228/3284/3887/4058/4635 | 2593/2663/3351/3245/4004/3981/4645 | match (-2.1..+3.8%) |
| 3 | Latency_2 second trace | 782/827/868 at 1/2/3 levels | 782/782/827/826/867/867/908 | match |
| 4 | Latency_6 warm / cold / difference | 1640 / 2650 / about 1000 | 1612 / 2593 / 981 | match |
| 5 | Throughput_1 rays/cycle at 1/2/4/8/16/20 warps | .0195/.031/.0445/.058/.080/.090 | .0200/.0316/.0450/.0586/.0811/.0910 | match (+1..+2.6%) |
| 5 | Throughput_1 saturated steady state | 0.134 | gap 216..225 cycles per warp = 0.142..0.148 | +6..+10%, edge of the 10% band |
| 6 | Throughput_8 siblings, long queries (par:256:s) | 1.02..1.05 | 1.03, 1.05 (pair 2,3: 1.14 at 80..100% overlap) | match (partial overlap inflates 2,3, as on triton) |
| 6 | Throughput_8 different TPCs | 2.0 | 1.99, 1.99, 2.01 | match |
| 7 | Throughput_6 par:64:s G=30 / G=60 | 29.8 / 31.0 times one SM | 29.8 / 31.1 | match (prediction text said tri1, the numbers were par:64:s) |
| 7 | Throughput_6 tri1 (short queries) G=30 / G=60 | not predicted | 30.0 / 41.3 | siblings add 38% for short queries (a single SM does not saturate the unit); not comparable with par:64:s |
| 8 | Memory_1 L1 size by chase, no / with shared memory reserved | about 90 KB / about 20 KB | 40 cycles up to 80 KB, 120 at 96 KB / 41 at 16 KB, 100 at 24 KB, 210 at 32 KB, 285 at 48 KB | match |
| 8 | Memory_1 TTU second trace slowdown, small vs big L1 | 1.9..2.5 times beyond one path | R1 1.56, 4 2.16, 8 2.64, 16 2.72, 32 3.22, 64 2.20, 128 1.48, 256 1.42 | mechanism yes; per R numbers outside the range at R1 and R32 and depend on the ray pool (absolute latencies differ from triton's, so a different pool): inconclusive on magnitude |
| 9 | Memory_2 refetch (ncu), L1 about 90 KB | found again below about 60 KB, half lost near 100 KB | 0.00 up to 15 KB, 0.02 at 28, 0.03 at 42, 0.04 at 61, 0.12 at 95, 0.51 at 115, 0.54 at 173, 0.75 at 228, 1.00 at 722 KB | match |
| 9 | Memory_2 small L1 (about 20 KB) | not predicted | refetch 1.0..1.9 (over 1 means fills exceed the first session) up to 42 KB, 3 to 5 at 61..230 KB | as on triton in kind |
| 10 | Memory_5 B after A on the sibling | 782/827/867 | 782/827/867 | exact |
| 10 | Memory_5 B after A in another TPC | 1682/1956/2385 | 1569/1863/2169 | -7%/-5%/-9%, match; A alone 1602/1912/2258 |

Not run (not in the core set, v1 has no formula from SM count): random ray plateau (L2 and per GPC sector ceilings), Throughput_6 random rays.
