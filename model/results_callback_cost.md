# Cost constants of the SM callback term (microbenchmark model/cbcost/cbcost.cu, plain CUDA, RTX 4070 Ti on io GPU 0, 2026-11-15)
One block of W warps on each of the 60 SMs, 4,096 loop iterations, window = slowest warp of the SM (clock64), mean over SMs, third launch of three.

| kind | pattern | SM cycles per warp instruction | per unit |
|---|---|---|---|
| independent FFMA (4 chains, loop overhead included) | 32 warps | 0.318 | 0.25 is the 4 scheduler issue limit; 0.318 includes about 1 loop instruction per 4 FFMA |
| load, 1 line per instruction (lane stride 4 B) | L1 resident 16 KB | 6.98 | address generation dominates (about 7 instructions per load); not used |
| load, 32 lines per instruction | L1 resident 16 KB | 32.08 | 1.00 per L1 tag request (32 tags) |
| load, 32 random lines per instruction | 8 MB (L2 resident) | 92.9 | 2.90 per sector (= 1.00 tag + 1.90 extra); 8, 16, 32 warps identical (throughput bound) |
| load, 32 random lines per instruction | 32 MB (L2) | 95.4 | 2.98 per sector |
| load, 32 random lines per instruction | 256 MB / 1 GB (DRAM) | 1016 / 1326 | 32 / 41 per sector (TLB and DRAM page misses; no callback here touches that much) |

Used: c_i 0.25, c_tag 1.00, c_x 1.90. First version of the benchmark used a 64 bit modulo for the address and measured 38 to 95 cycles per instruction (address arithmetic, not memory); replaced by a power of two mask before any number was used.
