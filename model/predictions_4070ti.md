# Model v1 predictions for the RTX 4070 Ti (io), written 2026-11-13 BEFORE any core set measurement

Pre-registration. Written after: detection of the hardware, stage 0 on io, check_seq, memdump confirming OptiX's traversal group. No pointer chase, latency, throughput or memory number has been measured on io yet.

## Inputs (detected on io, not assumed)
| | triton (RTX 6000 Ada) | io (RTX 4070 Ti) |
|---|---|---|
| SMs / TPCs | 142 / 71 (one TPC disabled) | 60 / 30 (full AD104) |
| L2 | 96 MB | 48 MB (50,331,648 B) |
| Max SM clock (nvidia-smi) | about 2.5 GHz in our logs | 3105 MHz max; sustained boost under load unknown, I assume 2.6 GHz |
| Memory | GDDR6, 384 bit | GDDR6X, 192 bit (max mem clock 10501 MHz) |
| Placement rule (one block per SM) | block b on SM 2b, then odd SMs | same rule holds (probe_placement, 5 launches, 0 mismatches) |

## How v1 is used
v1 is fitted in cycles per ray on ONE SM. Everything inside one SM (unit, L1, TTU group, per level node cost with the BVH in L1) is assumed identical on the same architecture (SM89), so those predictions are the triton numbers unchanged. Parts that depend on the chip:
- TPC scaling: one traversal unit per TPC, sibling SM adds 4% (long coherent queries). Apply with 30 TPCs.
- L2 capacity: the BVH sizes at which things leave L2 move from 96 MB to 48 MB.
- DRAM latency: assumed constant in ns (about 250 ns on triton: 616 to 654 cycles at 2.5 GHz), converted at 2.6 GHz.
- L2 and per GPC sector ceilings (random ray plateau): v1 has no formula for these from SM count; NOT predicted here (would be a guess). Not in the core set.

## Predictions (cycles unless said; "match" = within 10%, cold and DRAM within 15%)
1. Pointer chase (plain CUDA, one thread): L1 hit 41 (16 and 64 KB, .ca). L2 hit 285 (range 260 to 310) for 1 to 16 MB and for .cg. 64 MB is bigger than this L2: NOT 298; expect between L2 and DRAM, about 450 (range 350 to 600), the one place where io must differ from triton. DRAM (512 MB, or after a 256 MB flush) 650 (range 560 to 780). L1 contents do not survive a kernel boundary (cross mode gives L2, about 285).
2. Latency_1 (fixed cost, one triangle, one warp): second trace in the same kernel (everything cached) 782 cycles at one level; fixed part about 740, plus 43 per level. Across launches, warm in L2: 1641 (range 1500 to 1780). Cold after a 256 MB read: 2650 (range 2400 to 3000).
3. Latency_2 (depth, one warp, ray on triangle N/2): warm 1641 / 1652 / 1908 / 1920 / 2315 / 2315 / 2639 for 1 tri, 12, 144, 200, 2500, 4096, 20736 triangles; cold 2648 / 2680 / 3228 / 3284 / 3887 / 4058 / 4635 (each within 10% warm, 15% cold). In kernel second trace: 782 / 827 / 868 at 1, 2, 3 levels.
4. Latency_6 (cold vs warm, one triangle): warm 1640, cold 2650; the 256 MB flush exceeds the 48 MB L2 so it flushes completely here too. Cold minus warm about 1000 cycles.
5. Throughput_1 (one SM, tri1 coherent, warm), rays per cycle at 1 / 2 / 4 / 8 / 16 / 20 warps (original 20 warp layout): 0.0195 / 0.031 / 0.0445 / 0.058 / 0.080 / 0.090. Saturated steady state 0.134 rays per cycle from 20 to 32 warps. Pure SM and unit behaviour, so unchanged.
6. Throughput_8 (full overlap, long coherent queries): two siblings (2k, 2k+1) together 1.02 to 1.05 times one SM; two SMs in different TPCs 2.0 times. Short query version: siblings each 60 to 80% of solo, pair total 1.25 to 1.4 times.
7. Throughput_6 coherent (tri1, 32 warps per SM, one block per SM): G = 30 (one per TPC) 29.8 times one SM (99.4% efficient, as 70.6 of 71 on triton); G = 60 (all SMs) 31.0 times one SM (sibling adds 4%). Absolute: about 0.10 rays per cycle per SM, so about 3.0 rays per cycle at G = 30 and 3.1 at G = 60, about 7.8 and 8.1 rays per ns at 2.6 GHz.
8. Memory_1 (shared memory carveout shrinks L1): L1 from about 90 KB to about 20 KB (chase); a repeated trace with more than one path of nodes is 1.9 to 2.5 times slower in the small L1. Same SM L1, so unchanged.
9. Memory_2 (node cache size): every node found again below about 60 KB, half lost at about 100 KB. Unchanged.
10. Memory_5 (TPC sharing): one SM's queries put about 50% of node fetches in its own L1 and 50% in the sibling's L1; a trace on the sibling right after its partner costs the same as a second trace on the partner (782 / 827 / 867 at 1, 2, 3 levels); on an SM in another TPC 1,682 to 2,385 cycles. Unchanged.

## What a miss would mean
- Items 2 to 6 off by more than the range: the cycle costs are not SM invariant (clock domain, L2 path), v1's per SM assumption fails across chips.
- Item 1 (64 MB) equal to 298: the L2 is not 48 MB for these accesses (detected size wrong or partitioned).
- Item 7 efficiency far below 99%: TPC scaling is not just the TPC count (different shared resource).
