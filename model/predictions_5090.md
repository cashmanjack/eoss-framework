# The performance model predictions for an RTX 5090 (tgrogers-pc14), written 2026-11-19 BEFORE any core set measurement on it

Pre-registration, from the model's Ada constants and the detected config only. None of the core set tests has been run on this GPU (and no TTU code of ours has run on it at all). What was run on this GPU before writing this: gpucfg detection, a plain CUDA block placement probe, an OptiX one triangle launch with a read of OptiX's own code (memdump, decode only). Nothing in this file comes from a timing on the 5090. Results go in a separate file.

## Inputs
| | triton (RTX 6000 Ada) | 4090 (tethys) | 5090 (pc14) |
|---|---|---|---|
| Architecture | Ada AD102, SM89 | Ada AD102, SM89 | Blackwell GB202, SM120 (a different RT core generation) |
| SMs / TPCs | 142 / 71 | 128 / 64 | 170 detected / 85 ASSUMED (2 SMs per TPC, only evidence: blocks land on SM pairs 2k, 2k+1) |
| L2 | 96 MB | 72 MB | 96 MB detected (100,663,296 bytes) |
| Memory | GDDR6, 384 bit | GDDR6X, 384 bit | GDDR7, 512 bit detected |
| SM clock | about 2.5 GHz | 2.52 official, 2.6 to 2.8 under load | 3,090 MHz max (nvidia-smi), 2,437 MHz CUDA clock attribute; under load unknown, 2.4 to 2.9 assumed |
Driver 580.126.09, not in the validated list for TTU runs (validated for memdump only). The GPU drives the display and is shared (Steam helper present), so clocks and power state are not controlled.

## The assumption behind every number
Everything inside one SM and its TTU is taken as the Ada number (triton constants, confirmed on the 4090 and the 4070 Ti), measured in SM cycles. This is a null hypothesis, not a belief: the 5090 has a newer RT core and a different memory system. The OptiX traversal code on SM120 is the same group as on SM89 in structure (see notes/protocol.md SM120), which is why the null hypothesis is worth testing, but code equality says nothing about unit timing. Items with an L2 or DRAM in them use the 5090's detected L2 size and the 4090's measured L2 ramp scaled by the L2 size.

## Predictions (cycles unless said; "match" = within 10%, cold and DRAM within 15%; ranges are wider than for Ada because the architecture is new)
1. Pointer chase (plain CUDA, one thread): L1 hit 41 (range 30 to 50; 16 and 64 KB, .ca). L2 hit 290 (range 250 to 340) for 1 to 64 MB and for .cg. Cyclic chase across the L2 size C = 96 MB, using the 4090's measured ramp scaled by C (4090: 70, 72, 74, 76 MB = 0.97, 1.00, 1.03, 1.06 C at 290, 354, 451, 572): 93 MB 290, 96 MB 354 (range 290 to 520), 99 MB 451, 101 MB 572, 106 MB and above 650. Core set sizes: 64 MB L2 hit (about 290), 96 MB about 354 (this point is on the ramp and is the least certain), 128 MB DRAM level 650. DRAM (512 MB, or after a 256 MB flush) 650 (range 560 to 800, the range widened for GDDR7 and the clock). L1 contents do not survive a kernel boundary (cross mode gives L2, about 290). Scaling the ramp by C is untested (the triton ramp at 96 MB was never measured, F177), so a miss here is a statement about the ramp form, not about the 5090.
2. Latency_1 (fixed cost, one triangle, one warp): second trace in the same kernel 782 cycles at one level (fixed part about 740, plus 43 per level; range 700 to 900). Across launches, warm in L2: 1,630 (range 1,500 to 1,900). Cold after a 256 MB read: 2,600 (range 2,400 to 3,100); 256 MB is bigger than the 96 MB L2, so it flushes it.
3. Latency_2 (depth, one warp, ray on triangle N/2): warm 1,600 / 1,650 / 1,900 / 1,920 / 2,250 / 2,250 / 2,580 for 1 tri, 12, 144, 200, 2,500, 4,096, 20,736 triangles; cold 2,550 / 2,680 / 3,300 / 3,280 / 3,890 / 4,000 / 4,640 (each within 10% warm, 15% cold). In kernel second trace: 782 / 827 / 868 at 1, 2, 3 levels.
4. Latency_6 (cold vs warm, one triangle): warm 1,630, cold 2,600, difference about 950 to 1,000.
5. Throughput_1 (one SM, tri1 coherent, warm), rays per cycle at 1 / 2 / 4 / 8 / 16 / 20 warps: 0.0195 / 0.031 / 0.0445 / 0.058 / 0.080 / 0.090; saturated steady state 0.14 to 0.148 rays per cycle from 20 to 32 warps. Long query scene (par:64:s type) 0.0178 rays per cycle per SM (the corrected number of the 4090 erratum, not 0.10).
6. Throughput_8: two siblings (2k, 2k+1) together 1.02 to 1.05 times one SM on long coherent queries; two SMs in different TPCs 2.0 times. Short query version: siblings each 65 to 80% of solo, pair total 1.3 to 1.5 times (triton 1.53, 4070 Ti 1.38; the 4090 value is a gap). Condition: siblings are SM 2k and 2k+1, which is the assumed TPC pairing; if the siblings are not the pair, long queries show 2.0 and this item is wrong in kind.
7. Throughput_6 par:G:s (long coherent queries, 32 warps per SM, one block per SM): G = 85 (one per TPC) 84.5 times one SM (99.4% efficient, as 70.6 of 71 on triton, 63.5 of 64 on the 4090); G = 170 (all SMs) 87.8 times one SM (sibling adds 4%). Absolute: 0.0178 rays per cycle per SM, so about 1.5 rays per cycle at G = 85 and 1.56 at G = 170. tri1 (short queries) is a different regime and is not predicted by these numbers. Condition: the Ada rule "block b on SM 2b for b < SMs/2, then the odd SMs" does NOT hold on this GPU (probe_placement: 825 mismatches in 5 launches; blocks go to the pairs 0, 22, 44, 66, 88, 110, 132, then 2, 24, ...). The harness must use the measured block to SM table, otherwise G = 85 does not mean one block per TPC and the 84.5 does not apply.
8. Memory_1 (shared memory carveout shrinks L1): L1 from about 90 KB to about 20 KB (chase), maximum shared memory per SM 100 KB (detected 102,400 bytes, equal to Ada); the TTU second trace slowdown with a small L1 is a mechanism check only. Unchanged.
9. Memory_2 (node cache size): every node found again below about 60 KB, half lost at about 100 KB. Unchanged.
10. Memory_5 (TPC sharing): one SM's queries put about 50% of node fetches in its own L1 and 50% in the sibling's L1; a trace on the sibling right after its partner costs the same as a second trace on the partner (782 / 827 / 867 at 1, 2, 3 levels); on an SM in another TPC 1,570 to 2,385 cycles. Unchanged; same pairing condition as item 6.

## What a miss would mean
- Items 2 to 6 off by more than the range: the 5090 RT core (or its clock and memory system) does not have the Ada per SM cycle costs. That is the expected result a new generation could give; it is a finding, not a failure of the model.
- Items 6, 7 and 10 off in kind (siblings behave like other TPCs, or the other way round): the TPC pairing assumption (2 SMs per TPC, SM 2k and 2k+1) is wrong on this GPU.
- Item 1 ramp points off by more than 5 MB: the ramp does not scale with C (or the usable L2 is not 96 MB, for example because of partitioning).
- Item 7 efficiency far below 99%: TPC scaling is not just the TPC count, or the shared display GPU changes clocks.
- A wide spread between repeats: power state and other users of this GPU (a game or the desktop); the GPU is not controlled here.

## How to run (when approved; none of it was run)
Rules of this machine: GPU 0 only, nvidia-smi must show only Xorg, gnome-shell and the Steam helper, every run under timeout -k 2 30, smallest launch first, short launches only (display watchdog). The core set needs TTU code of ours: the TTU sequence on this GPU must first be checked against the live SM120 OptiX group (notes/protocol.md SM120: close bit 74, N counts, DUAL open) and the driver validated for TTU in profiles.json by Jack. The plain CUDA items (1, part of 8) can run first.
