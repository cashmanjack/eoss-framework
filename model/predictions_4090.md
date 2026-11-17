# The performance model predictions for an RTX 4090 (tethys), written 2026-11-16 BEFORE any measurement on it

Pre-registration, from the chip's published specification and the model only. Nothing has been run on tethys: the machine could not be reached from triton (ssh asks for a password), no GPU detection, no stage 0, no memdump, no core set. The numbers below are replaced by nothing: results go in a separate file.

## Inputs (published specification, NOT detected; gpucfg on tethys must confirm each, a difference is itself a result)
| | triton (RTX 6000 Ada) | io (RTX 4070 Ti) | tethys (RTX 4090, assumed) |
|---|---|---|---|
| Die | AD102 | AD104 | AD102 (same die as triton) |
| SMs / TPCs | 142 / 71 | 60 / 30 | 128 / 64 |
| L2 | 96 MB | 48 MB | 72 MB |
| Memory | GDDR6, 384 bit | GDDR6X, 192 bit | GDDR6X, 384 bit |
| Boost clock | about 2.5 GHz | 3105 MHz max, 2.6 assumed | 2.52 GHz official, 2.6 to 2.8 under load assumed |
Driver and OptiX library on tethys are unknown; a driver without a validated entry in ttu/config/profiles.json is refused (validate with memdump first, as on io and triton 2026-11-16).

## How the model is used
Same as for the 4070 Ti (model/predictions_4070ti.md): everything inside one SM is SM89 and assumed identical, so those are the triton numbers (re-measured on triton 2026-11-16: warm one triangle 1595, cold 2543, second session 782). The 4090 is the same die as triton, so per SM numbers carry over with more confidence than to the AD104. Changed by the chip: 64 TPCs, 72 MB L2 with the asymmetric cyclic ramp of model/l2_capacity.py (hit 1 up to 0.97 C, 0 from 1.05 C), the DRAM latency in ns (about 250 ns, converted at the clock).

## Predictions (cycles unless said; "match" = within 10%, cold and DRAM within 15%)
1. Pointer chase (plain CUDA, one thread): L1 hit 41 (16 and 64 KB, .ca). L2 hit 285 to 295 (range 260 to 310) for 1 to 64 MB and for .cg. Cyclic chase across the L2 size, from the ramp at C = 72 MB: 68 MB 285, 70 MB 295, 72 MB 425 (range 340 to 520), 74 MB 554, 76 MB and above 658. Core set sizes: 64 MB L2 hit (about 290), 96 MB and 128 MB DRAM level 658. DRAM (512 MB, or after a 256 MB flush) 650 (range 560 to 780). L1 contents do not survive a kernel boundary (cross mode gives L2, about 285).
2. Latency_1 (fixed cost, one triangle, one warp): second trace in the same kernel (everything cached) 782 cycles at one level; fixed part about 740, plus 43 per level. Across launches, warm in L2: 1600 (range 1500 to 1780). Cold after a 256 MB read: 2550 (range 2400 to 3000); 256 MB is bigger than 72 MB, so it flushes the whole L2.
3. Latency_2 (depth, one warp, ray on triangle N/2): warm 1600 / 1650 / 1900 / 1920 / 2250 / 2250 / 2580 for 1 tri, 12, 144, 200, 2500, 4096, 20736 triangles; cold 2550 / 2680 / 3300 / 3280 / 3890 / 4000 / 4640 (each within 10% warm, 15% cold). In kernel second trace: 782 / 827 / 868 at 1, 2, 3 levels. Anchors measured on triton today: warm 1595 (1 tri), 2205 (2500), 2577 (20736); cold 2543, 3887, 4638.
4. Latency_6 (cold vs warm, one triangle): warm 1600, cold 2550, difference about 950 to 1000.
5. Throughput_1 (one SM, tri1 coherent, warm), rays per cycle at 1 / 2 / 4 / 8 / 16 / 20 warps: 0.0195 / 0.031 / 0.0445 / 0.058 / 0.080 / 0.090. Saturated steady state about 0.14 rays per cycle from 20 to 32 warps (the 4070 Ti measured 0.142 to 0.148 against 0.134 predicted: expect the upper end). Pure SM and unit behaviour, so unchanged.
6. Throughput_8: two siblings (2k, 2k+1) together 1.02 to 1.05 times one SM on long coherent queries; two SMs in different TPCs 2.0 times. Short query version: siblings each 60 to 80% of solo, pair total 1.25 to 1.4 times.
7. Throughput_6 par:64:s (long coherent queries, 32 warps per SM, one block per SM): G = 64 (one per TPC) 63.6 times one SM (99.4% efficient, as 70.6 of 71 on triton and 29.8 of 30 on the 4070 Ti); G = 128 (all SMs) 66.2 times one SM (sibling adds 4%). Absolute: about 0.10 rays per cycle per SM, so about 6.4 rays per cycle at G = 64 and 6.6 at G = 128. tri1 (short queries) is a different regime and is NOT predicted by these numbers (the 4070 Ti measured 30.0 and 41.3 at G = 30 and 60).
8. Memory_1 (shared memory carveout shrinks L1): L1 from about 90 KB to about 20 KB (chase); the TTU second trace slowdown with a small L1 is a mechanism check only (the per R numbers were inconclusive on the 4070 Ti). Unchanged.
9. Memory_2 (node cache size): every node found again below about 60 KB, half lost at about 100 KB. Unchanged.
10. Memory_5 (TPC sharing): one SM's queries put about 50% of node fetches in its own L1 and 50% in the sibling's L1; a trace on the sibling right after its partner costs the same as a second trace on the partner (782 / 827 / 867 at 1, 2, 3 levels); on an SM in another TPC 1,570 to 2,385 cycles. Unchanged.

## What a miss would mean
- Items 2 to 6 off by more than the range: per SM cycle costs are not invariant across boards of the same die (clock domain, a different L2 or memory configuration, power state), or a display / power limit on tethys.
- Item 1 ramp points off by more than 4 MB: the ramp scales with C only on the 4070 Ti (one chip, two points), or the L2 in use is not 72 MB (detected size decides).
- Item 7 efficiency far below 99%: TPC scaling is not just the TPC count.
- If tethys reports 128 SMs and 72 MB, but placement (probe_placement.cu) breaks the block to SM rule: items 6 to 7 are void until the rule is re-measured.

## How to run (when access exists)
GPU: an idle, non display GPU (nvidia-smi shows no Xorg or gnome-shell on it); set TTU_GPU. Set up ttu/config/gpucfg.py detection first (it must print 128 SMs, 72 MB), run probe_placement.cu, validate the driver by memdump (byte identical GAS group), then ttu/bench/io_core/core_set.py WORK chase|latency|thr1|thr8|thr6|thr6par|mem5|mem2 as on the 4070 Ti, adding chase points 68, 70, 72, 74, 76 MB.

## Erratum, 2026-11-16 (added after the run; the text above is unchanged)
Item 7 gave about 0.10 rays per cycle per SM for the long query scene (par:64:s). That was the wrong scene's rate. Measured on tethys 0.0178, and triton measures the same 0.0178. The 63.6 and 66.2 times scaling in that item is a ratio and was right. This is a writing error in the prediction, not a model miss and not refitted. Item 1 (L2 ramp at 72, 74, 76 MB) was a real miss: see F176 and F177.
