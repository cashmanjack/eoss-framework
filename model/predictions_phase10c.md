# Predictions, 2026-11-18 (tethys, RTX 4090 GPU 1): F229 mechanism, ALU speedup clocks, micromaps with loads in the any hit, cheaper divergence probe

Committed before any run of these four items. Smoke runs (tiny launches) are allowed; any number read from them is disclosed in the results file. Only already validated TTU sequences (the Conc_1 lean group of F229). No transform sequence. Raw files are timestamped and never overwritten.

## 1. F229 mechanism: L1 bypassing loads and bandwidth levels (Conc_1 extension)

New compute type memcg: the same 1 GB streaming kernel as mem but with ld.global.cg (L1 bypass). The existing mem type uses a plain load (cached in L1 on SM89). Arms: RT G 32 (SMs 0, 2, ..., 62) beside mem (cached) and memcg, on sib (odd SMs below 64, 32 SMs) and hi (SMs 64 to 127), at 1, 2 and 4 compute blocks per SM (sib 32, 64, 128 blocks; hi 64, 128, 256 blocks), random rays (soup 100,000) and coherent rays (tri1). Compute bandwidth is reported from its own iteration counter (196,608 bytes per iteration of a block). Matched block counts: sib 2 against hi 1 (64 blocks), sib 4 against hi 2 (128 blocks).

- L1 eviction does not explain the sibling collapse: memcg on sib, random rays, 4 blocks gives RT ratio within a factor 2 of the cached value (0.01 to 0.05; F229 cached 0.022).
- Compute bandwidth of memcg within 15 percent of cached (pure streaming, nothing to reuse).
- RT ratio rises as the compute bandwidth falls, on both placements: sib random at 1 block per SM 0.08 to 0.35.
- Placement effect survives matching: random rays at 128 blocks, hi (2 per SM) at least 3 times the sib (4 per SM) ratio (hi 0.1 to 0.4).
- Coherent rays: memcg on sib and hi both close to the cached values (within 0.1 absolute). The coherent flip (sib above hi, 0.464 against 0.277) shrinks at matched blocks (sib over hi 1.0 to 1.5 against 1.7 at unmatched), so part of it is the block count, part of it placement.
- Zero wrong RT records in every run.

## 2. ALU speedup: clocks

Sampler: NVML SM and memory clock and power polled about every 1 ms by a separate process, with CLOCK_MONOTONIC stamps; the harness prints launch start and end stamps (CLOCK_MONOTONIC). Arms: alone G 32, hi_alu, sib_alu, random rays and coherent rays, 5 repeats, in the same order as F229.
- The 7 percent faster RT is again present (1.04 to 1.10).
- The SM clock during launches 1 to 5 differs by under 2 percent between alone and with ALU (not a clock boost). My expectation is that the GPU is already at the power limited boost clock in both, or the ALU kernel lowers it. If so the 7 percent stays unexplained by clock and I will say so (candidate: block to SM placement or the RT blocks sharing an SM with nothing else).
- Memory clock identical in all arms (same P state, under 1 percent).
- Power during launches higher with ALU by at least 30 W.

## 3. Micromaps with dependent global loads in the any hit (Omm_1 extension)

Any hit does AHL dependent loads from a random permutation of 128M u32 (512 MB, far above L2 72 MB) and ignores the intersection only if the last value is 0xFFFFFFFF (never), so the hit results are identical to the plain run. Variants as before: A (any hit on every candidate), D (no any hit), B(L) 4 state micromaps (any hit only for unknown micro triangles), C(L) 2 state. N 100,000 and 1,000,000 triangles, random rays, AHL 4 and 16, all variants in one process, round robin, 8 processes.
- Result hit, primitive and t identical to A in every B variant (0 mismatches).
- A over D time at AHL 4: 3 to 8 (the any hit is now on the critical path). At AHL 16 above 10.
- B(L) over A at AHL 4: B(2) 0.6 to 0.9, B(4) 0.2 to 0.45, B(6) 0.08 to 0.25, B(8) 0.05 to 0.2 (time follows invocations, 0.55, 0.14, 0.04, 0.01 of A, plus a fixed part from unit lookups).
- C(6) within 1.0 to 1.6 times D.
- So the saving shows once the any hit has memory latency: yes (B(4) under 0.5 of A at both AHL values).

## 4. Cheaper divergence probe on RTNN (Div_1)

RTNN rebuilt from the overlay with two new runtime modes (env): LHSAMPLE=7 (only warps with (warp and 7) == 0 stamp clocks and write histograms, raygen start and end stamps only for them) and LHNOTIME=1 (all warps, only the lane count histogram, no clock reads, no raygen stamps). Synthetic data, r 0.025 and 0.05, K 50, 100,000 points as F231; kernel time from the printed KERNEL line, probe off in the same build as baseline, 3 repeats each.
- Overhead against probe off: full probe 1.23 to 1.28 (F231), sampled 1/8 under 1.06, no timing 1.03 to 1.12.
- Sampled lane statistics (mean lanes per callback) within 3 percent of the full probe; time share at 8 or fewer lanes within 0.03 absolute.
- No timing: lane histogram identical to the full probe's (spread 0.0 percent as in F231, the lane counts do not depend on timing).

## 1b. Addendum (written after the first full repetition of section 1, before any further run; the numbers below that repetition are disclosed here)
Seen in that repetition (raw notes/conc2/smoke, I started the full cg group by mistake instead of smoke, it is repetition 0): the streaming kernel reaches only about 120 GB/s (alone) whatever the block count (1, 2 or 4 per SM), so block count does NOT set bandwidth levels; RT random 0.022 on sib for both mem and memcg and every block count, 0.14 to 0.15 on hi; coherent sib 0.45 to 0.48, hi 0.26 to 0.30. So in section 1: L1 eviction prediction held; "sib random rises at 1 block" missed (flat); the matched block flip prediction for coherent (1.0 to 1.5) missed (sib over hi 1.65 to 1.76 at 64 and 128 blocks).
The stream is 16 bytes per thread at a 4 MB stride (every lane its own line and page), not a coalesced stream. New arms (group lvl):
- Throttle: a sleep per iteration (150 us and 800 us, about 1/4 and 1/16 of the alone rate), on mem and memcg, sib (4 blocks per SM) and hi (2 per SM, the matched 128 blocks).
- memseq: coalesced contiguous stream (each warp reads 512 contiguous bytes per load), plain load, sib at 1 and 4 blocks per SM and hi at 1 and 2.
Predictions:
- Throttled, random rays: RT ratio rises monotonically with the compute rate falling on sib: at 1/4 rate 0.08 to 0.4, at 1/16 rate 0.4 to 0.9. On hi: 1/4 0.5 to 0.9, 1/16 0.8 to 1.0. The sib ratio stays below the hi ratio at the same level (at least 1.3 times lower).
- memseq on sib, random rays, 4 blocks: compute bandwidth at least 2 times the strided rate (240 GB/s or more) and RT ratio at least 5 times the strided 0.022 (0.11 or more), because the collapse is about the access pattern (one line and page per lane), not bytes. If memseq also gives about 0.02 the pattern is not the cause and bytes or request rate is.
- Coherent rays, throttled: RT ratio rises with the rate falling on both placements (sib 1/16 above 0.8, hi 1/16 above 0.6); at each level sib above hi (placement effect), compute side lower on sib.
