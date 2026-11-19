# RTX 5090 (pc14) predictions for the fourth batch, written 2026-11-19 BEFORE any of these measurements

Not measured yet when this was written: node state on Blackwell (F150, F159, F161, F174 replicas), Throughput_6 with one triangle queries, the TTU half of Memory_1. Nothing here comes from a timing of these arms on the 5090. All inputs are Ada results (triton, 4090, 4070 Ti) and the earlier 5090 core set results (model/results_5090_core.md, results_5090_plain.md).

Disclosure of what I had seen: (a) the Memory_2 latency script output on the 5090 (second trace small L1 over big L1 was 1.08 at 1 ray to 1.86 at 512 rays, 16 warps), which is a different script from the Memory_1 script below; (b) the SM120 fence compiled by nvcc has one more instruction (CGAERRBAR) than the SM89 one; (c) the completion gap steady rate of the 5090 core set Throughput_1 data (see the correction in model/results_5090_chip_inputs.md), which I looked at while preparing the saturated rate analysis. That analysis is of existing data, so it has no prediction.

Rules: saved fraction s = 1 minus (second session cycles over single session cycles), one TTU warp on SM 0, two sessions of the same ray, scenes par:N:s, medians of 3 repetitions of 30 launches. A prediction holds if the measured value is inside the stated range. Stage 0 of every build first, then one stage 1 launch per arm, then the repetitions; stop at the first fault, hang, timeout, wrong exit, a launch where not exactly one TTU block ran, or a sibling timeout.

## 1. Node state on Blackwell (replicas of F150, F159, F161, F162, F174; null hypothesis: the Ada structure carries over)
Reasoning for the ranges: the second session on a warm node state is unit and L1 bound, which was the same on the 5090 (790 against 782 cycles); the first session is partly L2 bound and the 5090 L2 costs 349 against 290 cycles, so s may be a little larger than on Ada where the BVH fits in the node state, and about equal where it does not (both sessions are L2 bound).
- 1a Capacity curve (arms single, twice, cctl; N = 16, 64, 256, 1024, 2048, 3072, 4096, 6144, 8192, 16384, 65536): s in 0.50 to 0.75 for N up to 2,048 (68 and 132 KB; Ada 0.54 to 0.63), s in 0.12 to 0.33 at N 3,072 and 4,096 (199 and 263 KB; Ada 0.23), s at most 0.10 at N 8,192 and above (Ada 0.04 and below), s at most 0.04 at N 65,536; the cctl arm within 0.06 of 0 at every N. Hold = all of these. Probability 55 percent (the knee can move by one step: I accept the knee between 132 and 330 KB; the new generation may have a different node cache).
- 1b Fence decomposition (N 1,024 and 2,048): sibling polling without a fence 0.45 to 0.75; cta scope fence on the sibling 0.45 to 0.75 (no loss); gpu scope fence on the sibling 0.15 to 0.40; gpu scope fence on the TTU's own SM 0.15 to 0.40; gpu fences on both SMs at most 0.12. The SM120 gpu fence is MEMBAR.ALL.GPU, ERRBAR, CGAERRBAR, CCTL.IVALL (copied from nvcc's own output); I predict CGAERRBAR changes nothing. Probability that all five hold: 45 percent.
- 1c F161 TTUCCTL alone on the sibling SM (build with a second TTU block, 170 blocks, SMs 0 and 1): s at most 0.06 at both sizes. Probability 65 percent. F162 TTUCCTL on the own SM: s at most 0.06. Probability 70 percent.
- 1d F174 own L1 stream (as the Ada arm: a plain block on the TTU's own SM streams 1 MB of ordinary loads between the sessions) against the same wait with no loads: s(stream) at least 0.10 below s(wait) at both sizes. Probability 55 percent.
- Overall: at least 3 of these 4 groups (1a, 1b, 1c with 1c counted as one group, 1d) hold: 55 percent.
Blackwell specific risks I cannot exclude: the plain sibling block (384 threads) may not land on SM 1 because the placement on this chip is irregular (the host then stops with "no block landed", no TTU kernel started); the arm needing 170 resident TTU blocks (h) depends on all blocks being resident at once.

## 2. Throughput_6 with one triangle queries (tri1, hit, 32 warps per SM, one block per TPC through --sm-set, then all SMs; same method as the par64 run)
Reference: Ada 4070 Ti, F105: 30.0 times one SM at 30 SMs (one per TPC), 41.3 times at 60 SMs (both SMs of every TPC, so the sibling adds 38 percent on short queries; Throughput_8 short version 1.3 to 1.5, triton 1.53). One SM span rate on the 5090 at 32 warps is 0.104 rays per cycle (Throughput_1, same build family).
- G = 1: 0.100 to 0.108 rays per cycle (the Throughput_1 value).
- G = 85 (one block per TPC): 82 to 85.5 times one SM (at least 97 percent efficiency; Ada 30.0 of 30).
- G = 170 (every SM): 1.25 to 1.50 times the G = 85 aggregate, so 104 to 128 times one SM. Probability 60 percent. Per SM rate at G = 170: 0.065 to 0.078.
- Misses I consider plausible: a shared limit (L2 or the harness record traffic) below the unit limit would give efficiency under 97 percent at G = 85, and I put 25 percent on that.

## 3. Memory_1, TTU half (second trace latency, R distinct random rays on soup:100000, warps = max(1, R/32), L1 about 90 KB against --smem 98304 reserved, L1 about 20 KB; ratio = small L1 over big L1)
Ada: no change when the warp's rays touch one path (ratio about 1), 1.9 to 2.5 times longer from several paths on (triton), 1.4 to 3.2 (4070 Ti, depends on the ray pool).
- R = 1: 0.95 to 1.15. R = 4 and 8: 1.0 to 1.8. R = 32 and above: 1.4 to 3.2. Hold = R = 1 in range and R 32 and above all in range.
- Ordering: the ratio does not decrease with R by more than 10 percent from R = 8 on. Probability 55 percent for the whole set (the 5090 L2 is slower, which lowers the ratio when the fetches are L2 latency bound in both arms).

## What a miss would mean
A miss in 1a or 1b says the node state structure (capacity, which SM state a fence clears) is different on SM120. A miss only in the fence arms with the CGAERRBAR instruction in the build would point at the fence sequence. A miss in 2 at G = 85 says there is a shared resource limit on the 5090 below the unit limit for short queries. A miss in 3 says the L1 dependence of node caching differs on SM120.
