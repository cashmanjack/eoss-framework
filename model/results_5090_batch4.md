# RTX 5090 (pc14): node state on Blackwell, Throughput_6 short queries, Memory_1 TTU half, scored against model/predictions_5090_batch4.md (commit 6c44ef2, made before any of these runs), 2026-11-19

Rules kept: only checked SM120 builds (check_seq ALL PASS on all 14 node builds), stage 0 of every build first (12 arms, no TTU instruction), one stage 1 launch per arm at the smallest scene, then 3 repetitions of 30 launches, medians; nvidia-smi check (pc14_gpucheck.sh, run.py) immediately before every process; timeout -k 2 30; GPU 0; no fault, hang, timeout, wrong exit, missing block or sibling timeout in any stage 1 run. Stops that were NOT faults: another user's job (vipe python, 1.8 to 3.9 GB) on the GPU made the check refuse 4 times (one capacity pass, one arms pass, two Memory_1 starts); each time the partial files were left and the run restarted in a NEW directory after the check passed. Raw (private): notes/pc14_node/n3 and n4, notes/pc14_core/thr6tri1_*, mem1ttu_*.

## Harness port (what changed to run SM120 sibling builds)
- build_patch.py: --sibling (with --ownfence, --ownccctl, --sibccctl) allowed with --sm 120. The SM120 fence nvcc emits for fence.acq_rel.gpu has four instructions (MEMBAR.ALL.GPU, ERRBAR, CGAERRBAR, CCTL.IVALL); the builder copies all four from the host's own compiled sib_fence kernel. TTUCCTL.IVALL: the SM120 encoder gives the same 16 bytes as SM89 (asserted).
- BUG FOUND at stage 0 (no TTU code): SM120 encodes BRA offsets differently (offset in 4 byte words v: bits 23:16 = v & 0xff, bits 63:34 = v >> 8; SM89 has the byte offset in bits 63:32). The first sibling build, which spins on a flag, branched to a wrong address: "illegal memory access, 0 blocks ran" at stage 0 (first fault of the session; stopped, then diagnosed from the disassembly, no further run of that binary). The encoding was read from nvcc output and from the BRAs of the compiled host binary and verified byte for byte against five of them (offsets of 64, 48, 672, 6,192 and 3,376 bytes). check_seq.py had decoded branch targets the SM89 way, so it had passed my wrong build (it matched my own wrong encoding); it now decodes SM120 branches correctly. All 14 builds were rebuilt and re-passed, then stage 0 (12 of 12 arms ran), then stage 1.
- check_seq.py B15: on SM120 the ownfence rule requires the four instruction fence in a row. This is a rule extension for SM120 only; SM89 rules unchanged. For Jack to confirm.
- New runners: ttu/bench/pc14/node_run_pc14.py, mem1_pc14.py.

## 1. Node state on Blackwell (saved fraction s = 1 minus second over single session; one TTU warp on SM 0)
1a Capacity curve (arms single / twice / cctl; cycles single, twice):
| N (KB of BVH) | 16 | 64 | 256 | 1,024 (68) | 2,048 (132) | 3,072 (199) | 4,096 (263) | 6,144 | 8,192 | 16,384 | 65,536 (4 MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| s twice | 0.60 | 0.70 | 0.65 | 0.65 | 0.53 | 0.23 | 0.09 | 0.08 | 0.04 | 0.03 | 0.00 |
| s cctl | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| single (cycles) | 4,229 | 6,386 | 27,024 | 88,796 | 180,230 | 337,940 | 391,583 | 768,871 | 876,460 | 1,946,115 | 8,983,010 |
Predicted: s 0.50 to 0.75 up to 2,048 (held: 0.53 to 0.70), 0.12 to 0.33 at 3,072 and 4,096 (3,072 held at 0.23; 4,096 MISSED: 0.09, 0.03 below the range; Ada 0.23), at most 0.10 from 8,192 (held), at most 0.04 at 65,536 (held), cctl within 0.06 of 0 (held at every N). The whole group did not hold: the knee is sharper than on Ada, the saving is gone by 263 KB where Ada keeps 0.23. Same flat part and the same 130 KB scale. Single session cycles at 68 KB: 88,796 (Ada 100,740): the first session is not slower on the 5090 here; the L2 latency being higher did not make the first session of the long query scene longer.
1b Fence decomposition (3 x 30 launches, N 1,024 and 2,048; single 88,801 and 180,245):
| Arm | 68 KB s | 132 KB s | Predicted | Held |
|---|---|---|---|---|
| sibling polling, no fence (b1) | 0.65 | 0.51 | 0.45 to 0.75 | yes |
| cta fence on the sibling (e_cta) | 0.65 | 0.51 | 0.45 to 0.75 | yes |
| gpu fence on the sibling (c1_gpu) | 0.23 | 0.16 | 0.15 to 0.40 | yes (0.16 close to the edge) |
| gpu fence on the TTU's own SM (f_own) | 0.24 | 0.24 | 0.15 to 0.40 | yes |
| gpu fences on both (g_both) | 0.00 | 0.00 | at most 0.12 | yes |
All five hold, with the SM120 four instruction fence (CGAERRBAR changes nothing visible). Ada: 0.26 / 0.23 for the sibling gpu fence, 0.00 for both.
1c TTUCCTL alone on the sibling SM (h, second TTU block, 170 blocks, SMs 0 and 1): 0.00 / 0.00 (predicted at most 0.06: held). TTUCCTL on the own SM (i): 0.00 / -0.00 (held).
1d Own L1 stream (1 MB of ordinary loads on the TTU's own SM between the sessions, 0.24 and 0.26) against the same 70 us wait with no loads (0.49 and 0.37): 0.25 and 0.11 lower (predicted at least 0.10 lower: held, the 132 KB point only just).
Tally: 3 of 4 groups held (1b, 1c, 1d); 1a missed at one point. Prediction "at least 3 of 4": held. Node state on Blackwell is the Ada structure (flat saving to about 130 KB, TTUCCTL from either SM clears everything, two SM fence effect, own L1 stream removes about half of the rest), with a sharper edge.

## 2. Throughput_6, one triangle queries (tri1, 32 warps, one block per TPC through --sm-set, then every SM; 5 launches after the first, medians)
| G | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 85 (one per TPC) | 170 (every SM) |
|---|---|---|---|---|---|---|---|---|---|
| rays per cycle per SM | 0.1039 | 0.1036 | 0.1036 | 0.1035 | 0.1038 | 0.1041 | 0.1041 | 0.1041 | 0.0720 |
| aggregate x one SM (sum of block rates) | 1.0 | 2.0 | 3.99 | 7.98 | 15.95 | 32.06 | 64.08 | 85.02 | 120.93 |
| peak blocks running at the same time (global timer; computed for G 4, 16, 85, 170 only) | | | 3 | | 10 | | | 64 | 125 |
Predicted: G = 1 0.100 to 0.108 (held), G = 85 82 to 85.5 times (held: 85.02), G = 170 1.25 to 1.50 times G = 85 (held: 1.42) and per SM 0.065 to 0.078 (held: 0.0720). BUT: a block lasts 3.9 microseconds and the blocks start over 8 to 13 microseconds, so the blocks did not overlap fully (peak concurrency 64 of 85 and 125 of 170; the harness's own overlap flag is 0 of 5 from G = 4). The aggregate numbers are sums of per block rates, a whole GPU rate only if the blocks overlap, so the claim "99.7 percent efficiency at 85 TPCs" is NOT established: what the data show is no slowdown per SM with up to 64 blocks running at once. For the sibling pair, 277 pairs whose two blocks overlapped 90 to 100 percent ran at 0.0705 rays per cycle per SM (pair 1.36 times one SM, 0.1039 alone), pairs overlapping 50 to 90 percent 0.0781 (139 pairs), 5 to 50 percent 0.0957 (9 pairs). 1.36 is the Ada short query pair (4070 Ti 1.38, triton 1.53). Not done: a short query run with every block resident together for long enough (needs more rays per block, as in a persistent loop build; the stage 1 loop builds are not ported to SM120).

## 3. Memory_1, TTU half (second trace, R distinct random rays on soup:100000, warps = max(1, R/32), 11 launches; L1 about 90 KB against --smem 98304 reserved)
| R | 1 | 4 | 8 | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|---|---|
| L1 big (cycles) | 2,529 | 6,006 | 6,382 | 8,411 | 9,622 | 12,923 | 18,506 | 23,058 |
| L1 small | 5,935 | 15,467 | 14,993 | 16,029 | 16,344 | 15,551 | 23,589 | 31,530 |
| ratio | 2.35 | 2.58 | 2.35 | 1.91 | 1.70 | 1.20 | 1.27 | 1.37 |
Predicted: R = 1 0.95 to 1.15 (MISSED: 2.35), R = 4 and 8 1.0 to 1.8 (MISSED: 2.58, 2.35), R of 32 and above 1.4 to 3.2 (32 held at 1.70; 64, 128, 256 MISSED, 1.20, 1.27, 1.37, below 1.4), ordering "does not decrease by more than 10 percent from R = 8 on" (MISSED: 2.35 down to 1.20). The Ada picture (no effect while the rays touch one path, a 1.9 to 2.5 times slowdown once they do not) does not carry over: on the 5090 the reservation slows even ONE ray's second trace by 2.35 times, and the ratio falls with R. I had seen, before writing the predictions, the Memory_2 latency script (16 warps) giving 1.08 at one ray: that script and this one differ (warps, ray pool) and the two disagree at one ray (1.08 against 2.35); not reconciled. One ray's path (about 20 nodes) fits any L1, so the one ray slowdown is not capacity. Candidates, untested: the reservation changes what the unit's node cache can use (a minimum size or a fixed carve out), or L1 latency or banking changes with the carveout on SM120. The Memory_2 refetch counters (F408) say the 20 KB L1 refetches more than the whole second session from 4 rays up, but 0.56 already at one ray, in line with the one ray slowdown here.

## Tally
Held: 1b (5 of 5), 1c (2 of 2), 1d (2 of 2), 1a except one point, Throughput_6 tri1 (all 4 numbers, with the concurrency caveat). Missed: capacity at 263 KB, Memory_1 TTU ratios at R of 1 to 8 and 64 to 256 and the ordering. Group tally: node state 3 of 4 (prediction at least 3 held), Throughput_6 held, Memory_1 missed.
