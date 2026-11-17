# Between launch sequence arm (form and predictions fixed BEFORE the sequence runs; 2026-11-16, triton GPU 1, OptiX only)

Follows F171 (the replica with Mochi's program and instance level takes 11.3 us, Mochi 100 us at frame 0).

## Checks done first (results known before the predictions below)
1. Clock watch (cheap, unpatched Mochi, 40 frames, nvidia-smi SM clock every 20 ms, notes/q31): 498 of 589 samples during the run at 2,700 MHz, 66 at 2,685, 25 at 2,505 (the 71 samples at 210 MHz are before and after the run); max clock 3,105 MHz. No sag at the 20 ms resolution (gaps between launches are tens of microseconds, below what nvidia-smi can see). In kernel check from the F168 data (run C, frame 0): the longest warp's raygen is 242,321 cycles and the launch span 100.4 us, so the SM clock inside the launch is at least 2.41 GHz (the warps of the last wave end at about the end of the span), not a low clock. Clock ruled out as the explanation (the 2.5 GHz assumed in F168 is within 8 percent).
2. Nodes per query of the replica (replay of the replica's own GAS dump with the F165 tool, model/predict_mochi.py): 6.25 nodes, 1.00 leaf slots, 1.08 primitives per query, lane efficiency 0.602: IDENTICAL to Mochi frame 0 (6.25, 1.00, 1.08, 0.602). The replica replicates the traversal work.
3. OWL flags (owl/UserGeomGroup.cpp line 167, InstanceGroup.cpp line 235): accelOptions.buildFlags = this->buildFlags for the GAS and the instance group (staticBuildOn), and Mochi passes OPTIX_BUILD_FLAG_ALLOW_UPDATE: nothing is ORed in. The PREFER_FAST_TRACE | ALLOW_UPDATE at InstanceGroup.cpp line 481 is in motionBlurBuildOn, not used. The replica builds the same flags.
4. A difference seen in the data (not yet explained): Mochi's raygen cycles per warp at frame 0 are very skewed: 10th percentile 17,147, median 39,658, 90th 189,321, max 242,321 (mean 78,314), growing with the warp index (correlation 0.65, mean per tenth of the launch 31,000 to 162,000). The replica: 10th 15,800, median 19,200, 90th 22,200, max 26,800 (narrow). So the excess is a tail of long warps, not a uniform factor.

## Test (ttu/bench/Load_1/load3.cu with SEQ, WARM)
Replica (Mochi program, instance level, update flag only, real dataset in Morton order, 100,000 queries). Before each launch, as Mochi's substep: SEQ 0 nothing (back to back); SEQ 1 an update kernel (reads and writes every particle and its force, clears the force, values unchanged) and a host synchronisation; SEQ 2 the update kernel, a bounds kernel (centre -/+ 2 radius), optixAccelBuild UPDATE of the GAS, optixAccelBuild UPDATE of the instance group, with a synchronisation after each. 200 warmup iterations of the same sequence (as Mochi), then 400 timed launches, median span; per launch the 10th, 50th, 90th percentile and max of the warp raygen cycles. Smallest launch (32 queries, 3 iterations) first, then the full; timeout -k 2 30, GPU 1, stop on first fault.

Predictions (written before these runs; the sequence is the leading remaining candidate but I do not trust it):
- SEQ 0 in this process: 10 to 13 us (as F171).
- SEQ 1: 10 to 16 us.
- SEQ 2: 20 to 100 us, my centre 40 us; the warp distribution widens (90th percentile / median at least 2).
Decision rule (set now), on SEQ 2: THE SEQUENCE IS THE ANSWER if the median span is >= 70 us; RULED OUT if <= 30 us; between: partial, report the share and the distribution and stop. If SEQ 1 alone is >= 30 us the update kernel (cache state) is the part. Not tuned afterwards. One day box.
Caveat stated now: the update kernel and refit here do not move the particles (velocity zero, values unchanged), so the tree stays the fresh build topology; Mochi's real refits follow a falling lattice (F166: topology constant).
