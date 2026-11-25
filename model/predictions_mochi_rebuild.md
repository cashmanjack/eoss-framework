# Mochi_1: periodic BVH rebuild, predictions and rule (tethys, RTX 4090, GPU 1), 2026-11-25

Written BEFORE any patched binary was built or run. OptiX only, no TTU code. Baseline: F453 / F452 (stock Mochi, 100,000 particles, 10 frames of 400 sub steps: build 1,179 to 1,184 ms, collision 299 ms, update 61 ms, total 1,542 to 1,544 ms).

## What the stock program does (read from src/s23-3Dparticlesim/hostCode.cu, no run)
The task says "rebuild periodically instead of every frame, refit in between". Stock Mochi does NOT rebuild every frame. It builds the BVH once at start and then calls owlGroupRefitAccel on the particle group and the instance group at EVERY sub step (4,000 times in 10 frames, 200 more in the warmup). The "build" term of F453 is therefore refit time (host side: temp buffer allocation, bounds kernel, OptiX update, sync). The change that exists is the reverse: refit by default, a full rebuild (owlGroupBuildAccel on both groups, then owlBuildSBT because the raygen record holds the traversable handle) every N sub steps. Interval N counts sub steps globally (rebuild when sub step counter mod N is 0, counter starts at 0 in the timed loop). N = 0 is the stock behaviour. A "frame" is 400 sub steps, so N = 400 is one rebuild per frame.

## Change (one thing)
Env var MOCHI_REBUILD_EVERY=N in a patched copy of hostCode.cu (private diff in notes/). Everything else unchanged. Baseline is the stock binary AND the patched binary at N = 0 (they must agree within 2 percent on total; if not, the patch itself costs time and that is reported first).

## Intervals
N in {0 (stock), 400, 100, 25, 10, 1}. Five timed runs each, warmup process first, interleaved by round, full clock check as in Base_1.

## Correctness (every frame, not a selection)
Observer build (COUNT_COLLISIONS=1, same patch) dumps particle state and the program's collision count for the launch of sub step s at s mod 16 = 15 of every frame (25 per frame, 250 per run, including the last sub step of every frame, 10 of 10 frames). The CPU counts the same ordered pairs with the device test (the Base_1 checker) on each dump. Every dump of every configuration is compared; none is skipped. The timed runs use the COUNT_COLLISIONS=0 build.

## Predictions (central, range in brackets)
P1 Counts: program equals CPU in every dump of every N (all 250 per run). Refit and rebuild both give valid bounds, so only speed can change, not the answer. Confidence 95 percent. A mismatch anywhere would be a bug in the patch (stale handle, missing SBT rebuild), reported as such.
P2 Collision launch time falls with more frequent rebuilds, monotone in 1/N: N = 1 gives 0.6 times stock collision time (0.4 to 0.95); N = 400 gives 0.95 (0.85 to 1.02). Reason: the stock tree is built from the initial positions only and refit since (F170: refit trees are worse than fresh ones), but the stock collision term is already small (75 microseconds per launch).
P3 Build term (rebuild plus refit) rises with 1/N. One rebuild costs 4 times one refit including SBT (1.2 ms vs 0.3 ms per call pair; range 2 to 10 times). Build term in ms: N 400 about 1,200 (1,190 to 1,250); N 100 about 1,240; N 25 about 1,400; N 10 about 1,800; N 1 about 5,000 (2,500 to 12,000).
P4 Total time: no N beats stock by 5 percent or more. The collision term is only 19 percent of the stock total, so even removing half of it gains at most 10 percent before paying for rebuilds. Best N is 400 or stock, within 3 percent of stock; N = 10 and N = 1 are slower than stock by 20 percent or more.
P5 The patched N = 0 binary matches the stock binary total within 2 percent.

## Rule (fixed now)
Win: total median at most 0.95 times stock median at that N with all dumps equal to the CPU. Otherwise "no gain". Tradeoff reported for all six N with build, collision, update and total medians and the spread over 5 runs. If a count mismatch appears, that N is reported as incorrect regardless of time.

## Not done here
No sub step adaptive policy (rebuild when quality drops), no fixed AABB padding, no compaction or build flag change (F172 ruled them out for Mochi), no other particle counts.
