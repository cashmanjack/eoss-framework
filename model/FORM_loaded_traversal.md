# Loaded traversal test: launch ramp or queueing (form and predictions fixed BEFORE the run, 2026-11-16, triton GPU 1, OptiX only, no TTU code of our own)

Question (F168 open item): the Mochi launch of 100,000 point queries (about 6 nodes, 1 candidate) takes 0.100 ms, 2 to 3 times the model and 27 times the unit capacity bound. Is the extra time a launch ramp (first touch, warps that start late, a fixed cost that does not grow with load) or queueing (time grows with the number of queries in flight)?

Test (ttu/bench/Load_1): custom primitive GAS of 100,000 disjoint boxes on a 50 x 50 x 40 lattice (half width 0.4, spacing 1). Query = point ray (tmax 1e-16) at the centre of one box, so exactly 1 candidate per query (checked by counting intersection program calls). Query count NQ swept from 32 to 1,638,400: 32, 1,024, 4,544 (one warp per SM), 9,088, 18,176, 36,352, 100,000, 409,600, 1,638,400, in two query orders (lattice order, random permutation). Per launch: in kernel global timer span (earliest warp start to latest warp end, from per warp records), per warp raygen cycles (clock64), SM id, intersection program calls. 2 untimed then 20 timed launches per NQ, median. Smallest launch first, each run under timeout -k 2 30, stop on first fault.

Derived: S(NQ) span; L(NQ) mean raygen cycles of a warp (from raygen start, so a late start is not in it); W(NQ) = sum of warp cycles / (142 x span in cycles) warps in flight per SM; c(NQ) = span cycles x 142 / NQ (cycles per query per SM). Fit S = a + b NQ over NQ >= 4,544; intercept share = a / S(100,000).

Predictions (written before the run; my reading of F168 is queueing):
- One warp per SM (4,544): span 4 to 7 microseconds, L about 11,000 cycles (the F168 small launch value, a different tree and 6 nodes; 1 candidate lattice may be a little lower).
- 100,000 queries: span 40 to 120 microseconds, L at least 3 times the one warp value, W 5 to 16.
- Intercept share at most 0.25; c roughly constant (max over min within 1.5) for NQ >= 18,176; L grows with NQ up to residency (22 warps per SM) and then stays.
- Lattice order against random order: random order slower (more distinct nodes in flight), at most 1.5 times at 100,000.

Decision rule (set now):
- QUEUEING if intercept share <= 0.25 AND L(100,000) / L(4,544) >= 2 AND c max/min <= 1.5 over NQ >= 18,176.
- LAUNCH RAMP if intercept share >= 0.5 AND L at 8 warps per SM within 1.25 of L at 1 warp per SM AND c falls at least 3 times from 4,544 to 100,000.
- Anything else: not separated; report what was ruled out and stop. Not tuned afterwards. Time box: one day.
- If the answer is queueing, the model needs a load dependent term (L grows with W), not a fixed ramp; if ramp, a fixed cost per launch of a (reported) is added to F127 and the 27 times is a launch cost, not a rate.
Caveats stated now: the 1 candidate lattice has fewer nodes per query than Mochi (about 6); this tests the mechanism, not Mochi's number. L2 is warm after the untimed launches (Mochi's 400 launch frames are warm too); a cold first launch is reported separately, not in the median.
