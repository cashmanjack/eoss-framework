# Predictions: rehosting R2 (candidates in rounds, set equality against the OptiX intersection call log), committed before any R2 run

2026-11-24, triton. R1 passed (model/results_rehost.md will hold the scored table; raw notes/rehost/r1_s1_box1_*): the emit built group returns the miss form with no candidate on outside and miss rays, and a candidate on inside and hit rays: tag 0x50 in the top seven bits of the first word of 0x300, hit flag 1, a record pointer (word pair times 8 = GAS + 0x180, the start of an 8 byte per primitive record table named by the GAS header at +0xa0), instance word echoed, pending entries zero, 1,227 to 1,242 cycles (a miss session 1,238 to 1,241).

## What runs (host rounds, one session per kernel launch, the same checked ttu_one SASS as R1)
Build: the R1 recipe (`--emit --clock`) on the rebuilt host (ttu_one SASS identical to the R1 kernel, checked by diff of cuobjdump). Each round: next query = previous answer's pending entries fed back with halves swapped (0x3C0 high to 0x180 low and so on, exactly as the triangle resume runs), tmax unchanged (the intersection program never reports, so nothing shrinks it), a lane is finished when an answer has no pending entry word. Candidate decode: tag 0x50 and hit flag 1; record = (R45:R44 times 8) minus (GAS + header word at +0xa0), primitive = word 0 of that 8 byte record, read from a host copy of the GAS.
One warp, all 32 lanes the same ray, GPU 1, `timeout -k 2 30`, who and nvidia-smi checked before each run, raw files never overwritten. Order (stop at the first fault, hang or wrong result): box2 with rays inside, inside2, xrow, outside; boxov:2 inside; boxov:4 inside and hit; boxov:8 inside; boxov:64 inside (64 nested boxes, several tree levels).
CPU expectation computed offline (float64 slab test): box2 inside {0}, inside2 {1}, xrow {0,1}, outside {}; boxov:K inside {0..K-1}; boxov:2 and 4 hit {0..K-1}.

## Pass rule (fixed now)
For every run and every lane: the multiset of decoded candidate primitives (in answer order) equals the OptiX call log of the same process (islog) AND the CPU expectation; no lane disagrees; no fault or hang; status no error. Order is reported (equal to the OptiX order or not) but is not part of the gate. Sessions per ray are reported. Any difference stops the series and R3 does not start.

## Predictions
1. Set equality (three way) in all 9 runs: 0.6. For boxov:64 alone: 0.5. Most likely failure: the swap rule for pending entries does not carry the remaining candidates after a candidate return (then fewer candidates than OptiX), or one answer can hold more than one candidate (a hidden candidate).
2. One candidate per answer (sessions = number of candidates, K for boxov:K, since I stop when no pending entry comes back): 0.65. Sessions K + 1 (a pending entry stays after the last candidate): 0.2. Fewer than K sessions: 0.15.
3. Pending entries are nonzero after every candidate except the last one: 0.75.
4. The order of candidates equals OptiX's call order, given set equality: 0.7.
5. The decoded primitive of a record equals the OptiX primitive index (the record table word 0 reads the same index OptiX reports): 0.85 given set equality.
6. Session cycles 1,100 to 1,800 for every session, including the later ones of a ray (each round is a new kernel launch, L1 cold): 0.7; a later session cheaper than 900: 0.2.
7. No fault, no hang: 0.95.
8. box2 inside2 returns prim 1 (record table entry 1 or 0 whichever the tree order gives): the decoded prim is 1: 0.85.
