# Refit arm: is the Mochi excess tree quality? (form and predictions fixed BEFORE the run, 2026-11-16, triton GPU 1, OptiX only)

Follows F169: a fresh 1 candidate lattice takes 12 us (lattice order) to 17 us (random) for 100,000 queries; Mochi takes 100 us at 1.08 calls per query (frame 0) and 0.239 ms at 12.9 (frame 9).

Design fix (Jack): a refit of a static lattice does not degrade a tree, so the boxes move. The same 50 x 50 x 40 lattice (half width 0.4), 100,000 particles, is squeezed along z by a factor f (z' = f z, fixed small deterministic jitter of 0.05 in x, y, z so that nothing coincides), f = 1, 0.85, 0.7, 0.55, 0.4, 0.3, 0.2: neighbours overlap more and more, as Mochi's particles pile up. At each step the queries are the 100,000 moved particle centres (point rays, tmax 1e-16, lattice order) and three trees are measured at the same positions:
- R: ONE tree built with the update flag at f = 1 and refitted (optixAccelBuild UPDATE) at every following step: the Mochi path (degrading topology).
- U: a fresh build with the update flag at every step (the flag alone, topology fitted to the current positions).
- F: a fresh build without the flag at every step (the best tree for these positions).
Per tree and step: span (global timer), median of 20 launches after 2 untimed, calls per query, mean warp cycles, hit count. NQ = 100,000. Smallest launch first (a 32 query smoke run of the same program), timeout -k 2 30, stop on the first fault.

Predictions (written before the run; I expect tree quality is NOT the answer):
- f = 1: R, U, F all 12 to 16 us at 1.03 calls per query (R equals U there by construction).
- Spans rise with calls per query for all three trees; calls per query at f = 0.2 is far above 1 (several to tens). At f = 0.2 refit R at most 1.5 times F (log error at most 0.4); at the steps where calls per query is 1.05 to 1.15 (Mochi frame 0 has 1.08) R is 12 to 20 us.
- The update flag alone (U against F) changes the span by at most 15 percent.

Decision rule (set now):
- TREE QUALITY IS (PART OF) THE ANSWER if R at the step with calls per query 1.05 to 1.15 is at least 50 us (half of Mochi's 100), OR R / F >= 2 at any step.
- TREE QUALITY IS RULED OUT (for this kind of motion) if R stays within 1.5 of F at every step AND R at calls per query 1.05 to 1.15 is below 30 us. Then the next candidates are the heavier raygen and intersection program and the launch sequence (not run here).
- Anything between: not separated; report and stop. Not tuned afterwards. Time box one day.
Caveats stated now: this motion (z squeeze) is not Mochi's falling and piling, and calls per query depend on the squeeze, so spans are compared at equal calls per query, not equal step; the Mochi trace window at its frames 5 to 9 had 37 percent more warp calls than the replay counts (F168), a different quantity.
