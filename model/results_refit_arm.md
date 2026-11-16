# Refit arm: is the Mochi excess tree quality? (F170; form, predictions and decision rule: model/FORM_refit_arm.md, committed before each run)

2026-11-16, triton GPU 1, OptiX only (ttu/bench/Load_1 load2.cu; raw private in notes/q29). 100,000 point queries, 100,000 disjoint boxes (half width 0.4), span = global timer span (1.024 microsecond steps), median of 20 launches. R = tree built with the update flag once and refitted at every step; U = fresh build with the update flag; F = fresh build without. Calls per query in brackets.

## Motion 1: z squeeze (order preserving; overlap grows, the tree stays good)
| step | f (squeeze factor or scramble cells) | R span us (calls/query) | U span us (calls) | F span us (calls) | R / F span | R warp cycles | F warp cycles |
|---|---|---|---|---|---|---|---|
| 0 | 1.00 | 13.31 (1.065) | 13.31 (1.065) | 13.31 (1.065) | 1.00 | 14255 | 14086 |
| 1 | 0.85 | 13.31 (1.065) | 13.31 (1.059) | 12.29 (1.059) | 1.08 | 14266 | 14123 |
| 2 | 0.70 | 13.31 (1.065) | 13.31 (1.074) | 13.31 (1.074) | 1.00 | 14253 | 14393 |
| 3 | 0.55 | 13.31 (1.065) | 13.31 (1.080) | 13.31 (1.080) | 1.00 | 14242 | 14610 |
| 4 | 0.40 | 23.55 (2.172) | 22.53 (2.227) | 22.53 (2.227) | 1.05 | 24971 | 23337 |
| 5 | 0.30 | 25.60 (3.097) | 24.58 (3.105) | 25.60 (3.105) | 1.00 | 27233 | 26638 |
| 6 | 0.20 | 35.84 (4.169) | 33.79 (4.271) | 32.77 (4.271) | 1.09 | 37940 | 36214 |

## Motion 2: scramble (each particle displaced at random by up to f cells per axis; the refitted tree degrades, density about constant)
| step | f (squeeze factor or scramble cells) | R span us (calls/query) | U span us (calls) | F span us (calls) | R / F span | R warp cycles | F warp cycles |
|---|---|---|---|---|---|---|---|
| 0 | 0.00 | 13.31 (1.065) | 13.31 (1.065) | 13.31 (1.065) | 1.00 | 14257 | 14099 |
| 1 | 0.50 | 21.50 (1.391) | 19.46 (1.316) | 19.46 (1.316) | 1.11 | 21459 | 18894 |
| 2 | 1.00 | 27.65 (1.746) | 20.48 (1.530) | 20.48 (1.530) | 1.35 | 27744 | 21714 |
| 3 | 2.00 | 37.89 (2.219) | 22.53 (1.548) | 22.53 (1.548) | 1.68 | 40182 | 22986 |
| 4 | 4.00 | 104.45 (4.122) | 22.53 (1.509) | 22.53 (1.509) | 4.64 | 119428 | 23858 |
| 5 | 8.00 | 528.38 (13.283) | 22.53 (1.424) | 22.53 (1.424) | 23.45 | 679888 | 23985 |
| 6 | 16.00 | 2890.75 (53.711) | 22.02 (1.280) | 21.50 (1.280) | 134.43 | 3701145 | 22652 |

Verdict by the rule set before the runs: squeeze: tree quality ruled out for that motion (R within 1.09 of F at every step, 13.3 us at 1.065 calls per query), but the motion is order preserving so a refit cannot degrade (stated in the addendum). Scramble: TREE QUALITY IS PART OF THE ANSWER (R / F >= 2 from f = 4, 4.6 times; R at calls per query 1.05 to 1.15 was 13.3 us, so the first branch of the rule did not fire).

Predictions: squeeze held (R at most 1.5 times F, R 12 to 20 us at calls 1.05 to 1.15, U within 15 percent of F). Scramble: MISSED on size. Predicted R / F 1.5 to 4 at f = 8 and 16; measured 23 and 134 times (528 and 2,892 us). Predicted the calls to agree between trees; they do not: the refitted tree makes 13.3 and 53.7 calls per query at f = 8 and 16 against 1.4 and 1.3 for a fresh tree at the same positions (its leaf boxes overlap, so a query visits many more leaves and runs the intersection program for their primitives). F and U within 1.05 of each other (the update flag alone costs nothing).

What it shows
- A refitted tree that has lost spatial order is slow in a way the tree-fresh model cannot see: at the same positions, same queries, the refit tree takes 4.6 times (104 us) at f = 4 and 23 times (528 us) at f = 8. The excess scales with the extra leaf calls and nodes it visits (warp cycles 119,000 and 680,000 against 24,000).
- This reaches Mochi's range: R takes 104 us at 4.1 calls per query and 528 us at 13.3, Mochi 0.239 ms at 12.9 calls (frame 9), so Mochi's growth over frames 5 to 9 can be a degrading refit tree. The fresh tree stays at 22 us at every f.
- It does NOT explain Mochi's frames 0 to 4: there Mochi has 1.0 to 1.08 calls per query (a falling lattice, replay exact, F165) and takes 100 us, while the refit tree at 1.065 calls takes 13 us. A tree at calls about 1.08 does not slow down by refitting; so the 100 us at frame 0 (and the 27 times of F168) needs the other candidates: Mochi's heavier raygen and intersection program, the launch sequence, how its tree is built at frame 0 (unknown here), the query positions.

Not tested: Mochi's real motion (falling, piling, bouncing); the nodes visited per query (not counted here; the warp cycles are the evidence); the heavier program and the launch sequence.
