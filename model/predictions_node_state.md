# Where the shared node state lives: predictions for tests (i) and (ii) (committed before any build or run)

Date 2026-11-16, triton GPU 1. NOT RUN YET.

## Hypotheses (as in F159, F161, F162)
- A: the traversal unit's node lines live in the L1 of both SMs of the TPC, and TTUCCTL is TPC wide (it clears the L1 of both SMs).
- B: the shared node state is a separate structure (not the L1 data cache). TTUCCTL clears it from either SM, and clears the issuing SM's own L1 (F142), but not the other SM's L1.
My lean: B at about 60 percent (an SM clearing another SM's L1 would be unusual; F159 says a gpu fence on both SMs is needed to remove all of the saving, which fits two layers). This is a lean, not a result.

## Test (i): does TTUCCTL from SM A empty the ordinary data in SM B's L1?
Setup. One TTU warp on SM 0 (A), builds as F161/F162 (sib = no TTUCCTL, sibO = one own TTUCCTL.IVALL after the first session and the flag1 store; both already validated, no new TTU sequence type). B is a new plain kernel (setup3 --sibling S:KB:0:0:6): one block of 384 threads on SM S; it loads a 64 KB buffer with ordinary cached loads (all four sectors of every line, 4 passes), sets ready, waits for flag1 (the TTU warp finished its first session), waits D = 100 us on the global timer (no loads, no stores, no fence), then times one more pass over the 64 KB (clock64 around the pass, whole block), then stores flag2 (relaxed). No fence anywhere in B except arm C. TTU warp scene par:1024:s (68 KB), 5 repetitions x 30 launches, rotating arm order, medians.
Arms (reload cycles of B per pass):
- N: sib build, B on SM 1 (the TPC sibling), nothing: warm level W (every line an L1 hit).
- C: sib build, B on SM 1, B executes fence.acq_rel.gpu before the reload (compiles to MEMBAR, ERRBAR, CCTL.IVALL): cold level K (positive control; F157).
- P: sibO build (TTUCCTL from SM 0), B on SM 1: the test.
- X: sibO build, B on SM 4 (another TPC): control for TPC wide.
Prediction before running: K is at least 3 times W (otherwise the reload is not sensitive and nothing is concluded). Define g = (P minus W) divided by (K minus W).
- A predicts g at least 0.8 (B goes cold), and X warm (g at most 0.2: TPC wide, not GPU wide).
- B predicts g at most 0.2 (B stays warm) and X warm.
- 0.2 to 0.8: partial, reported without interpretation.
Race rule decided now: if P is warm (g at most 0.2) I repeat P with D = 400 us; the reading "B stays warm" stands only if both are warm. If X is cold (g at least 0.8) the effect is not TPC bound and I report that instead.
Run rules: stage 0 first (NOP for TTUCCTL, flag logic live), check_seq on both stages, one stage 1 launch, then 5 x 30; timeout -k 2 30; GPU 1 idle; stop at the first fault, hang, timeout, or a launch where B did not land or did not finish. Raw output timestamped, mode x.

## Test (ii): stream 1 MB of ordinary loads through the TTU's own SM between the two sessions
Setup. One TTU warp on SM 0, two sessions of the same ray, scenes par:1024:s and par:2048:s (68 and 132 KB), 5 x 30, rotating order, saved fraction = 1 minus second over single (as F159). Arms:
- single, twice: no loads, nothing between (references; saved 0.62 / 0.54 expected).
- thr: build_patch --twice --thrash 256 --sectors 4 (the TTU warp itself, 1,024 ordinary loads, 256 pages of 4 KB per load batch, every sector of 8,192 lines = 1 MB), between the sessions, outside any session.
- dly: same build without the loads and a counted ALU loop of the same duration (new build_patch --delay N, no memory access; N calibrated in stage 0 so the kernel duration matches thr within 5 percent, events, medians of 30).
- bypass (extra control, only if the build fits): the same 1,024 loads that skip L1 (--bypass): same memory traffic, no L1 allocation.
Predictions (saved fraction at 68 / 132 KB):
- A: thr about 0.26 / 0.25 (accept 0.15 to 0.38 / 0.14 to 0.36): the L1 part of the saving is lost, as for a gpu fence on the own SM (F159 f), and dly and bypass stay at 0.55 to 0.64 / 0.48 to 0.56.
- B: thr 0.50 to 0.64 / 0.44 to 0.56, within 0.08 of dly (nothing lost).
- Between: partial, reported without interpretation.
Rule: only thr compared with dly counts (not with the single session); if dly itself is below 0.50 / 0.44 the delay alone costs something and the comparison is thr against dly.
Run rules as (i): stage 0 first, check_seq, smallest launch first, timeout -k 2 30, GPU 1, stop on first fault.

## What survives
If (i) gives g at least 0.8 (P cold, X warm) and (ii) thr near 0.26: A. If (i) g at most 0.2 (both D) and (ii) thr equal dly: B. Any mixed result: reported as not separated, no third hypothesis added after the fact.
