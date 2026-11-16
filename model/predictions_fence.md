# Fence decomposition: predictions (committed before any run)

Date 2026-11-16, triton GPU 1. Setup: F155 / F157 (one TTU warp on SM 0, two sessions of the same ray, a plain block of 384 threads between them; BVH par:N:s at 68 KB (N 1,024) and 132 KB (N 2,048); saved fraction = 1 minus second session over single session, medians of 5 x 30 launches). Baselines measured before: no sibling 0.62 / 0.54; fence.acq_rel.gpu on the sibling SM (F157 c) 0.27 / 0.22; control SM on another TPC 0.61 / 0.54.

Hypothesis (Jack): the traversal unit's node lines are spread across both L1 data caches of the TPC (the unit of one SM reads through its own L1 and the sibling's), and a fence.gpu invalidates (CCTL.IVALL) the L1 of the SM that runs it.

Arms (all between the two sessions, plain block with one fence by one thread):
- e: fence.acq_rel.cta on the sibling SM (SM 1). Compiled form to be read from the SASS before the run.
- f: fence.acq_rel.gpu on the TTU's own SM (SM 0) only.
- g: fence.acq_rel.gpu on both SM 0 and SM 1 (two blocks).
- h: TTUCCTL alone on the sibling: NOT run if it needs a TTU sequence type that is not already checked (a TTUCCTL outside any session on a second TTU block is a new sequence type).

Predictions, saved fraction at 68 KB / 132 KB (no sibling 0.62 / 0.54 is the ceiling, the F157 sibling gpu fence 0.27 / 0.22 is the reference):
- e (cta fence, sibling): no loss, 0.62 / 0.54 (accept 0.55 to 0.65 / 0.47 to 0.57). Reason: a cta scope fence does not need to leave the SM, so I expect no L1 invalidate.
- f (gpu fence, own SM only): about half of the saving lost, 0.31 / 0.27 (accept 0.22 to 0.40 / 0.19 to 0.35).
- g (gpu fence on both): nearly all lost, 0.05 / 0.04 (accept at most 0.12 / 0.10).
Consistency check on the hypothesis: f should be about equal to the F157 sibling fence (0.27 / 0.22) because the TPC is symmetric; if f is near 0.62 / 0.54 (no loss) the hypothesis that the TTU SM's own L1 holds node lines is wrong, and if g equals f (no additional loss) the two L1s are not both holding lines.
Cycles are reported next to the fractions (single about 100,700 / 176,400; no sibling about 38,200 / 81,000).
