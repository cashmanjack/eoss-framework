# Arm (h) prediction: TTUCCTL.IVALL alone on the sibling SM (committed before any run)

Date 2026-11-16, triton GPU 1. NOT RUN YET: needs Jack's approval after reading this.

Setup (as F157 / F159): one TTU warp on SM 0, two sessions of the same ray, scene par:N:s at 68 KB (N 1,024) and 132 KB (N 2,048), saved fraction = 1 minus second session over single session, medians of 5 x 30 launches. New: between the sessions, a second block on SM 1 (the TPC sibling) executes exactly one TTUCCTL.IVALL and nothing else (no TTUOPEN, MACROFUSE, TTUST, TTUGO or TTULD; build_patch --sibccctl, host --sibling 1:4096:0:0:5, --blocks 142 --sms 0,1 --warps 1). It spins on flag1 (set by the TTU warp after its first session), issues the TTUCCTL, then sets flag2, which the TTU warp waits for before its second session. This is a new sequence type: a TTUCCTL with no session on its warp has never run.

References (F157, F159, saved fraction at 68 / 132 KB): no fence 0.62 / 0.54; cta fence on the sibling 0.61 / 0.49; gpu scope fence on the sibling 0.26 / 0.23; gpu scope fence on the TTU's own SM 0.26 / 0.25; both SMs 0.00 / 0.00.

## Prediction
Saved fraction about 0.27 / 0.24, the same as the sibling gpu fence (accept 0.15 to 0.40 / 0.13 to 0.36). Reason: F142 showed TTUCCTL.IVALL empties the L1 data cache and the unit's node state of the SM that issues it. The sibling gpu fence (an L1 invalidate plus a memory barrier) already removed 58 percent when issued on the sibling, so a bare L1 invalidate there, with no memory barrier, should remove about the same if the invalidate is the effective part.

## What each outcome means (decided now, before the run)
- 0.15 to 0.40 / 0.13 to 0.36 (about half removed): TTUCCTL on the sibling clears the sibling's share of the saved node state. Matches the reading that each SM of the TPC holds part of it. Also separates the two parts of the fence: TTUCCTL has no memory barrier, so the invalidate alone is enough and the gpu barrier is not needed.
- At most 0.12 / 0.10 (nearly all removed): the invalidate by TTUCCTL is TPC wide (it also clears the state seen by the TTU on SM 0). Would contradict the F159 result that a fence on one SM removes only half, unless TTUCCTL acts differently from CCTL.IVALL.
- At least 0.55 / 0.47 (nothing removed, inside the no fence range of e): a TTUCCTL outside a session on a warp with no session does nothing to the node state seen by the other SM. Then the effect of the sibling gpu fence comes from its L1 invalidate or barrier, not from the TTU's own cache control.
- 0.41 to 0.54 / 0.37 to 0.46: partial, reported as such, no interpretation.
Cycles reported next to the fractions (single about 100,400 / 176,100).

## Run rules
Stage 0 first (same build, NOP in place of the TTUCCTL, flag logic live, no TTU instruction), then stage 1 one launch, then 5 x 30 per size in rotating order against the single and twice arms. Every run under timeout -k 2 30, GPU 1, only when idle, smallest launch first, stop at the first fault, hang, timeout, or a launch where not exactly one TTU block ran. Raw output timestamped, opened with mode x. If TTUCCTL without a session faults or hangs the GPU: stop, report, no retry. No tuning after seeing the result; any second attempt is labelled.
