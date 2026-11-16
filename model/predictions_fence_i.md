# Arm (i) prediction: TTUCCTL.IVALL from the TTU warp's own SM (committed before any build or run)

Date 2026-11-16, triton GPU 1. NOT RUN YET.

Setup (as F157, F159, F161): one TTU warp on SM 0, two sessions of the same ray, scene par:N:s at 68 KB (N 1,024) and 132 KB (N 2,048), saved fraction = 1 minus second session over single session, medians of 5 x 30 launches, rotating arm order. Same sibling test build family (build_patch --sibling), new option --ownccctl: the TTU warp itself executes exactly one TTUCCTL.IVALL after its first CLOSE and its flag1 store, outside any session, before its spin on flag2; a plain block on the sibling SM (SM 1) polls only (F157 arm b, mode 2, no fence). Arms: single, twice, b1 (polling sibling, no TTUCCTL), i (own TTUCCTL, polling sibling).

References (saved fraction at 68 / 132 KB): no fence 0.62 / 0.54; polling sibling 0.61 / 0.49; sibling gpu fence 0.26 / 0.23; own SM gpu fence 0.26 / 0.25; both fences 0.00 / 0.00; sibling TTUCCTL alone (F161) 0.00 / 0.00. F140 and F142 (other build): TTUCCTL on the own SM takes the second session back to a first session cost.

## Prediction
Saved fraction 0.00 / 0.00 (second session as slow as a fresh first one, about 100,400 / 176,100 cycles). Accept at most 0.10 / 0.08. Reason: F161 says TTUCCTL from the sibling alone clears everything, which reads as TPC wide node state; the issuing SM cannot be weaker than the other SM, and F140 / F142 already saw a complete loss from the own SM.

## What each outcome means (decided now)
- At most 0.10 / 0.08: consistent with TTUCCTL clearing TPC wide state from either SM (F161). Reported as the symmetric half of F161; no new interpretation.
- 0.11 to 0.40 / 0.09 to 0.36: the own SM TTUCCTL leaves part of the saving: TTUCCTL is not symmetric between the two SMs. Would contradict F140 / F142 for this build; reported as such, build differences listed.
- At least 0.55 / 0.47 (nothing removed): TTUCCTL outside the build of F140 does nothing here; would indicate the sibling build (flag spin between the sessions) changes what TTUCCTL does. Stop and look at the SASS before any reading.
- 0.41 to 0.54 / 0.37 to 0.46: partial, reported without interpretation.

## Run rules
Stage 0 first (NOP in place of the TTUCCTL, flag logic live), check_seq on both stages, then one stage 1 launch at 68 KB, then 5 x 30 per size. Every run under timeout -k 2 30 (the harness run.py wrapper; I will call it under timeout), GPU 1, only when idle, smallest launch first, stop at the first fault, hang, timeout or a launch where not exactly one TTU block ran. Raw output timestamped, opened with mode x. No tuning after seeing the result; a second attempt is labelled.
