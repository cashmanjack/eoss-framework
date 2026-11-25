# Prediction before the single trace run of R3 C1 (2026-11-24, triton)
Run: /dev/shm/t9r/k_s1tr (stage 1, checked, 60 round cap, trace of 22 answer registers per session), C1: 4,096 points, r 0.08, seed 7, one warp, once.
In the earlier C1 run 28 of 32 lanes had fewer sessions than OptiX candidates, 24 lanes mismatched. Candidate sets differ by skipped candidates in the middle of OptiX's order (not a truncated tail).
Predictions:
1. Lanes that lose candidates: the same 24 lanes (the run is deterministic): 0.85. Sessions per lane equal the earlier run: 0.85.
2. First divergence from OptiX's candidate order is at session index 3 to 8 (1 based) in at least 20 of the 24 bad lanes: 0.6; never at session 1 or 2: 0.9.
3. At the diverging session the previous answer had both pending entry pairs nonzero (two entries at different levels): 0.55; only one pair: 0.3; none: 0.15.
4. (a) tmax: our R17/R31 never change (reloaded from the record); the answers' distance words are not used for tmax: tmax is NOT the cause: 0.9.
5. (b) stop rule ends a lane with traversal state remaining: a lane's last answer is the miss form while OptiX still has candidates for it: 0.5 as the main mechanism for lanes whose final answer is empty (all 24 end with the empty form: 0.7).
6. (c) inactive lanes lose state: divergence sessions fall after the first lane exits (session index > the smallest session count in the warp, 6): 0.45; not related: 0.4.
7. Another mechanism (an answer carrying a second candidate in words we do not read, e.g. the 0x310 or 0x380 words nonzero in diverging sessions): 0.3.
8. No fault, no hang: 0.95.
Fix to be proposed after reading the trace, with its own prediction, then stop for approval.

## Addendum 2026-11-24 (before the run; Jack's change of design)
The test is ONE warp with only lane 0 active, the 32 queries one after another as 32 launches of <<<1, 32>>> (lanes 1 to 31 leave before any load: check_seq rule M1), stage 0 with injected candidates first, then stage 1 once with the trace build. Not 32 warps. The same predictions as the proposal: all 32 queries exact 0.6; sessions equal OptiX candidates or + 1: 0.55; still inexact 0.4. The run also saves the GAS record table and OptiX's full candidate sequence per query, so a remaining mismatch can be traced to its first diverging session. If exact: lane interference in a shared session. If inexact: compare single lane sessions with R2's host rounds offline. No further TTU runs after this one.
