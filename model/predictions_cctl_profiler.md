# TTUCCTL as a profiler, and TTUCCTL modes (prepare only): predictions and plan, committed before any build or run

Date 2026-11-24, triton, GPU 1 (CUDA_VISIBLE_DEVICES=1, TTU_GPU=1). Nothing run yet.

## Part A: the profiler method (stage 1 allowed: only builds that already ran on triton)

Method. Run the same TTU work twice in one warp (two sessions, phase 1 then phase 2), once with nothing between the phases and once with
one TTUCCTL.IVALL between them. Cycles of phase 2 without it: T_keep. With it: T_flush. Share of phase 2 that comes from cached
node state: C = 1 - T_keep / T_flush. T_flush should equal the cycles of a fresh single session T_single (checked as a control).

Builds (all three already ran on triton, F140, F150, F157 to F162): `single` = C, `twice` = C + --twice, `cctl` = C + --twice --cctl, with
C = --clock --remap --compact-out --regs 104 --smfilter, same flags as ttu/bench/Node_4090/node_run.py. Scenes par:N:s, N = 1,024
(68 KB), 2,048 (132 KB), 4,096 (263 KB). One warp, one block on SM 0, 3 repetitions of 30 launches per cell, medians, rotating cell order.
Stage 0 of every build first (30 launches), then one stage 1 launch per arm at N 16 and N 1,024, then the matrix. Every run under run.py's
timeout 30, only when GPU 1 is idle, stop at the first bad exit. check_seq ALL PASS on all six builds first.

Worked example scope (what it is not). Phase 2 is the SAME ray as phase 1, so it is the best case for node reuse: C is an upper bound for
a multi query run whose queries overlap less. A multi query run with different rays and a TTUCCTL inside a loop is a new sequence class
(not validated); it is listed in the plan, not run. TTUCCTL also empties the L1 data cache (F142), so C is "node state plus L1 held lines",
not nodes alone.

Predictions (from F150 triton and F256 on the 4090, which matched triton in cycles):
- T_flush equals T_single within 1.0 percent at all three sizes (F161, F162: 100,370 against 100,545 at 68 KB).
- C = 0.62 at 1,024 (accept 0.56 to 0.68), 0.54 at 2,048 (accept 0.47 to 0.61), 0.23 at 4,096 (accept 0.15 to 0.31).
- Order C(1,024) > C(2,048) > C(4,096): holds.
- Spread of C over the 3 repetitions at most 0.02 at every size (profiler resolution).
- 0 faults, exactly one TTU block per launch in every launch.
What counts as failure: T_flush more than 1.0 percent away from T_single (then TTUCCTL is not a clean "remove the cache" switch in this
build); any C outside its accept range (reported as a miss, not refit).

## Part B: TTUCCTL encoding fields and modes (prepare only; stage 1 NOT run, needs Jack's approval)

From the EoSS spec JSON (JSON/SM89/json_89_with_conditions.json, class ttucctl_; SM120 identical field positions) and the EoSS paper
(Table of standalone latencies: TTUCCTL 10 cy; "executes in its default configuration but faults when patched with an active scheduling
scoreboard write"):

| field | bits | spec |
|---|---|---|
| opcode | 91, 11:0 | 0x3d5 |
| Pg, Pg_not | 14:12, 15 | predicate, default PT |
| ivall (named ftz in the table) | 80 | the only modifier: table IVALLONLY_ttucctl has one value, IVALL = 1 |
| req_bit_set | 121:116 | scoreboards to wait for |
| src_rel_sb | 115:113 | read scoreboard to release (7 = none) |
| dst_wr_sb | 112:110 | write scoreboard (7 = none); the paper: faults when active |
| pm_pred | 103:102 | performance monitor predicate (default PMN) |
| opex | 124:122, 109:105 | batch_t, usched_info (default NOP, DRAIN) |

So the spec has NO mode besides invalidate all. SM89 and SM120 share the fields. Anything else is an unspecified value of a field, so
the variants below are probes of undefined encodings, ranked by risk, not "modes from the spec".

Variants (the baseline stage 1 `cctl` build with only these bits of the one TTUCCTL changed; stage 0 is the same binary for all, the
slot is a NOP there):
- V0 baseline: ivall = 1 (validated).
- V1 ivall = 0 (bare TTUCCTL, no modifier; not in the spec's table). Risk: could fault or be a no op. Prediction: no op or fault, I cannot
  rank them; if it runs, the saved fraction equals the `twice` arm (0.62 at 1,024), not 0.00.
- V2 predicated off (@!PT, ivall = 1): a control. Prediction: no effect, saved fraction as `twice`, 0.62 at 1,024 (accept 0.56 to 0.68).
- V3 src_rel_sb = 0 (a read scoreboard released, nothing waits on it). Risk: low to medium (the paper only reports a fault for the write
  scoreboard). Prediction: runs, saved fraction 0.00 at 1,024 (same as V0).
- V4 dst_wr_sb = 0. The paper reports a fault. Built and checked for completeness, NOT in the run plan.
Not built: pm_pred, opex, req_bit_set variants (no informative outcome expected, and req_bit_set waiting on a scoreboard nobody sets
could hang the warp).

check_seq cannot pass the variants: it requires the TTUCCTL bytes to equal the EoSS encoder output for IVALL (rules `cctl: bytes equal`,
`cctl: unpredicated`). The wrapper ttu/bench/Cctl_1/check_variants.py instead (1) runs check_seq on the baseline and demands ALL PASS,
(2) demands the variant binary to differ from the baseline in exactly the 16 bytes of the one TTUCCTL, (3) decodes the variant with the EoSS
SM89 encoder fields and prints the changed bits. check_seq itself is untouched. The variants carry no `.checked` marker, so run.py refuses
them: stage 1 cannot start by accident. A check_seq rule set for the variants is Jack's decision.

Plan if approved (not started): V2, then V3, then V1, last of all, one at a time: stage 0 (once), one stage 1 launch at N 1,024, then 3 x 30
launches at 1,024 and 2,048; stop at the first fault. V4 never.
