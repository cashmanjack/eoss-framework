# Rehosting R0 to R2: results (2026-11-24, triton, GPU 1, one warp per run, no fault, no hang)

Predictions: model/predictions_rehost_r1.md and predictions_rehost_r2.md, both committed before the runs they cover (15eb523, c543f6c). Raw files (private, never overwritten): notes/rehost/r1_*, r2_*. Code: ttu/rehost, ttu/harness/setup3 (build_patch --emit, check_seq E rules, setup3 --aabb), driver ttu/rehost/r_run.py and r2_series.py.

## R0 (offline)
- The 19 instructions of the SM89 traversal group and the ISETP that selects the P0 true path are emitted by `ttu/rehost/emit_group.py` from the EoSS encoder alone (no dump, no OptiX code, no binary read; operands, predicates, scoreboards and stall counts are tables in the module). An offline comparison (private script, never an input of a build) finds all 20 instructions byte identical to the ones the earlier builds copied from OptiX. So provenance changes, the bytes do not: an emit build and the earlier copy build of the same configuration are identical files (cmp).
- check_seq got rules E1 to E5 (plain SM89 only; the 19 instructions equal the encoder output regenerated at check time; a second, independent table of offsets, predicates, registers and control fields; the ISETP; no dump key in the layout). Negative tests: an emitter with one store offset changed passes E2 (it regenerates the same wrong bytes) and fails E3; one predicate flipped fails E3. Diff: notes/check_seq_emit_20261124.diff (additive, every new line is inside `if meta.get('emit')`).
- Regression, same host binary, HEAD scripts against new scripts: 24 build configurations x stage 0 and 1 (plain, clock, sync, regs, the 7 node state builds, dualgo, ias, ias remap, cesar8 1 and 2, smfilter, gtime, twice, remap, slim loop, delay, stream, persistent outer): all 48 binaries byte identical, layout json identical, check_seq output identical. Not covered: SM120 builds (no SM120 host here), the hash listed variant builds (their files are gone from /dev/shm; their code path is untouched). The host change leaves the ttu_one kernel identical instruction for instruction (cuobjdump diff).
- Host: `--aabb` builds a custom primitive GAS from the scene (the box of each scene triangle's vertices; scenes box1, box2, boxov:K), an intersection program that never reports and logs its calls per ray. Same pipeline constant and handle type as the triangle pipeline (the host's own guards pass), GAS 896 bytes for 1 to 4 boxes.
- GAS layout (three scenes, stage 0 copies): a leaf slot for a custom primitive has the type byte 0x41 and byte 6 zero; the header names (at +0xa0) a table of 8 byte records, one per primitive: word 0 reads the primitive index (0; 0,1; 3,2,1,0 for four nested boxes), word 1 zero (SBT index). The leaf block holds box data and no primitive index.
- Not done: R0.1 (compile only OptiX's wrapper around an intersection program and read it): needs a GPU context, not offline; deferred. The wrapper's status codes are not needed by a rehosted kernel that inlines the program.

## R1: one box (box1), emit build, 3 launches per ray, all 32 lanes and all launches identical
| ray | OptiX intersection calls | answer | session cycles |
|---|---|---|---|
| outside (point in no box) | 0 | miss form (first result word pair all ones, hit flag 0, everything else 0) | 1,239 |
| miss (crosses no box) | 0 | miss form | 1,238 |
| inside (RTNN point query) | 1 (primitive 0) | candidate | 1,227 |
| hit (crosses the box) | 1 (primitive 0) | candidate | 1,242 |
A candidate answer: bit 31 of the second word of the first result pair set and its top seven bits 0x50 (a tag); hit flag 1; the 0x300 high pair times 8 is the address of the primitive's record in the GAS (GAS + 0x180 in both runs, record word 0 = 0 = the primitive OptiX reported); the 0x310 words are zero (no leaf block address); the instance word is echoed; both pending entries zero. inside and hit differ only in words that hold process addresses (the pointer value and the echoed instance word). R1 passes its pre fixed rule (A, B, C, D).

## R2: rounds of sessions, one candidate per session, set equality (ours, OptiX call log, CPU float64 slab test) on all 32 lanes
| scene, ray | candidates (OptiX order = ours) | sessions | pending after each candidate | cycles first / later | result |
|---|---|---|---|---|---|
| box2 inside | {0} | 1 | none | 1,242 | equal |
| box2 inside2 | {1} | 1 | none | 1,227 | equal |
| box2 xrow | 0, 1 | 2 | yes, then none | 1,226 / 918 | equal |
| box2 outside | none | 1 | none | 1,242 | equal |
| boxov:2 inside | 1, 0 | 2 | yes, then none | 1,241 / 927 | equal |
| boxov:4 inside, hit | 3, 2, 1, 0 | 4 | yes x3, then none | 1,243 / 927 to 1,244 | equal, equal |
| boxov:8 inside | 7 down to 0 | 8 | yes x7, then none | 1,242 / 926 to 1,244 | equal |
| boxov:64 inside | 63 down to 0 | 65 | yes after all 64 | 1,241 / 917 to 1,244, last 3,412 | equal |
For 64 nested boxes the 64th candidate still came back with a pending entry (a root like word with bit 31), the 65th session returned the miss form (no candidate, 3,412 cycles): one extra session, as OptiX's loop would also need. Stopping when no pending entry comes back is right for K up to 8 and costs one extra session at 64. All candidates were delivered one per session; no answer held two.
The candidate order of the nested boxes is the reverse of the box index (3, 2, 1, 0) in both OptiX and ours.

## Predictions scored
R1 (12): held 1, 2, 3 (tag 0x50: 0.55), 4 first part (pointer inside the GAS: 0.45), 6, 7, 8, 10 first part (1,000 to 2,500 cycles), 11, 12. Missed: 4 second part (the pointer is GAS + 0x180, not the leaf block at + 0x200: 0.25), 5 (0x310 high zero: no leaf address in a candidate answer, I gave 0.55), 9 as written (differences are process addresses; offsets identical), 10 second part (a candidate session is not longer than a miss session: 0.8 wrong).
R2 (8): held 1 (all 9 cases, also 64: 0.5), 4 (order equal), 5 (decoded primitive equals OptiX's), 7, 8. Mixed: 2 (sessions = K up to K = 8, K + 1 at 64, my 0.2 alternative), 3 (pending after every candidate but the last holds up to K = 8, not at 64). Missed: 6 (later sessions cost 917 to 1,244, not 1,100 to 1,800, and the finishing session 3,412; my alternative below 900 did not occur either).

## What this does and does not show
Shown: a kernel whose TTU group is emitted by the encoder returns the miss form as before and delivers custom primitive candidates, identified exactly as OptiX's intersection calls, for nested boxes with several tree levels (64 boxes), with the next query built from the returned pending entries only. Not shown: anything with instances, a radius test inside the loop (R3), scenes where lanes differ (all lanes ran the same ray), more than one warp, scenes beyond 64 boxes, time against OptiX. The cost per candidate (about 920 to 1,240 cycles per session) is of the size OptiX pays per candidate (F49: 850 to 900 more than a triangle), so no per candidate gain is indicated.
