# Non dual session in the layout of Cesar's SM120 file: predictions, committed before any build or run

2026-11-24, triton GPU 1. Approved by Jack (narrowly: the fuse 8 shape rules in check_seq and a --cesar8 builder flag). NOT BUILT YET.

Forms (one warp, single session build with the usual flags --clock --remap --compact-out --regs 104 --smfilter; base = the same build with the OptiX fuse 17 group and DUAL, the "dual form"):
- ces1: Cesar's store set as is: TTUOPEN without DUAL, TTUMACROFUSE 8, TTUST 0, 0x40, 0x50, 0x80, TTUGO, TTULD 0x300, 0x320, TTULD.CLOSE 0x3D0 (close bit 74). Every instruction is a verbatim copy of the OptiX group's, so registers and predicates are those of the checked 17 slot group. His TTULD offsets are placeholders (all 0); 0x300, 0x320 and the close 0x3D0 are my choice (the loads that deliver t, u, v, prim and the hit flag).
- ces2: the same plus the stores at 0xC0 and 0xD0 (the ray words the unit reads, F441): fuse 10, those two with their OptiX predicate @P0.
Neither has 0x100, 0x180, 0x190 (instance node, root).
Checks: check_seq ALL PASS (new rules C8 only for these two shapes), stage 0 (no TTU instruction) runs, then one stage 1 launch on a hit ray (tri1) and one on a miss ray scene (par:1024), then 30 launches each for the session cycles. Rules as before: before each run nvidia-smi and who, only with no other user's GPU process, one warp, timeout -k 2 30, one variant at a time, a hang that does not clear stops everything.

Predictions (from F439 to F442):
- ces1: ran without fault, no hit on the hit ray (hit flag 0, miss pattern or zeros), a short session (under 2,500 cycles on the miss scene, against about 100,500 for the dual form): probability 0.6. Fault: 0.2. Correct hit (t 1.0, u 0.25, v 0.5): 0.1. Hang that does not clear: 0.05 (count and close are exact). Other (a miss with the full session length): 0.05.
- ces2: the same as ces1 (no root, no instance node): no hit, short session 0.55; correct hit 0.15; fault 0.2; hang 0.05; other 0.05.
- Cycles of the hit ray in the dual form: about 1,586 (known, F441).
What DUAL changes: not predicted beyond F440 (the non dual open ends the fuse 17 group's session at once with no hit); the outcome of these two forms says whether a non dual open traverses with a different store set (hit) or not (no hit in both).
