# TTUCCTL variants: prepared, NOT run (plan in model/predictions_cctl_profiler.md, part B)

2026-11-24, triton. Stage 1 of every variant waits for Jack's approval.

Spec result: the EoSS spec for the TTUCCTL class has one mode, invalidate all (one bit; the table of allowed values holds only that one). The other fields are the predicate, the wait and release scoreboards, the performance monitor predicate and the scheduling bits. SM89 and SM120 have the same fields. No write back, no per level, no partial invalidate is in the spec, so any other variant is a probe of an undefined encoding.

Built (CPU only), each differing from the checked baseline stage 1 build in the bytes of the one TTUCCTL and nowhere else (checked by comparing the files): V1 modifier bit cleared; V2 predicated off (control, never executes); V3 read scoreboard released; V4 write scoreboard set (the paper reports a fault: built for completeness, never to be run). Stage 0 of all variants is the baseline stage 0 binary (the slot is a NOP there) and ran, 30 launches, no fault.
check_seq: the baseline passes. Each variant fails exactly the rules that pin the baseline (the build hash, the byte equality with the encoder output, and for V1 and V2 the unpredicated and modifier rule) and passes every structural rule (decode, one session shape, fuse count, close bit, offsets). The variants have no check marker, so run.py refuses them. Adding rules for the variants would change check_seq.py: Jack decides.
Run order if approved: V2, V3, V1, one at a time, stage 0, one stage 1 launch at 1,024 triangles, then 3 x 30 launches at 1,024 and 2,048; stop at the first fault. V4 never. Builds are in /dev/shm/cctl1 (lost on reboot; rebuild with ttu/bench/Cctl_1/cctl1.py prep then variants; delete with clean after use).
