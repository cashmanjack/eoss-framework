# Predictions: rehosting R1 (first stage 1 run of an encoder emitted group, custom primitive scene), committed before any stage 1 run

2026-11-24, triton. Written after: R0 (encoder emitter byte compared offline with the checked group, check_seq rules E1 to E5, 24 old build configurations x 2 stages unchanged, host aabb mode, stage 0 runs with the OptiX reference). No stage 1 run of the new kind has happened.

## What runs
Build: `build_patch.py --stage 1 --emit --clock` on the aabb capable host (group and ISETP from ttu/rehost/emit_group.py, no dump read), check_seq ALL PASS first. One warp, one block, GPU 1, `timeout -k 2 30`, `--repeat 3`. Scene box1: one custom primitive, box [-0.5, 0.5]^3, one leaf slot. Rays, in this order (stop at the first fault, hang or wrong result):
1. `outside` (origin (3,3,0), direction +z, tmax 1e-16): in no box. OptiX: 0 intersection calls.
2. `miss` (origin (2,2,-1), +z, tmax 1e16): crosses no box. OptiX: 0 calls.
3. `inside` (origin (0,0,0), +z, tmax 1e-16, the RTNN point query): OptiX: 1 call, primitive 0.
4. `hit` (origin (0,0,-1), +z, tmax 1e16): crosses the box. OptiX: 1 call, primitive 0.
OptiX reference seen at stage 0 (raw notes/rehost/r1_s0a_box1_*): calls [1,0] on hit and inside, [0] on miss and outside; GAS 896 bytes, root offset 0x300, c[3][0x34] and the handle type as for triangles.

## Pass rule (fixed now)
R1 matches OptiX exactly iff all hold: (A) on outside and miss the answer is the miss form already known for triangles (hit flag word 0, the four words of 0x300 all ones) and no pending entry word is nonzero; (B) on inside and hit the answer is NOT the miss form, it names a candidate whose decoded primitive is 0 where a decode exists (leaf address inside the GAS buffer, record pointer inside it), and the answers of inside and hit agree on every word except at most the distance related ones; (C) one session returns one candidate (no second candidate word set), the same answer in all 32 lanes and in all 3 launches; (D) no fault, no hang, ttu_one status "no error". If (A) holds and (B) fails or is ambiguous, I stop and report; R2 only after a full pass.

## Predictions (probability, then what would refute it)
1. Outside and miss give the miss form (A): 0.85. Session cycles 900 to 2,500: 0.7. Refuted by any nonzero word that is not all ones in 0x300, a nonzero hit flag, or a cycle count outside 600 to 6,000.
2. Inside and hit return something that is not the miss form (a candidate is delivered at all): 0.85. Refuted by the miss form on either ray (then the box is invisible to our group: store words or flags for custom primitives differ).
3. The first answer word of 0x300 (R33) has bit 31 set (a tag): 0.75; its top seven bits equal 0x50: 0.55; 0x58: 0.15; something else: 0.3 (these three overlap with "bit 31 clear": 0.25).
4. R44:R45 read as one 64 bit value times 8 lies inside [GAS, GAS + 896): 0.45; equals GAS + 0x200 (the leaf block): 0.25.
5. 0x310 high (R30:R31) holds the leaf block address, GAS + 0x200 (low 48 bits): 0.55.
6. The hit flag word (R3) is nonzero on a candidate: 0.45 (zero: 0.45, other: 0.1).
7. Pending entries (0x3C0, 0x3D0: R56 to R61, R70, R71) are all zero for this one leaf scene: 0.8.
8. The instance word (R34:R35 after the loads) equals its input: 0.8.
9. inside and hit give identical answers apart from nothing (the answer carries no ray dependent word for a candidate): 0.5; differ only in a distance related word: 0.35.
10. Candidate session cycles (clock window, --clock) 1,000 to 2,500: 0.65; longer than the miss session: 0.8.
11. No fault, no hang, status "no error": 0.97. A hang: 0.02 (the group is byte identical to the checked one, so this is the same risk as every earlier stage 1 run).
12. The emit build behaves identically to the copy build of the same group on the same scene (cycles within 5 percent, same words): 0.9. I do not run the copy build on a custom scene (it is the same bytes: sha256 of the two group slices are equal, shown in the log).

## What I will report either way
The four answers word by word, the decode attempt, cycles per ray, the pass rule outcome, and whether the answer of a candidate resembles any word pattern seen in the OptiX code (tag, record pointer): as findings, labelled inferred where the decode rests on the code reading.
