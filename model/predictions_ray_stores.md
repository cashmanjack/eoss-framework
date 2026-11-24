# What the stores at 0xC0 and 0xD0 carry: predictions, committed before any run

2026-11-24, triton GPU 1. Approved by Jack. NOT RUN YET.

Question. F439: predicating off the TTUST at 0xC0 loses the hit. notes/protocol.md called 0xC0 and 0xD0 "the same values as 0x40 and 0x50". Which words of 0xC0 and 0xD0 does the unit use, and for what?
Method. One word at a time (not predicate off): in the single session build (one session, the timed one) the last NOP before TTUOPEN is replaced by a MOV32I that sets ONE source register of one store to a constant, so one 32 bit word of the stored 16 bytes changes. Store words (the register pairs of our remapped build, read from the TTUST fields, Rc low pair then Rb high pair): word 0 = Rc, 1 = Rc+1, 2 = Rb, 3 = Rb+1. Hypothesis from notes/protocol.md for the contents: 0xC0 = origin x, y, z, tmin; 0xD0 = direction x, y, z, tmax; 0x40 and 0x50 the same for the ray. Constants: words 0 and 1 of an origin or direction store 0.1; origin z -2.0; tmin 1.5; direction z 2.0; tmax 0.5. 16 variants: 8 on 0xC0 and 0xD0, 8 on 0x40 and 0x50 as controls (same constants).
Rays. A hit ray (tri1: origin (0, 0, -1), direction (0, 0, 1), one triangle, base result hit, t = 1, u = 0.25, v = 0.5) and the miss ray scene par:1024 (session length). Mandatory for every variant from now on (Jack): results compared on both a hit ray and a miss ray, one launch each against the unchanged build, and the session cycles (30 launches).
Rules as before: before each run nvidia-smi and who, only with no other user's GPU process, one warp, timeout -k 2 30, one variant at a time, outcome classes (ran, fault, timeout or hang), the 22 result registers read as hit flag, t, u, v, prim. A hang that does not clear stops everything. The variants are accepted by check_seq by hash after the base build passes and only the one MOV32I differs (diff, decode check).

Hypothesis H1: 0xC0 and 0xD0 are the ray in the space of the geometry (object space; equal to the world ray here because the instance transform is the identity) and the unit uses them for the triangle test; 0x40 and 0x50 are the world space ray used at the instance level (box test of the instance node).
Predictions (probability that the whole row holds 0.5):
| stores | word | constant | hit ray result predicted |
|---|---|---|---|
| 0xC0 | 0, 1 (origin x, y) | 0.1 | hit stays, u and v change, t = 1 |
| 0xC0 | 2 (origin z) | -2.0 | hit stays, t = 2.0 |
| 0xC0 | 3 (tmin) | 1.5 | hit lost |
| 0xD0 | 0, 1 (direction x, y) | 0.1 | hit stays, u and v change |
| 0xD0 | 2 (direction z) | 2.0 | hit stays, t = 0.5 |
| 0xD0 | 3 (tmax) | 0.5 | hit lost |
| 0x40, 0x50 | all words except tmin and tmax | as above | no change to hit, t, u, v |
| 0x40 word 3 (tmin 1.5), 0x50 word 3 (tmax 0.5) | | | hit lost (the instance box test fails) |
Session length on the miss scene: unchanged within 3 percent except where the hit is lost on the hit ray (then it may change by tens of percent, as in F439).
If H1 fails the report says what the data show instead; no variant is added to chase it.
No fault and no hang expected (probability 0.95 each variant).

## Part B (non dual form after Cesar's SM120 file): prepare only until check_seq can validate it
check_seq has no rule for a fuse 8 group of 4 TTUST, TTUGO, 2 TTULD and a closing TTULD (it knows the fuse 17 GAS group, the dualgo group and the fuse 15 IAS group). Per Jack's instruction I stop and show before running. Nothing of part B is run.
