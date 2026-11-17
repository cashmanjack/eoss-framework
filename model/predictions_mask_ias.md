# Mask_1 (instance mask 0x0F against ray masks): predictions, written 2026-11-17 BEFORE any run (tethys GPU 1, OptiX only)

Code reading done first (dumped traversal code, base window and a mask 0 window from fnaddr-0x12000; private, notes/flagenc):
- The ray mask (R37 bits 15:8) is used in the traversal function in two ways only: copied into the store words (top byte of the first word, bits 23:16 of the third) and tested for non zero (three places, one per traversal variant). No instruction ANDs it with an instance mask in software.
- When the test says the mask byte is zero or the handle is zero, the CALL to the traversal body (the function that holds the TTU group) is skipped, with the status register left at its "no exception" value: the trace returns a miss without any TTU instruction. So OptiX rejects a zero ray mask in software before the unit (reading of the predicate polarity: the call only makes sense for a non zero mask and non zero handle).
- Consequence for Flags_2: mask 0 hit with our sequence because our sequence has no such software check; the unit itself did not reject a zero ray mask against the identity instance (single GAS pipeline).

Scene: one instance (visibility mask 0x0F, identity transform, z = 0) of the one big triangle GAS in an IAS, flags NONE, ray from z = -1 along +z at (0.1, 0.05) and the 262,144 ray set (all inside the triangle).
Ray masks: 0xF0, 0x01, 0x10, with controls 0x0F, 0xFF, 0x00.

Predictions (OptiX):
1. 0xF0 miss, 0x10 miss (no common bit with 0x0F): p = 0.97.
2. 0x01 hit, 0x0F hit, 0xFF hit: p = 0.97.
3. 0x00 miss: p = 0.97 (software rejection, seen on the GAS pipeline already).
4. Hit fraction on the big launch exactly 0 or 1 in every case (all rays inside the triangle): p = 0.95.
Reading rule: any deviation means the instance mask is not ANDed as documented (stop and report).

Hypothesis for our own sequence (not run here, written for the next step): the unit does the AND against byte 6 of the IAS child slot (F111), taking the ray mask from the store words (top byte of word 0). With the IAS group (BVH_4 / F91, 17 instructions) the mask words are different stores from the GAS group, so Flags_2's words do not carry over; the IAS group needs its own word extraction before any run.
