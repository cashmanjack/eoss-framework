# Flags_2 stage 1 results (2026-11-17, tethys GPU 1; predictions model/predictions_flags2.md; approved by Jack 2026-11-17)

Our own TTU sequence (Setup_3, the OptiX group verbatim, one open, macro fuse 17, one close) with the F183 computed store words of each variant, one warp, one launch, rays hit (+z ray, back face, front 0) and back (front face), each run: check_seq ALL PASS on that binary, run.py (idle GPU, timeout -k 2 30), marker deleted again. 23 runs, all exit 0, no fault, no hang, GPU idle after. Stopped at the first unexpected result (mask0 on the hit ray); the mask0 back ray was not run. Runner ttu/bench/Flags_2/run_stage1.py. OptiX comparison: Flags_1 (OptiX only, one big triangle at z = 0, non opaque geometry, no any hit program; flip=0 is the hit ray, flip=1 the back ray) with the same flags and masks, files notes/flags2/optix_* (private): terminate, disable and enforce any hit, cull opaque, masks 1 / 15 / 128: hit on both rays; cull back: hit ray misses, back ray hits; cull front the reverse; cull non opaque: both miss; mask 0: both miss. (The brief said F179 to F181; F180 is the flags result and F181 the masks result, F179 is Memory_2; the comparison here is a fresh OptiX run of the same flags on the same kind of scene.)

| variant | hit ray | back ray | OptiX (same flags) | note |
|---|---|---|---|---|
| none (control) | hit, t 1, u -0.25, v 0.5 | hit, u 0.25, v 0.5 | same | as Setup_3 |
| TERMINATE | same as control | same | same | |
| ENFORCE_ANYHIT | same | same | same | |
| DISABLE_ANYHIT | hit, t, u, v same but the stored t word has its sign bit CLEAR (+1.0, control -1.0) and the 64 bit word pair R34:R35 is 0 (control: a non zero address like pair) | same | hit | NOT bit identical |
| CULL_OPAQUE | same as control | same | same | |
| CULL_NON_OPAQUE | miss | miss | miss | all result words 0xFFFFFFFF, hit word 0 |
| CULL_BACK | miss | hit (as control) | same | |
| CULL_FRONT | hit (as control) | miss | same | |
| mask128, mask15, mask1 | same as control | same | same | uninformative: the instance's own mask is 255 |
| mask0 | HIT (OptiX: miss) | not run | miss | UNEXPECTED, stopped |
(Results of a culled ray: u, v, R44, R45 words all ones, hit word 0, the same as an ordinary miss.)

## Against the predictions
1. Same as control, bit for bit (p 0.80, any hit flags 0.70): held for none, TERMINATE, ENFORCE, mask 1 / 15 / 128, CULL_OPAQUE; MISSED for DISABLE_ANYHIT (t word sign and R34:R35 differ).
2. CULL_BACK / CULL_FRONT select the face as predicted (p 0.75; the reverse assignment p 0.2): held, face assignment as OptiX.
3. CULL_NON_OPAQUE both miss, CULL_OPAQUE both hit (p 0.55): held, same as OptiX.
4. mask0 misses (p 0.8): MISSED. The hit ray hit with the ray mask words zero. So the ray mask words (top byte of the first word, bits 23:16 of the third) of the computed store words do not make the unit reject a ray against the identity instance (mask 255) on their own: either another word carries the mask for the unit (the word 0x100 side, the instance node) or OptiX rejects a zero mask before the unit (SM side, not decoded). Masks 1 / 15 / 128 cannot tell (they intersect 255).
5. Safety held (no fault in 23 runs).
Reading rule applied: mask0 unexpected, the computed word for the mask is NOT confirmed (the flag words are: cull back, cull front, cull opaque, cull non opaque, terminate (no visible effect on one triangle), enforce / disable any hit (a visible effect only for disable)). Not repaired by guessing; no further runs.
## What DISABLE_ANYHIT changes (observed)
The hit result (hit word, primitive, u, v) is the same, but R45 comes back +1.0 instead of -1.0 (the sign that every other hit has) and R34:R35 comes back 0 instead of a non zero pair (which differs from process to process in the other variants, looks like an address). Hypothesis (not tested): with any hit disabled the unit commits the hit as opaque, so no candidate record pointer is returned and the distance is returned without the negation flag. This is the only variant whose result words differ.
