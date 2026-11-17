# IAS mask run and opaque geometry run, results (2026-11-17, tethys GPU 1, driver 580.178.04; predictions model/predictions_ias_opaque.md committed first, approved by Jack)

Correction: "no IAS stage 1 has ever run" (F192 first text, results_mask_ias.md, STATUS, NEW.md, setup3 README) was wrong; F139 ran the same IAS group on triton. What differs here: only the host (instance visibility mask 0x0F, --imask 15) and the mask carrying store words (stores 0 and 0x80); same group, same builds, check_seq ALL PASS on all of them. First run of the IAS group on this GPU and driver.

## Run 1, IAS group, instance mask 0x0F, one warp, one launch each, ray +z at the triangle (hit ray)
Stage 0 (no TTU) clean first. Order 0F, FF, 01, F0, 10, 00; all exit 0, one block, no fault, GPU idle after.
| ray mask | result flag | u, v | distance word | pair R26:R27 |
|---|---|---|---|---|
| 0F | hit | -0.25, 0.5 | -1.0 (sign set) | non zero (address) |
| FF | hit | same | same | non zero |
| 01 | hit | same | same | non zero |
| F0 | MISS | all ones | all ones | 0 |
| 10 | MISS | all ones | all ones | 0 |
| 00 | MISS | all ones | all ones | 0 |
Predictions: (1) 0F, 01, FF hit as F139 / OptiX: held. (2) F0 and 10 miss (the unit ANDs against the child slot byte 6): held (p 0.6). (3) mask 00 misses in the IAS case: held (p 0.55); the GAS group with the identity instance hit with mask 0 (Flags_2). (4) no fault: held.
Reading: the unit itself does the ray mask AND instance mask test, against the instance's visibility byte in the IAS child slot; the mask words in stores 0 and 0x80 are the source. A zero ray mask is rejected by the unit in the IAS group, so OptiX's software rejection (F192) is redundant for IAS and is the only rejection for a single GAS (identity instance carries no mask test in the unit).

## Run 2, GAS group (Flags_2 sequence, words of the none variant), geometry built opaque (--gopaque, OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT), flags NONE; control (non opaque) in the same session
| arm | ray | result flag | u, v | distance word | pair R34:R35 |
|---|---|---|---|---|---|
| control | hit | 1 | -0.25, 0.5 | -1.0 | equals the input instance word |
| control | back | 1 | 0.25, 0.5 | -1.0 | equals the input |
| opaque | hit | 1 | -0.25, 0.5 | +1.0 (sign CLEAR) | 0 |
| opaque | back | 1 | 0.25, 0.5 | +1.0 (sign CLEAR) | 0 |
Prediction 2 (sign clear and pair zero, p 0.5) held; unchanged (p 0.4) missed. Reading: the sign clear and zero pair is the unit's "no any hit needed" marker for any cause: the ray flag (F190) or opaque geometry (the slot bit 7 clear). Still open: a force opaque instance flag, and a scene with an any hit program that does run.
