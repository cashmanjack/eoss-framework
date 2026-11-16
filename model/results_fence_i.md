# Arm (i) result: TTUCCTL.IVALL from the TTU warp's own SM (prediction: model/predictions_fence_i.md, 7c16a38)

2026-11-16, triton GPU 1. Same setup as F157 / F159 / F161. Between the two sessions the TTU warp on SM 0 executes exactly one TTUCCTL.IVALL (after its first CLOSE and flag1 store, outside any session, before its spin on flag2); a plain block on SM 1 only polls (mode 2, no fence). New option build_patch --ownccctl, check_seq B15 rules (stage 0 NOP in its place), both stages ALL PASS. 5 x 30 launches per cell, medians, rotating arm order, every run exit 0, exactly one block with a session per launch, no timeout, no fault. Stage 0 first (80 cycles, no TTU), then one stage 1 launch (100,590), then the matrix. Raw (private): notes/q22/fence_i_stage1_20261116T172843.json.

| N (KB) | single | twice | b1 (sibling polls) | i (own TTUCCTL) |
|---|---|---|---|---|
| 1,024 (68) | 100,556 | 38,239 | 39,342 | 100,344 |
| 2,048 (132) | 176,072 | 81,032 | 88,816 | 176,159 |

Saved fraction 68 / 132 KB: twice 0.62 / 0.54, b1 0.61 / 0.50, **i 0.00 / -0.00**. Prediction 0.00 / 0.00 (accept at most 0.10 / 0.08): HELD.
