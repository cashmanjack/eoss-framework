# Arm (h) result: TTUCCTL.IVALL alone on the sibling SM (prediction: model/predictions_fence_h.md, 966b726)

2026-11-16, triton GPU 1. Same setup as F157 / F159. Between the two sessions of the TTU warp on SM 0, a second block on SM 1 (the TPC sibling) executes exactly one TTUCCTL.IVALL and no session instruction. 5 repetitions x 30 launches per cell, medians, rotating arm order, every run exit 0, exactly one block with a session per launch, no timeout, no fault. Stage 0 (same builds, NOP in place of the TTUCCTL) first, then one stage 1 launch, then the matrix. Raw data (private): notes/q22/fence_h_stage1_20261116T170734.json.

| N (KB) | single (cycles) | twice | b1 (sibling polls) | h (TTUCCTL on sibling) |
|---|---|---|---|---|
| 1,024 (68) | 100,545 | 38,231 | 39,385 | 100,370 |
| 2,048 (132) | 176,043 | 81,000 | 89,643 | 176,191 |

Saved fraction (1 minus second session over single session), 68 / 132 KB: twice 0.62 / 0.54, b1 0.61 / 0.49, **h 0.00 / -0.00**. Per repetition medians of h at 68 KB 100,572 / 100,564 / 100,356 / 100,345 / 100,343 (single 100,324 to 100,589); at 132 KB 176,059 to 176,255 (single 176,000 to 176,074).

Prediction was 0.27 / 0.24 (accept 0.15 to 0.40 / 0.13 to 0.36): MISSED. Outcome per the rule written before the run: "at most 0.12 / 0.10, nearly all removed: the invalidate by TTUCCTL is TPC wide".
