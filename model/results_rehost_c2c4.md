# R3 C2 to C4 (2026-11-24, triton, GPU 1): exact after fixing the result record layout
Prediction first: model/predictions_rehost_c2.md (065d1f4): C2 exact 0.9, sessions equal answers 0.9, no fault or hang 0.95. Raw (private): notes/rehost/r5c_*, r500_s0*, GAS per scene gas_r5c_*.bin.
## Cause of the C2 mismatch (found at stage 0, before any TTU run)
Stage 0 was extended: answer 1 with a run count from {1, 2, 3, 8, 17, 31, 63} in round 1, answer 2 (another run, count from the same set) in the 4 further rounds (the NOP group leaves the registers as they are, answer 2 arrives through the pending entry swap and the reloads), 512 lanes, expected list compared in order. Before the fix it reproduced the defect (5 to 29 bad lanes per run, always from list position 16): the neighbour ids are written to slots 16 to 79 of the lane's result record, and the final stores of the last answer's registers went to the slot equal to the register number (32, 33, 44, 45, 56 to 61, 70, 71), overwriting list positions 16, 17, 28, 29, 40 to 45, 54, 55 (R32 and R33 of a miss form are 0xffffffff, which is the id seen in C2). C1 had at most 16 neighbours per query, so it was untouched. After moving the last answer words to slots 80 to 92, 6 stage 0 runs are exact. NOT a defect of the count loop, the record table, the unit or OptiX; the F454 reading that the count loop still had a defect is withdrawn.
## Stage 1, one warp, 32 queries, 3 launches each, no fault, no hang
| config | points, r, seed | differ from OptiX / CPU | answers per query (mean, max) | OptiX candidates per query | neighbours per query | loop cycles per lane (mean) |
|---|---|---|---|---|---|---|
| C2 | 4,096, 0.12, 8 | 0 / 0 of 32 | 39.9, 56 | 48.8 | 24.3 | 147,444 |
| C3 | 65,536, 0.03, 9 | 0 / 0 of 32 | 13.3, 20 | | 6.8 | 60,820 |
| C4 | 1,024, 0.2, 10 | 0 / 0 of 32 | 43.8, 67 | | 26.5 | 142,190 |
With C1 (0 of 32, earlier) all four configurations of R3 now match exactly. The session windows are 32 to 45 percent of the loop cycles (47,739 of 147,444 in C2; the rest is the inline test, reloads and moves in one warp: a breakdown for R5 only after R4). Lane efficiency 0.65 to 0.71.
Predictions: all held (exact 0.9, sessions 0.9, no fault or hang 0.95).
## Not done
R4 (many warps), R5 (timing against OptiX and RTNN, breakdown), BITTU on the rehosted kernel. Not started by order.
