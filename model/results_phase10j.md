# Results Phase 10j (tethys GPU 1, TTU_GPU=1, 2026-11-19): F264 stage 1 and the wait sweep. Predictions: model/predictions_phase10i.md (test 3) and predictions_phase10j.md, both committed before any run.

## F264 stage 1 (same record loop build, approved by Jack 2026-11-19)
Idle GPU, clock sampled (50 to 200 ms). Stage 0 at one block passed (no TTU, 25.3 ms). Stage 1, one block one warp, 32 rays (the smallest launch): exit 0, no error, no fault, launch 0.95 ms. The check of the final records against the OptiX reference (hit, t, primitive, unstored marker) FAILED, twice (the second run after fixing my own reading code): 30 of 32 records unstored, t and primitive differ on the rest. The warp's registers (raw json) show hit and the expected t (1.0), so the answer looks right and the fault is in where the build stores the final record, or in my reading of it. Per the order (stop on the first wrong result) the 64 and 128 block runs were NOT done, so no rate is scored (predictions: one SM at least 0.06 rays per cycle, G 64 at least 60 times, G 128 over G 64 1.3 to 1.6: all unscored). Mistakes: my first runner passed --outer to the host (ray count error, exit 2, before any launch); my first check demanded results from stage 0; my check deleted the dumps before stopping. /dev/shm/t10k deleted as ordered (rebuild: build_patch --same-record, flags in F264 and the log).

## Wait sweep (F265)
Node_4090 builds single, twice, sib (check_seq ALL PASS), stage 0 for all arms, stage 1 check for all arms (one launch each), then 3 repetitions x 30 launches, clock 2,520 MHz, no fault. Median cycles and saving (1 - arm/single):
| N | single | twice | wait 10 us | wait 70 us | wait 300 us |
|---|---|---|---|---|---|
| 1,024 (68 KB) | 100,432 | 38,348 (0.62) | 42,809 (0.57) | 42,673 (0.58) | 43,310 (0.57) |
| 2,048 (132 KB) | 172,684 | 79,435 (0.54) | 93,410 (0.46) | 92,889 (0.46) | 92,636 (0.46) |
Prediction (loss grows with wait: 10 us at most 0.03; 70 us reproduces 0.08 and 0.15 within 0.04; 300 us at least 1.5 times the 70 us loss) MISSED: losses are 0.04 to 0.05 and 0.08 at every wait; 70 us gives 0.04 and 0.08 (F256 had 0.08 and 0.15: 0.07 off at 132 KB). The rival (wait length does not matter, within 0.03) held. Not tested: wait zero with the role block present (separates the block from the wait). Raw notes/p10j (private); /dev/shm/t9m deleted.
