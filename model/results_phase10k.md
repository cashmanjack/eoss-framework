# Results Phase 10k (tethys GPU 1, TTU_GPU=1, 2026-11-19): F264 second attempt and the F265 zero wait arm. Predictions: model/predictions_phase10k.md (057af79, committed before any run). Runners ttu/bench/Phase10k/f264_run.py and wait0_run.py. Raw notes/p10k (private).

## F264
Cause (code read first): the same record build is made with the compact record option (16 byte record per ray); the host expands compact records only when given the same option, and my phase 10j runner did not pass it. The store code was right. (Build hashes differ from the 10i note because /dev/shm/t9/setup3 was rebuilt since; check_seq ALL PASS for both stages.)
| Step | Result | Prediction |
|---|---|---|
| Stage 0, 1 warp and 13 warps | 32 of 32 and 416 of 416 stored, four stored words identical in all lanes, rounds left 0 | stored 85 percent: held; "hit word 0": MISSED (R3 holds an input flags word in stage 0, 0x024840ff; one earlier launch showed 0x024840fe, unexplained) |
| Stage 1, 1 block, 1 warp | 32 of 32 stored, 32 hits, t, primitive equal to OptiX, 0 wrong, 0.95 ms, no fault | held (75 percent) |
| G 1, 13 warps | 416 of 416 equal, 4.32 ms, 0.038 rays per SM per cycle | at least 0.06: MISSED (void by my rule) |
| G 64 | 26,624 of 26,624 equal, 7.62 ms, 0.022 per SM per cycle, 36 times one SM in whole GPU rate | at least 60 times: MISSED |
| G 128 | 53,248 of 53,248 equal, 14.75 ms, 0.011, whole GPU 3,609 rays/us | 128 over 64 1.3 to 1.6: MISSED (1.03) |
Rays = lanes x 1000 rounds; clock 2,520 MHz (sampler, 40 samples, one 2,730). Overlap not scored: the build has no timing probes (the host json gt fields are not valid for it). The whole GPU rate stops growing at 64 blocks; whether that is the unit of a pair or the memory system is not separated.

## F265 zero wait arm
Builds single, twice, sib (check_seq ALL PASS), stage 0 and stage 1 check for all arms, 3 x 30 launches, no fault. Median device cycles (saving): 
| N | single | twice | w0 | w70 |
|---|---|---|---|---|
| 1,024 | 100,368 | 38,321 (0.62) | 41,998 (0.58) | 42,430 (0.58) |
| 2,048 | 172,487 | 79,591 (0.54) | 94,143 (0.45) | 92,886 (0.46) |
Loss against twice: w0 0.04 / 0.09, w70 0.04 / 0.08. Prediction (block causes it: w0 within 0.03 of w70) HELD. Clock sampler missed the short launches (210 MHz idle readings only); the measure is device cycles.

Mistakes: runner without the host compact flag (the F264 cause); stage 0 check demanded hit word 0 (my prediction wrong) and compared host expansion leftover columns (fixed to the four stored words before the passing run, raw files of the failed checks kept); first run of the runner used the venv python without numpy.
