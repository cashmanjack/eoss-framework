# Queue 15: hand off term for custom primitives, form fixed BEFORE any comparison with launch times (2026-11-16, 4070 Ti)
Source: Finding 49 (BVH_3): every custom primitive candidate costs one SM round trip of 850 to 900 cycles (triton, measured with an always false intersection program). Not refitted here: H = 875 (midpoint).
Term (SM cycles per SM): T_hand = H x N_calls / (60 x W)
- N_calls = warp calls of the intersection program. Two versions: predicted N = candidates / (32 x lane efficiency) from the replay (needs no run); measured N = probe count.
- W = warps in flight (ncu sm__warps_active.avg.per_cycle_active), the concurrency that hides the round trip.
- Variant B (per lane): N = candidates (lane calls), same H, W. Reported next to the primary.
Combination (primary): T_cb = T_floor + T_hand. The round trip is outside the program's own instructions (probe window), so it is additive to the callback's own issue and memory cost of Finding 131. Sensitivity: max(T_floor, T_hand).
Everything else as in model/q14/FORM.md: RTNN and kNN c0 refit (one parameter) on RTNN as before, same held out splits; Arkade callback term alone. Error = log ratio predicted over measured, rms.
Tests: RTNN (12 configurations), growing radius kNN rounds 1 to 4 (candidates per query = hardware candidates of the round), Arkade (3 configurations) and the Arkade build with an empty intersection program (23.1 ms in Finding 136; N measured on a fresh run).
Prediction written down now: RTNN gets a little better or unchanged, because c0 already absorbs a per query constant. kNN: no change of consequence (its error is the any hit insertion). Arkade: stays 50 to 100 times too low, because the hand off at 875 cycles x 1,750 calls per SM / 15 warps is about 0.1 ms against 22 to 30 ms measured. Fails if Arkade error falls under 0.5 or RTNN error rises by more than 0.1.
