# Variant (b): measured callback cost on each real dataset, repredict (form committed before the calibration runs, 2026-11-17, triton GPU 1)

One calibration run per program and dataset (3 runs for TrueKNN, which already exist in the F207 raw files because its build always has the clock64 probes). No refit of any constant of the F207 forms. The counts are the mean count forms of F207 (and the warp max forms of variant (a), both reported).

- TrueKNN: round time = callback warp cycles of that round (probe, sum over warps) x 1.485 / (142 x 12.8 warps in flight) / 2.5e9 s. Constants 1.485 and 12.8 are those of F160 (uniform 3D, k = 16, set A); they are not refitted. Measured callback cycles are GIVEN here (same runs), so this is a diagnosis, not a prediction (as F160).
- Arkade (probe build, CBK f0 cycles = warp cycles inside the callback): predicted launch time = T_ref x F_d / F_ref, with F the total callback warp cycles and ref = 100,000 uniform 2D points and 100,000 queries, r 0.005 (the F207 calibration set), same build variant (orig = double pow, fixed = float square). T_ref is measured in the same session with the event timer.
- RTNN (no probes): a stub build whose intersection program returns at once (same traversal, no distance check, no write). Callback cycles per candidate per SM cb = (T_full - T_stub) x 142 x 2.5e9 / (1.10 x cube candidates in total). Repredict: cq_d = 4.9 + (cb_d - cb_ref) with cb_ref from the uniform reference (100,000 uniform 3D points, r chosen for about 10 candidates), time = (110 + cq_d x (C_eff + 1)) N / (142 x 2.5e9).

Success rule: a row closes if within a factor 2 of the measured time. A program is "closed by (b)" if all its rows close. Reported next to variant (a) and to F207, with the median absolute log error per program.
