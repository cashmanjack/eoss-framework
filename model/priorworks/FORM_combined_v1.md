# Model v1: the combined form (frozen 2026-11-17, triton session, committed before it is scored)

T(launch) = max( T_callback, T_chain ), per launch (TrueKNN: per round, then summed).

- T_callback = the throughput form with the MEASURED callback cost of that program and input (F216 (b), unchanged: RTNN stub build difference, TrueKNN probe warp cycles with the F160 constants 1.485 and 12.8, Arkade T_ref x F_d / F_ref).
- T_chain = the slowest warp chain (F219, unchanged): largest per lane candidate count x L / 2.5e9, L = 575 (RTNN), 5,800 (TrueKNN), 21,000 (Arkade double; fixed 0.09 x).

No constant is new or refitted. Rows: the 14 real data rows (RTNN 4, TrueKNN 4, Arkade 6), same measured values as F216 (probe run for Arkade (b), as in results_priorworks_diag.md).
Honest status: NOT blind. Both parts were formed after the F207 measurements and the callback costs come from runs on the same inputs; the score below is a diagnosis of this one form, not a prediction. No more form changes after this commit (recorded in notes/performance_model.md).
Success rule as before: within a factor 2 (ratio 0.5 to 2); also median absolute log error per program and over all 14 rows.
