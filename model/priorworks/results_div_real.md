# Lane statistics of the real data rows against the misses of model v1 (2026-11-18, triton GPU 1, OptiX only, report only)

Predictions: predictions_phase11.md part C. Probe: Div_1 lane histogram (F231, full probe in Arkade and TrueKNN, one warp in eight for RTNN), tethys built binaries run on triton. Script div_real.py. Raw (private): notes/div1d. Rows are the 14 rows of results_model_v1.md; the Arkade original rows reuse the fixed build's lane statistics. Arkade runs 10,000 queries = 313 warps with callbacks only (a small sample).

```
rtnn_bunny_r1    files  1 warps    1416 callbacks 2.253e+03 mean lanes 21.24 (eff 0.664) time weighted lanes 28.23  callbacks <=8: 0.365  warp time <=8: 0.095  <=16: 0.096
rtnn_bunny_r2    files  1 warps    1416 callbacks 4.746e+04 mean lanes 14.17 (eff 0.443) time weighted lanes 18.07  callbacks <=8: 0.438  warp time <=8: 0.302  <=16: 0.417
rtnn_dragon_r1   files  1 warps   14100 callbacks 3.606e+05 mean lanes 14.30 (eff 0.447) time weighted lanes 17.15  callbacks <=8: 0.423  warp time <=8: 0.323  <=16: 0.449
rtnn_dragon_r2   files  1 warps   14100 callbacks 2.431e+06 mean lanes 15.16 (eff 0.474) time weighted lanes 17.59  callbacks <=8: 0.398  warp time <=8: 0.309  <=16: 0.478
tk_road_k5       files 13 warps  106889 callbacks 1.303e+06 mean lanes  6.87 (eff 0.215) time weighted lanes 14.92  callbacks <=8: 0.750  warp time <=8: 0.439  <=16: 0.567
tk_road_k659     files 13 warps  154158 callbacks 6.666e+07 mean lanes 12.80 (eff 0.400) time weighted lanes 19.95  callbacks <=8: 0.486  warp time <=8: 0.213  <=16: 0.392
tk_porto_k5      files 20 warps  163944 callbacks 5.143e+06 mean lanes  3.48 (eff 0.109) time weighted lanes  7.72  callbacks <=8: 0.896  warp time <=8: 0.688  <=16: 0.819
tk_uni_k5        files  5 warps  125062 callbacks 1.878e+06 mean lanes 14.07 (eff 0.440) time weighted lanes 19.49  callbacks <=8: 0.413  warp time <=8: 0.241  <=16: 0.342
ark_gowalla      files  1 warps     313 callbacks 2.653e+07 mean lanes  4.71 (eff 0.147) time weighted lanes  5.18  callbacks <=8: 0.843  warp time <=8: 0.816  <=16: 0.946
ark_glove3d      files  1 warps     313 callbacks 1.420e+08 mean lanes  7.97 (eff 0.249) time weighted lanes  8.90  callbacks <=8: 0.648  warp time <=8: 0.600  <=16: 0.818
ark_manuscript   files  1 warps     313 callbacks 1.227e+05 mean lanes 12.87 (eff 0.402) time weighted lanes 16.38  callbacks <=8: 0.463  warp time <=8: 0.346  <=16: 0.570

rows used: 14 (Arkade counted twice: fixed and original, same lane statistics)
mean_lanes_per_callback    Spearman with log(measured/predicted) over all rows: +0.18 (p 0.54)  within programs: rtnn -0.20, tk 0.80, ark 0.72
mean_lanes_time_weighted   Spearman with log(measured/predicted) over all rows: -0.21 (p 0.48)  within programs: rtnn -0.80, tk 0.40, ark 0.72
share_callbacks_le8        Spearman with log(measured/predicted) over all rows: -0.15 (p 0.61)  within programs: rtnn 0.20, tk -0.80, ark -0.72
warp_time_le8_lanes        Spearman with log(measured/predicted) over all rows: +0.21 (p 0.48)  within programs: rtnn 0.80, tk -0.40, ark -0.72

per program mean of the lane measure, and mean log miss:
rtnn  mean lanes 16.22  warp time <=8 0.257  mean ratio v1 2.07
tk    mean lanes 9.30  warp time <=8 0.395  mean ratio v1 0.71
ark   mean lanes 8.51  warp time <=8 0.587  mean ratio v1 2.80
```
