# RTX 4090 (tethys GPU 1), session 9 predictions, 2026-11-19. Written BEFORE any run. OptiX only (no TTU code of ours).

Test: the 5090 session's stack scene (F429) on the 4090, to give the like for like Ada triangle slope. ttu/bench/Sphere_1 sources unchanged (stack kind present), built for sm_89 (PTX compute_89); runner ttu/bench/pc14/run_sphere_pc14.py `stack` (N = 1, 4, 16, 64, 256, 1024, 4096; one warp and saturation 4,194,304 rays, 12 warp launches, 10 reps). The runner has pc14 only lines (GPU 0, notes/pc14_gpucheck.sh, which is not in this tree); I run a copy with only those lines changed (GPU 1, idle check by nvidia-smi), the arm code is identical. NSM = 128 for the cycles per ray per SM scaling.
5090 values to compare (F429): one warp slope 81.8, saturation slope 0.941 (R2 0.999), intercept at N = 1 14.52, saturation / one warp ratio 87, saturation cycles per ray per SM 65.4, 263, 841, 3,877 at N = 64, 256, 1,024, 4,096.
Predictions (4090):
1. Saturation slope over N (least squares, all N): between 0.8 and 1.6: 70 percent; central 1.15. 5090 / 4090 ratio between 0.6 and 1.1 (the 5090 cheaper or equal): 70 percent; ratio below 0.55 (4090 over 1.7, a 5090 gain of 1.8 or more): 7 percent; ratio above 1.1 (5090 slower): 20 percent. Basis: F429 random rays and one triangle equal within 10 percent; the near miss tessellated slopes 1.3 to 1.6 times cheaper on the 5090.
2. One warp slope per triangle between 70 and 100: 65 percent (F405: second trace equal on both chips, so latency equal); the ratio 5090 / 4090 within 0.9 to 1.1: 55 percent.
3. Intercept at saturation N = 1 within 10 percent of 14.4 (F186): 75 percent; 4090 / 5090 at N = 1 between 0.9 and 1.1: 75 percent.
4. Saturation / one warp slope ratio between 60 and 110: 65 percent.
5. R2 of the saturation fit above 0.98 for N >= 64: 85 percent.
6. Cycles per ray per SM at N = 4,096 between 3,300 and 5,500: 65 percent (5090 3,877; 4,096 x 1.15 = 4,700 central).
7. Zero hits at every N (as on the 5090): 95 percent.
Reading rules: ratio = 5090 value / 4090 value for the same quantity; "the 5090 gains on triangles" only if the saturation slope ratio is below 0.7 AND R2 above 0.98 on both; the whitepaper's 2x is "reached" only if the ratio is at or below 0.55. One run per cell, no cell repeated after reading it except a failed run (reported). Mixed method row of F429 (b) is replaced by this same method comparison; F429 text not edited.
