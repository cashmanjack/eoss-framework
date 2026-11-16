"""Arkade (sample01-knn, patched copy /dev/shm/ark, synthetic uniform 2D data, L2 build KN 16) predictions BEFORE measuring (CPU replay, no GPU timing).
    /usr/bin/python3 model/predict_arkade.py > model/predictions_arkade_4070ti.md"""
import numpy as np
print('# Arkade predictions (written before any measurement of the search launch), 2026-11-15, RTX 4070 Ti (io GPU 0)\n')
print('Data: synthetic uniform 2D points (the program reads x and y only, z = 0), 100,000 queries (the lines after the data points), one round, radius estimate given. Real Arkade datasets are not available here. Build: KN 16, NORM 2.')
print('Replay: model/replay/replay_aabb.cpp over the OptiX GAS of the cube boxes (dumped after the build, no search launch) and the real rays. Hardware candidates = 1.006 x primitives in entered leaf slots (Finding 125).')
print('Time: fixed part of Finding 127 (c0 110, cq 4.9 cycles per candidate per SM; per set fits c0 93 to 120, cq 4.2 to 5.9), miss term omitted; this program has no heap in global memory but a 16 entry scan per candidate in registers or local memory (any callback cost not in this prediction: see Finding 131).\n')
print('| config | data points | radius | queries | nodes / query | leaf slots / query | primitives in slots / query | hardware candidates / query | lane efficiency | predicted launch ms (c0 110, cq 4.9) |'); print('|---|---|---|---|---|---|---|---|---|---|')
for tag, n, rad in (('a1e5_r5', 100000, 0.005), ('a1e5_r11', 100000, 0.0112), ('a1e6_r16', 1000000, 0.0016)):
    r = np.fromfile('/dev/shm/ark/dump_%s/rep.bin' % tag, np.int32).reshape(-1, 8); N = len(r); c = r[:, 2] * 1.006
    w = c[: N // 32 * 32].reshape(-1, 32); eff = w.sum() / (32 * w.max(1).sum()); q = r[:, 2].mean() + 1
    print('| %s | %d | %g | %d | %.2f | %.2f | %.2f | %.2f | %.3f | %.3f |' % (tag, n, rad, N, r[:, 0].mean(), r[:, 1].mean(), r[:, 2].mean(), c.mean(), eff, (110 + 4.9 * q) * N / 1.5e11 * 1e3))
print('\nWhat can fail: candidates per query beyond 1 x 1.006; lane efficiency beyond 0.05; time beyond a factor 2 (the callback and miss terms are not in the prediction).')
