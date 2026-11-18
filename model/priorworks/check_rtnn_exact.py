"""RTNN (radius mode, K 50) result lists against the true 50 nearest (F227 addendum, 2026-11-17). CPU only. Dumps from notes/q38/run_rtnn_chk.sh (rt_chk observer build).
A query is exactly correct if the 50 returned points have the same sorted distances as the true 50 nearest (set equality up to equal distance ties, tolerance 1e-6 relative, float32 points widened to double) and no slot is empty.
A: the query point itself counts as one of the 50 (distance 0; the points are the queries; radius mode returns it). B: the 50 nearest OTHER points (knn mode appears to leave the query out). Also reported: mean recall (returned points no farther than the true 50th, over 50): fraction of queries whose r ball holds at least 50 points (self included), and the fraction of queries with a full list.
Usage: python3 -I model/priorworks/check_rtnn_exact.py STAMP            (radius mode rows)
       python3 -I model/priorworks/check_rtnn_exact.py knn BUNNY_STAMP DRAGON_STAMP   (RTNN knn mode, r99 rows)"""
import sys, numpy as np
from scipy.spatial import cKDTree
KNN = sys.argv[1] == 'knn'; ts = sys.argv[1]; D = '/dev/shm/rtchk/'; K = 50; U = 0xFFFFFFFF
print('| row | queries | full lists | ball holds >= 50 | exact, query point counts as one (A) | exact, 50 nearest other points (B) | mean recall of A | mean returned |\n|---|---|---|---|---|---|---|---|')
rows = (('bunny', 0.0062, 'r99', sys.argv[2] if KNN else ts), ('dragon', 0.5875, 'r99', sys.argv[3] if KNN else ts)) if KNN else (('bunny', 0.0062, 'r99', ts), ('bunny', 0.0012273, 'r2', ts), ('dragon', 0.5875, 'r99', ts), ('dragon', 0.6133, 'r2', ts))
for nm, r, tag, ts in rows:
    pre = D + ('knn_' if KNN else '') + '%s_%s_%s_' % (nm, tag, ts)
    q = np.fromfile(pre + 'q_b0.bin', np.float32).reshape(-1, 3).astype(np.float64); p = np.fromfile(pre + 'p_b0.bin', np.float32).reshape(-1, 3).astype(np.float64)
    res = np.fromfile(pre + 'r_b0.bin', np.uint32).reshape(len(q), K); n = len(q); t = cKDTree(p)
    full = (res != U).all(1); cnt = (res != U).sum(1)
    exA = np.zeros(n, bool); exB = np.zeros(n, bool); ball = np.zeros(n, bool); rec = np.zeros(n)
    for s in range(0, n, 200000):
        sl = slice(s, min(n, s + 200000)); qq = q[sl]
        dA, iA = t.query(qq, k=K + 1, workers=-1)       # K+1 so B can drop self
        ball[sl] = dA[:, K - 1] <= r * (1 + 1e-6)         # K nearest (self included) inside r
        rr = res[sl]; valid = rr != U; idx = np.where(valid, rr, 0).astype(np.int64)
        dr = np.linalg.norm(p[idx] - qq[:, None, :], axis=2); dr[~valid] = np.inf; dr.sort(axis=1)
        dtrueA = dA[:, :K]
        exA[sl] = valid.all(1) & (np.abs(dr - dtrueA) <= 1e-6 * np.maximum(dtrueA, 1e-12)).all(1)
        dtrueB = dA[:, 1:K + 1]; exB[sl] = valid.all(1) & (np.abs(dr - dtrueB) <= 1e-6 * np.maximum(dtrueB, 1e-12)).all(1)
        rec[sl] = ((dr <= dtrueA[:, K - 1:K] * (1 + 1e-6)).sum(1)) / K
    print('| %s %s %s (r %g) | %d | %.4f | %.4f | %.4f | %.4f | %.3f | %.2f |' % ('knn mode' if KNN else 'radius mode', nm, tag, r, n, full.mean(), ball.mean(), exA.mean(), exB.mean(), rec.mean(), cnt.mean()), flush=True)
