"""NNsphere exactness check. check_nn.py DATA K R QN SAMPLE IDS_A N_A CAND_A [IDS_B N_B CAND_B]  (files from the dump option). Reference: kd tree on float64 copies of the float32 points.
For SAMPLE random queries among the first QN: (1) list size = min(K, points within r), allowing points within 1e-5 r of the boundary to go either way; (2) distances of the list equal the K smallest distances (rel tol 1e-5); (3) ids unique and valid.
Also: ball count (points within r, self included) mean over the sample; mean candidates from the cand file; if a second dump is given, equality of the two lists over all QN queries (ids identical, or distance sequence equal when only ties differ)."""
import sys, numpy as np
from scipy.spatial import cKDTree
data, K, R, QN, S = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
P = np.fromfile('/dev/shm/nn/data/%s.bin' % data, np.float32).reshape(-1, 3).astype(np.float64); t = cKDTree(P)
import os
def load(ids, n, c):
    a, nn, cc = np.fromfile(ids, np.int32).reshape(QN, K), np.fromfile(n, np.uint32), np.fromfile(c, np.uint32)
    pf = ids.replace('_ids.bin', '_perm.bin')
    if os.path.exists(pf):   # sorted run: row i is the query of original point perm[i], ids are sorted positions; bring everything to the original order
        perm = np.fromfile(pf, np.uint32).astype(np.int64); a2 = np.where(a >= 0, perm[np.clip(a, 0, None)], -1).astype(np.int32)
        A2 = np.empty_like(a2); N2 = np.empty_like(nn); C2 = np.empty_like(cc); A2[perm[:QN]] = a2; N2[perm[:QN]] = nn; C2[perm[:QN]] = cc; return A2, N2, C2
    return a, nn, cc
A = load(*sys.argv[6:9]); rng = np.random.default_rng(21); qs = rng.choice(QN, min(S, QN), replace=False)
ball = np.array([len(x) for x in t.query_ball_point(P[qs], R * (1 + 1e-5), workers=-1)]); ball_lo = np.array([len(x) for x in t.query_ball_point(P[qs], R * (1 - 1e-5), workers=-1)])
dk, ik = t.query(P[qs], k=K, workers=-1)
bad_size = bad_dist = bad_dup = bad_range = 0; examples = []
for j, q in enumerate(qs):
    n = int(A[1][q]); ids = A[0][q, :n]
    if (ids < 0).any() or (ids >= len(P)).any() or len(set(ids.tolist())) != n: bad_dup += 1; examples.append(('invalid_or_dup', int(q))); continue
    if not (min(K, ball_lo[j]) <= n <= min(K, ball[j])): bad_size += 1; examples.append(('size', int(q), n, int(ball_lo[j]), int(ball[j])))
    d = np.sort(np.linalg.norm(P[ids] - P[q], axis=1)) if n else np.zeros(0)
    if n and d.max() > R * (1 + 1e-5): bad_range += 1
    m = min(n, K, ball_lo[j]); 
    if m and not np.allclose(d[:m], dk[j, :m], rtol=1e-5, atol=1e-9): bad_dist += 1; examples.append(('dist', int(q)))
full = (ball_lo >= K)
print('data', data, 'K', K, 'r', R, 'Q', QN, 'sample', len(qs))
print('ball count over the sample: mean %.3f, median %d, max %d; queries with >= K inside: %.2f%%' % (ball.mean(), np.median(ball), ball.max(), 100 * full.mean()))
print('mean candidates (all %d queries) %.3f ; candidates / mean ball %.3f' % (QN, A[2].mean(), A[2].mean() / ball.mean()))
print('wrong size %d, wrong distances %d, invalid or duplicate %d, beyond r %d  (of %d sampled queries)' % (bad_size, bad_dist, bad_dup, bad_range, len(qs)), examples[:5])
if len(sys.argv) > 9:
    B = load(*sys.argv[9:12]); same = (A[0] == B[0]).all(axis=1) & (A[1] == B[1]); neq = np.where(~same)[0]
    # ids differ: allowed only if the distance sequences agree (ties)
    tie_ok = 0
    for q in neq[:20000]:
        na, nb = int(A[1][q]), int(B[1][q])
        if na != nb: continue
        da = np.sort(np.linalg.norm(P[A[0][q, :na]] - P[q], axis=1)); db = np.sort(np.linalg.norm(P[B[0][q, :nb]] - P[q], axis=1))
        if np.allclose(da, db, rtol=1e-6, atol=1e-12): tie_ok += 1
    print('arm A vs arm B over %d queries: identical %d, differ %d (of the first %d differing, equal distance sequences %d), list sizes differ %d, mean candidates A %.3f B %.3f ratio A/B %.3f' % (QN, same.sum(), len(neq), min(len(neq), 20000), tie_ok, (A[1] != B[1]).sum(), A[2].mean(), B[2].mean(), A[2].mean() / B[2].mean()))
