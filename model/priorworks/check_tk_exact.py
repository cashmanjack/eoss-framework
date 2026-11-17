"""TrueKNN non exact queries (F218): distance ties, float precision, or genuinely wrong neighbours? (2026-11-17)
Neigh struct 24 bytes: int ind, float dist, int numNeighbors, pad, long long intersections. k 5, device buffer build (orig is bit identical, F218).
Reference: cKDTree on the float32 points widened to double, self excluded (program excludes only xID == primID, duplicates count).
A query is non exact (F218 rule) if any returned slot distance differs from the true sorted distance by > 1e-5 relative.
Classes (per non exact query, first that applies):
  A bad slot   : a returned index is invalid (<0, >= n, self, or repeated)
  B ordering   : the set of returned indices equals the true k set (only the order or stored dist differs)
  C tie        : returned set differs, but the sorted TRUE double distances of the returned points equal the true sorted distances (rel 1e-9): an equal distance swap
  D precision  : not C, and every sorted returned distance is within 1e-6 relative of the true one (float rounding of the program's compare)
  E wrong      : some sorted returned distance is above the true one by more than 1e-6 relative (a nearer true neighbour is missing)
Usage: python3 check_tk_exact.py STAMP"""
import sys, numpy as np
from scipy.spatial import cKDTree
D = '/dev/shm/pw/'; K = 5; ts = sys.argv[1]
dt = np.dtype([('ind', '<i4'), ('dist', '<f4'), ('nn', '<i4'), ('pad', '<i4'), ('isect', '<i8')])
def load(f, n): return np.array(open(D + 'data/' + f).read().replace(',', ' ').split(), dtype=np.float64).reshape(-1, 3)[:n]
out = []
for nm, f, n in (('road', '3droad.csv', 434874), ('uni1m', 'uni1m.csv', 1000000), ('porto1m', 'porto1m.csv', 1000000)):
    x = np.fromfile(D + 'nb2_%s_fixed_%s.bin' % (nm, ts), dt).reshape(n, K)
    p = load(f, n).astype(np.float32).astype(np.float64); t = cKDTree(p); d0, t0 = t.query(p, k=K + 1)
    selfm = (t0 == np.arange(n)[:, None]); selfm[~selfm.any(axis=1), -1] = True  # drop self by index (duplicates: self need not be column 0); if absent drop the last
    keep = ~selfm; ti = t0[keep].reshape(n, K); d = d0[keep].reshape(n, K)
    ind = x['ind'].astype(np.int64); sd = x['dist'].astype(np.float64)
    rel = np.abs(sd - d) / np.maximum(d, 1e-30); nonex = ((rel > 1e-5) & (np.abs(sd - d) > 1e-9)).any(axis=1)
    valid = (ind >= 0) & (ind < n) & (ind != np.arange(n)[:, None])
    srt = np.sort(ind, axis=1); rep = (srt[:, 1:] == srt[:, :-1]).any(axis=1)
    bad_slot = ~valid.all(axis=1) | rep
    real = np.linalg.norm(p[ind.clip(0, n - 1)] - p[:, None, :], axis=2); rmax = real.max(axis=1); dk = d[:, -1]
    same_set = np.array([set(a) == set(b) for a, b in zip(ind[nonex], ti[nonex])]) if nonex.any() else np.array([])
    idx = np.where(nonex)[0]; cls = np.empty(len(idx), dtype='<U1')
    sset = dict(zip(idx, same_set))
    for j, q in enumerate(idx):
        if bad_slot[q]: cls[j] = 'A'
        elif sset[q]: cls[j] = 'B'
        else:
            r = np.sort(real[q]); e_ = np.max((r - d[q]) / np.maximum(d[q], 1e-30))
            cls[j] = 'C' if np.all(np.abs(r - d[q]) <= 1e-9 * np.maximum(d[q], 1e-30)) else ('D' if e_ <= 1e-6 else 'E')
    # stored dist vs real distance of returned index (does the program report its own point's distance?)
    sr = np.abs(sd - real) / np.maximum(real, 1e-30); sdbad = int((sr > 1e-5).sum())
    cnt = {c: int((cls == c).sum()) for c in 'ABCDE'}
    # how wrong are the E queries: ratio of the farthest returned to the true k th, and how many true neighbours are missing
    e = idx[cls == 'E']; ratio = rmax[e] / np.maximum(dk[e], 1e-30)
    miss = np.array([K - len(set(ind[q]) & set(ti[q])) for q in e]) if len(e) else np.array([])
    # ties: for class C, how many points sit exactly at the k th distance (duplicates of distance)
    c_idx = idx[cls == 'C']
    # for E: rank (in true order) of the true nearest missing neighbour is <= k by definition; also check whether the returned far point lies in the query's box of half width dk (box stop rule)
    print('## %s n %d' % (nm, n))
    print('non exact queries %d (%.2f percent); slots with dist mismatch %d' % (len(idx), 100.0 * len(idx) / n, int(((rel > 1e-5) & (np.abs(sd - d) > 1e-9)).sum())))
    print('classes A bad slot %d, B ordering %d, C tie %d, D precision %d, E wrong %d' % tuple(cnt[c] for c in 'ABCDE'))
    print('stored dist differs from the real distance of the returned index (> 1e-5 rel): %d slots' % sdbad)
    if len(e):
        print('E: farthest returned / true k th: median %.4f, p90 %.4f, p99 %.4f, max %.3f; missing true neighbours per E query: mean %.2f, max %d' % (np.median(ratio), np.percentile(ratio, 90), np.percentile(ratio, 99), ratio.max(), miss.mean(), miss.max()))
        # Linf vs L2: is the returned far point inside the cube of half width dk (a box hit that is not a sphere hit)?
        lin = np.abs(p[ind[e]] - p[e][:, None, :]).max(axis=2).max(axis=1)
        print('E: Linf distance of the farthest returned / true k th L2: median %.3f (<= 1 means inside the k th ball cube; the stop rule counts box hits)' % np.median(lin / dk[e]))
        print('E: numNeighbors field values %s, true k th distance zero (duplicate points) in %d' % (np.unique(x['nn'][e][:, 0])[:5], int((dk[e] == 0).sum())))
    if len(c_idx):
        dup = int((dk[c_idx] == 0).sum()); print('C: queries with true k th distance 0 (duplicate points) %d of %d' % (dup, len(c_idx)))
    out.append((nm, n, len(idx), cnt))
