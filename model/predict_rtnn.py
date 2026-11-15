"""RTNN prediction (Phase C, 2026-11-15): candidates per query and search time from the performance model, written BEFORE any RTNN run.

    predict_rtnn.py gen DIR            write the point clouds (uniform random in [0,100]^3, seeds fixed, one 'x,y,z' line per point)
    predict_rtnn.py predict DIR        candidate counts (exact, CPU) and the predicted 'time search compute' per configuration

RTNN (Zhu, PPoPP 2022; ~/rtnn) maps radius search to the RT core: one AABB (half size r) per point as a CUSTOM PRIMITIVE, one ray of zero length per query at the
query point. The RT core finds every AABB that contains the query; each one is a call of the intersection program on the SM (F49: one SM round trip per
candidate); the program tests the sphere and writes the neighbor, and reports an intersection (terminating the ray) after K = 50 sphere hits.
So per query: C_eff candidates = RT queries (F81: a ray is resumed while anything is pending) plus the first one.
Model (a statement, with its inputs):
  C_cube  points inside the cube |d| < r of the query (includes the query's own point; exact, CPU)
  S       points inside the sphere (includes self)
  C_eff   candidates until the ray ends: C_cube if S <= K, else K (C_cube + 1) / (S + 1)   (the K th hit among C_cube candidates in random order)
  T       = N_q (C_eff + 1) c / (SMeq f)
  c       cycles of one SM equivalent per candidate: 7.4 (RT unit service on a hot scene, F3 / Throughput_1) to 20 (SM issue of the program and the traversal loop
          around it, about 100 to 200 thread instructions per candidate at 128 lanes per cycle is only 1 to 2 cycles; the rest is L1 misses and queueing, set to 2.7x the minimum)
  SMeq    41.4 (30 TPCs x 1.38, F105: cheap queries) to 60 (the SM side is not cheap, both SMs of a TPC work)
  f       2.4 to 2.6 GHz (measured SM clock under load on this chip)
The point prediction is the geometric mean of the corners of the three ranges.
Configuration of every run: -p 0 -s 0 (no query partitioning, no GAS sort), default grid Morton sort of points and queries (-ps 1 -qs 1), K = 50, radius mode, queries = points.
"""
import sys, os, math, numpy as np

CFG = [('u1e5', 100000, [2.0, 4.0, 6.0]), ('u1e6', 1000000, [1.0, 2.0, 3.0])]
K = 50
C_RANGE, SM_RANGE, F_RANGE = (7.4, 20.0), (41.4, 60.0), (2.4e9, 2.6e9)


def gen(d):
    os.makedirs(d, exist_ok=True)
    for name, n, _ in CFG:
        rng = np.random.default_rng(20261115 + n)
        p = rng.uniform(0, 100, size=(n, 3)).astype(np.float32)
        np.savetxt(os.path.join(d, name + '.txt'), p, fmt='%.6f', delimiter=',')
        print('wrote', name, n)


def counts(pts, r):
    from scipy.spatial import cKDTree
    t = cKDTree(pts)
    ccube = t.query_ball_point(pts, r, p=np.inf, return_length=True).astype(np.float64)
    ssph = t.query_ball_point(pts, r, p=2, return_length=True).astype(np.float64)
    ceff = np.where(ssph <= K, ccube, K * (ccube + 1) / (ssph + 1))
    return ccube, ssph, ceff


def predict(d):
    out = []
    for name, n, rs in CFG:
        pts = np.loadtxt(os.path.join(d, name + '.txt'), delimiter=',', dtype=np.float32)
        for r in rs:
            cc, ss, ce = counts(pts, r)
            q = float((ce + 1).sum())
            lo = q * C_RANGE[0] / (SM_RANGE[1] * F_RANGE[1]) * 1e3
            hi = q * C_RANGE[1] / (SM_RANGE[0] * F_RANGE[0]) * 1e3
            pt = math.sqrt(lo * hi)
            out.append((name, n, r, cc.mean(), ss.mean(), ce.mean(), lo, pt, hi))
            print('%-5s N %7d r %.1f | C_cube %7.2f  sphere hits %7.2f  C_eff %7.2f | predicted search compute ms: %.3f .. %.3f  point %.3f' % (name, n, r, cc.mean(), ss.mean(), ce.mean(), lo, hi, pt))
    print('\nratios (point predictions): %s' % ', '.join('%s r%.0f / r%.0f: %.2f' % (a[0], b[2], a[2], (b[5] + 1) / (a[5] + 1)) for a, b in zip(out, out[1:]) if a[0] == b[0]))
    print('unsorted queries and points (-qs 0 -ps 0): u1e6 at least 1.5x slower than sorted, u1e5 at most 1.2x (BVH of 1e6 AABBs is about 70 to 80 MB: beyond the 48 MB L2; 1e5 is 7 MB)')


if __name__ == '__main__':
    gen(sys.argv[2]) if sys.argv[1] == 'gen' else predict(sys.argv[2])
