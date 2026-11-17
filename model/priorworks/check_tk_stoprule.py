"""TrueKNN non exact lists: does the box stop rule reproduce them? (2026-11-17)
Expectation written before running: a CPU simulation of 'each round r = start * 2^j visits every point in the CUBE (Linf <= r) of the query; the query stops at the first round whose cube holds >= k other points; the list is the k nearest by L2 among the points of that cube' reproduces the dumped lists (distances equal to 1e-6 relative) for at least 99.9 percent of all queries, including every class E query; if not, the cause is something else.
Usage: python3 check_tk_stoprule.py STAMP"""
import sys, numpy as np
from scipy.spatial import cKDTree
D = '/dev/shm/pw/'; K = 5; ts = sys.argv[1]
dt = np.dtype([('ind', '<i4'), ('dist', '<f4'), ('nn', '<i4'), ('pad', '<i4'), ('isect', '<i8')])
def load(f, n): return np.array(open(D + 'data/' + f).read().replace(',', ' ').split(), dtype=np.float64).reshape(-1, 3)[:n]
for nm, f, n, r0 in (('road', '3droad.csv', 434874, 2.745e-05), ('uni1m', 'uni1m.csv', 1000000, 0.001834), ('porto1m', 'porto1m.csv', 1000000, 9e-06)):
    x = np.fromfile(D + 'nb2_%s_fixed_%s.bin' % (nm, ts), dt).reshape(n, K)
    p = load(f, n).astype(np.float32).astype(np.float64); t = cKDTree(p)
    d0, t0 = t.query(p, k=K + 1); selfm = (t0 == np.arange(n)[:, None]); selfm[~selfm.any(axis=1), -1] = True; d = d0[~selfm].reshape(n, K)
    ind = x['ind'].astype(np.int64); real = np.sort(np.linalg.norm(p[ind] - p[:, None, :], axis=2), axis=1)
    nonex = (np.abs(real - d) > 1e-6 * np.maximum(d, 1e-30)).any(axis=1)
    # stop round per query
    r = np.float32(r0); rounds = []; stop = np.full(n, -1); act = np.arange(n); j = 0
    while len(act):
        cnt = t.query_ball_point(p[act], float(r), p=np.inf, return_length=True) - 1  # minus self
        done = cnt >= K; stop[act[done]] = j; rounds.append(float(r)); act = act[~done]; r = np.float32(r * 2); j += 1
        if j > 60: break
    pred = np.full((n, K), np.nan); fail = 0
    for jj, rr in enumerate(rounds):
        Q = np.where(stop == jj)[0]
        if not len(Q): continue
        kk = 32
        while len(Q):
            dd, ii = t.query(p[Q], k=kk, distance_upper_bound=float(np.sqrt(3) * rr * 1.0001))
            ok = np.isfinite(dd); lin = np.where(ok, np.abs(p[np.where(ok, ii, 0)] - p[Q][:, None, :]).max(axis=2), np.inf)
            inc = ok & (lin <= rr) & (ii != Q[:, None])
            enough = inc.sum(axis=1) >= K
            # complete if the kk results cover the whole ball of radius sqrt3 r (last slot not finite) or already K cube points among the first kk with the K th L2 below every omitted point (omitted points are farther than the kk th)
            complete = enough | (~ok[:, -1])
            for a in np.where(complete)[0]:
                v = dd[a][inc[a]][:K]; pred[Q[a], :len(v)] = v
            Q = Q[~complete]; kk *= 4
            if kk > 20000: fail += len(Q); break
    good = (np.abs(pred - real) <= 1e-6 * np.maximum(real, 1e-30)).all(axis=1)
    print('## %s n %d: rounds %d, query stop round counts %s' % (nm, n, len(rounds), np.bincount(stop[stop >= 0]).tolist()))
    print('non exact (rel 1e-6 vs exact) %d; simulation matches the dumped list in %d of %d queries (%.4f percent); among non exact: matches %d of %d; among exact: matches %d of %d; unresolved %d' % (nonex.sum(), good.sum(), n, 100.0 * good.mean(), (good & nonex).sum(), nonex.sum(), (good & ~nonex).sum(), (~nonex).sum(), fail))
    simne = (np.abs(pred - d) > 1e-6 * np.maximum(d, 1e-30)).any(axis=1); print('cube stop rule simulation itself is non exact (rel 1e-6 vs the true kNN) in %d queries (%.2f percent); program %d (%.2f percent); both non exact: %d; program non exact and simulation exact: %d; simulation non exact and program exact: %d' % (simne.sum(), 100.0 * simne.mean(), nonex.sum(), 100.0 * nonex.mean(), (simne & nonex).sum(), (nonex & ~simne).sum(), (simne & ~nonex).sum()))
    rs = np.array(rounds)[stop]; inside = d[:, -1] <= rs
    print('true k th distance inside the stop round radius (the cube holds the whole k th ball, the rule is guaranteed exact): %d queries, program non exact among them %d (%.3f percent); k th ball NOT inside the radius: %d queries, program non exact among them %d (%.2f percent)' % (inside.sum(), (nonex & inside).sum(), 100.0 * (nonex & inside).sum() / max(inside.sum(), 1), (~inside).sum(), (nonex & ~inside).sum(), 100.0 * (nonex & ~inside).sum() / max((~inside).sum(), 1)))
    # also: share of non exact queries whose stop cube holds more than k points beyond the kth ball (info)
    ne = np.where(nonex & ~good)[0][:5]
    for q in ne: print('  mismatch example', q, 'stop round', stop[q], 'dumped', real[q], 'pred', pred[q], 'true', d[q])
