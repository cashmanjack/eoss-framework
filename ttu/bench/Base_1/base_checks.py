"""CPU reference checks for the baselines (numpy + scipy kd tree, float64). Each check takes files written by an UNTIMED run of the program
(or of an observer copy whose only difference is a dump line, diffs in notes/base1/*_observer_dump.diff) and returns a dict for the JSON."""
import numpy as np
from scipy.spatial import cKDTree

def load_csv(path, n=None, dim=None):
    a = np.loadtxt(path, delimiter=',', dtype=np.float64)
    return a if n is None else a[:n]

def knn_rounds(points, K, radii, active):
    """kNN (growing radius): Active(round i+1) must equal the number of queries with fewer than K points inside radius_i. Tests sphere and cube, self counted."""
    t = cKDTree(points); res = {}
    for name, p in (('sphere', 2), ('cube', np.inf)):
        fails = []
        for r in radii:
            cnt = np.array([len(x) for x in t.query_ball_point(points, r * (1 + 1e-6), p=p, workers=-1)])
            fails.append(int((cnt < K).sum()))
        res[name] = fails
    exp = active[1:] + [0] if len(active) else []
    for name in ('sphere', 'cube'):
        res[name + '_matches_program'] = res[name][:len(active) - 1] == active[1:len(active)]
    res['program_active'] = active
    return res

def arkade(out_file, pts, qs, radius, K):
    """Arkade: Neigh = {int ind; float dist (squared)}; per query the K nearest points inside the radius (2D). Compare returned index sets with the CPU set; for every
    query that differs, record how far the differing points lie from the radius (relative): float32 rounding at the boundary explains a difference only if that is tiny."""
    a = np.fromfile(out_file, dtype=np.dtype([('ind', '<i4'), ('dist', '<f4')])).reshape(len(qs), K)
    t = cKDTree(pts); n_exact = 0; diff = []; ind = a['ind']; sd = a['dist'].astype(np.float64)
    for q in range(len(qs)):
        d, i = t.query(qs[q], k=K, distance_upper_bound=radius * (1 + 1e-3))
        ok = np.isfinite(d); truth = set(i[ok][d[ok] <= radius * (1 + 1e-6)].tolist())
        got = set(ind[q][ind[q] >= 0].tolist())
        if got == truth: n_exact += 1; continue
        extra = got - truth; missing = truth - got
        # boundary distance of each differing point; also a differing point may be a tie at the K-th place (K = 50 full lists)
        dd = [abs(np.linalg.norm(pts[j] - qs[q]) - radius) / radius for j in (extra | missing)]
        diff.append(dict(q=q, got=len(got), want=len(truth), extra=len(extra), missing=len(missing), max_rel_distance_from_radius=float(max(dd)) if dd else None))
    q0 = int(np.argmax((ind >= 0).any(axis=1))); j = ind[q0][ind[q0] >= 0][0]; dd0 = np.linalg.norm(pts[j] - qs[q0])
    stored = sd[q0][list(ind[q0]).index(j)]
    return dict(queries=len(qs), identical_index_sets=n_exact, exact_fraction=n_exact / len(qs), differing_queries=len(diff),
                all_differences_within_1em4_of_radius=bool(diff) and all(x['max_rel_distance_from_radius'] is not None and x['max_rel_distance_from_radius'] < 1e-4 for x in diff) or not diff,
                differing=diff[:20], stored_dist_is='squared' if abs(stored - dd0 * dd0) < 1e-6 else ('plain' if abs(stored - dd0) < 1e-6 else 'other'))

def trueknn(fb_file, pts, K):
    """TrueKNN: Neigh {int ind; float dist; int numNeighbors; pad; long long intersections} (24 B). Exact K nearest (self removed by index)."""
    n = len(pts); dt = np.dtype([('ind', '<i4'), ('dist', '<f4'), ('nn', '<i4'), ('pad', '<i4'), ('isect', '<i8')])
    x = np.fromfile(fb_file, dt).reshape(n, K); p = pts.astype(np.float32).astype(np.float64); t = cKDTree(p)
    d0, t0 = t.query(p, k=K + 1, workers=-1)
    selfm = (t0 == np.arange(n)[:, None]); selfm[~selfm.any(axis=1), -1] = True
    keep = ~selfm; d = d0[keep].reshape(n, K)
    ind = x['ind'].astype(np.int64); valid = (ind >= 0) & (ind < n) & (ind != np.arange(n)[:, None])
    real = np.linalg.norm(p[np.where(valid, ind, 0)] - p[:, None, :], axis=2); real[~valid] = np.inf; real.sort(axis=1)
    sd = x['dist'].astype(np.float64)
    stored_is_sq = bool(np.allclose(np.sort(sd, axis=1)[valid.all(1)][:200], np.sort(real ** 2, axis=1)[valid.all(1)][:200], rtol=1e-4))
    exact = valid.all(1) & (np.abs(real - d) <= 1e-5 * np.maximum(d, 1e-12)).all(1)
    wrong = valid.all(1) & ~exact
    return dict(points=n, K=K, exact=int(exact.sum()), exact_fraction=float(exact.mean()), non_exact=int(wrong.sum()), invalid_rows=int((~valid.all(1)).sum()), stored_dist_squared=stored_is_sq)

def barneshut(dump_file, G=0.1, sample=2000, seed=1):
    """RT-BarnesHut: dump = N float forces then N x (mass, x, y, z) in the program's sorted body order (pointID order). Scalar force F_i = G m_i sum_{j!=i} m_j / r^2,
    float64 direct sum on a random sample, against the program's Barnes-Hut force (opening angle 0.5)."""
    raw = np.fromfile(dump_file, np.float32); n = len(raw) // 5
    F = raw[:n].astype(np.float64); b = raw[n:].reshape(n, 4).astype(np.float64); m, p = b[:, 0], b[:, 1:4]
    idx = np.random.default_rng(seed).choice(n, size=min(sample, n), replace=False); err = []
    for i in idx:
        r2 = ((p - p[i]) ** 2).sum(1); r2[i] = np.inf
        ref = G * m[i] * (m / r2).sum(); err.append(abs(F[i] - ref) / ref)
    err = np.array(err)
    return dict(bodies=n, sampled=len(idx), median_rel_err=float(np.median(err)), mean_rel_err=float(err.mean()), p99_rel_err=float(np.percentile(err, 99)), max_rel_err=float(err.max()),
                nan_or_zero=int((~np.isfinite(F) | (F == 0)).sum()))

def mochi(dump_file):
    """Mochi: dump = N Particle {center xyz, velocity xyz, radius, mass} (32 B) taken just before the collision launch of one sub step, then an int64 collision count
    summed from the program's col_count buffer for that launch. CPU: count ordered pairs (query i, hit j) with the device test
    (2 r_j > d) and (r_i + r_j > d) and (i < j or 2 r_i < d)."""
    raw = open(dump_file, 'rb').read(); n = (len(raw) - 8) // 32
    P = np.frombuffer(raw[:n * 32], np.float32).reshape(n, 8).astype(np.float64); prog = int(np.frombuffer(raw[n * 32:], np.int64)[0])
    c, r = P[:, :3], P[:, 6]; t = cKDTree(c); cpu = 0
    pairs = t.query_pairs(2 * r.max() * (1 + 1e-6), output_type='ndarray')
    for a, b in ((pairs[:, 0], pairs[:, 1]), (pairs[:, 1], pairs[:, 0])):          # both directions: query a hits primitive b
        d = np.linalg.norm(c[a] - c[b], axis=1)
        ok = (2 * r[b] - d > 0) & (-d + r[b] + r[a] > 0) & ((a < b) | (2 * r[a] - d < 0))
        cpu += int(ok.sum())
    return dict(particles=n, program_collisions=prog, cpu_collisions=cpu, equal=prog == cpu, overlapping_pairs_candidates=int(len(pairs)))

def knn_lists(prefix, pts, K, radii):
    """Growing radius kNN: per round dump (int na, int active_indices[na], uint32 heap[na][2K] = K pairs (prim id, dist^2 bits)). The program keeps the result of a query only in the
    round where its heap is full (count == K, later rounds only re-run the others), so the final answer of a query is its heap in the first round where all K slots are filled.
    CPU: exact K nearest of that query among all points (self counted), float64. exact = sorted distances equal within 1e-4 relative (the program uses fast math)."""
    n = len(pts); t = cKDTree(pts); d_true, _ = t.query(pts, k=K, workers=-1); done = np.zeros(n, bool); per_round = []
    ex_final = 0; bad_ids = 0; bad_d = 0
    for r, rad in enumerate(radii, 1):
        f = '%s_r%d.bin' % (prefix, r)
        raw = np.fromfile(f, np.uint32); na = int(raw[0]); act = raw[1:1 + na].astype(np.int64); heap = raw[1 + na:].reshape(na, 2 * K)
        ids = heap[:, 0::2]; d2 = heap[:, 1::2].view(np.float32).astype(np.float64); full = (ids != 0xFFFFFFFF).all(1)
        fi = np.where(full)[0]; q = act[fi]; ii = ids[fi].astype(np.int64)
        idv = (ii < n).all(1); ii = np.where(idv[:, None], ii, 0)
        real_d2 = ((pts[ii] - pts[q][:, None, :]) ** 2).sum(2)
        dok = idv & (np.abs(real_d2 - d2[fi]) <= 1e-4 * np.maximum(real_d2, 1e-12) + 1e-12).all(1)
        got = np.sqrt(np.sort(d2[fi], axis=1)); want = d_true[q]
        ex = idv & (np.abs(got - want) <= 1e-4 * np.maximum(want, 1e-9)).all(1)
        guaranteed = want[:, -1] <= rad * (1 - 1e-4)          # true K-th neighbour inside the radius (cube inscribes the sphere)
        per_round.append(dict(round=r, radius=rad, active=na, full_heaps=int(full.sum()), exact_of_full=int(ex.sum()), full_with_kth_inside_radius=int(guaranteed.sum()),
                              exact_of_those=int((ex & guaranteed).sum()), ids_or_dist_mismatch=int((~dok).sum())))
        ex_final += int(ex.sum()); bad_ids += int((~idv).sum()); bad_d += int((~dok).sum()); done[q] = True
    return dict(points=n, K=K, queries_with_a_full_heap=int(done.sum()), exact_final=ex_final, exact_fraction=ex_final / n, stored_dist_or_id_mismatch=bad_d, per_round=per_round)
