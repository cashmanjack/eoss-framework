"""CLOVER results (every point a query) against the true K nearest for ALL queries (F227 addendum, 2026-11-17). CPU only. Dumps from notes/q38/run_clover_full.sh (clover_bench.cu with DUMPFILE).
CLOVER counts the query point itself as one of the K (distance 0). A query is exactly correct if its K returned points are distinct, valid, and their sorted distances equal the true K smallest distances (kd tree, float32 points widened to double), tolerance 1e-6 relative (equal distance ties are not errors).
Usage: python3 -I model/priorworks/check_clover_full.py DUMPDIR"""
import sys, glob, re, os, numpy as np
from scipy.spatial import cKDTree
D = '/local/scratch/a/cashman3/thirdparty/pw/data/'; dd = sys.argv[1]
print('| method | dataset | K | queries | exactly correct | invalid slots | queries with a farther than true slot |\n|---|---|---|---|---|---|---|')
names = {'hubs': 'CLOVER hubs', 'bitp': 'CLOVER bitonic (patched)', 'bits': 'CLOVER bitonic (stock)', 'warp': 'CLOVER warpwise'}
cache = {}
for f in sorted(glob.glob(dd + '/*.bin')):
    m = re.match(r'(\w+?)_(\w+?)_k(\d+)_', os.path.basename(f)); tag, ds, K = m[1], m[2], int(m[3])
    csv = {'3droad': '3droad.csv'}.get(ds, ds + '.csv')
    if ds not in cache:
        p = np.array(open(D + csv).read().replace(',', ' ').split(), dtype=np.float64).reshape(-1, 3).astype(np.float32).astype(np.float64); cache[ds] = (p, cKDTree(p))
    p, t = cache[ds]; N = len(p); res = np.fromfile(f, np.int32).reshape(N, K)
    ok = np.zeros(N, bool); inv = 0; far = np.zeros(N, bool)
    for s in range(0, N, 200000):
        sl = slice(s, min(N, s + 200000)); dtrue = t.query(p[sl], k=K, workers=-1)[0].reshape(-1, K); rr = res[sl]
        valid = (rr >= 0) & (rr < N); idx = np.where(valid, rr, 0).astype(np.int64)
        dr = np.linalg.norm(p[idx] - p[sl][:, None, :], axis=2); dr[~valid] = np.inf
        srt = np.sort(rr, axis=1); dup = (srt[:, 1:] == srt[:, :-1]).any(1); bad = (~valid).any(1) | dup; inv += int(bad.sum())
        dr.sort(axis=1); eq = (np.abs(dr - dtrue) <= 1e-6 * np.maximum(dtrue, 1e-12)).all(1)
        ok[sl] = eq & ~bad; far[sl] = (dr > dtrue * (1 + 1e-6) + 1e-12).any(1)
    print('| %s | %s | %d | %d | %.4f (%d wrong) | %d | %d |' % (names[tag], ds, K, N, ok.mean(), int((~ok).sum()), inv, int(far.sum())), flush=True)
