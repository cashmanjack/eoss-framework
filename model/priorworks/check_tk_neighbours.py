"""TrueKNN original (managed memory) vs device buffer: are the final neighbour lists identical, and right? (2026-11-17)
Neigh struct: int ind, float dist, int numNeighbors, (pad 4), long long intersections = 24 bytes. k = 5. Reference: cKDTree, self excluded, duplicates count as neighbours (the program excludes only xID == primID).
Usage: python3 check_tk_neighbours.py"""
import numpy as np
from scipy.spatial import cKDTree
D = '/dev/shm/pw/'; K = 5
dt = np.dtype([('ind', '<i4'), ('dist', '<f4'), ('nn', '<i4'), ('pad', '<i4'), ('isect', '<i8')])
def load(f, n): return np.array(open(D + 'data/' + f).read().replace(',', ' ').split(), dtype=np.float64).reshape(-1, 3)[:n]
print('| dataset | n | ind identical | dist identical | intersections identical | numNeighbors all <= 0 (done) | vs CPU: distance mismatches (> 1e-5 rel) fixed / orig | vs CPU: returned ind is not at the stated distance (fixed / orig) |\n|---|---|---|---|---|---|---|---|')
for nm, f, n in (('road', '3droad.csv', 434874), ('uni1m', 'uni1m.csv', 1000000), ('porto1m', 'porto1m.csv', 1000000)):
    a = np.fromfile(D + 'nb_%s_fixed.bin' % nm, dt).reshape(n, K); b = np.fromfile(D + 'nb_%s_orig.bin' % nm, dt).reshape(n, K)
    p = load(f, n).astype(np.float32).astype(np.float64); t = cKDTree(p); d, _ = t.query(p, k=K + 1); d = d[:, 1:]
    res = []
    for x in (a, b):
        rel = np.abs(x['dist'] - d) / np.maximum(d, 1e-30); bad = int(((rel > 1e-5) & (np.abs(x['dist'] - d) > 1e-9)).sum())
        real = np.linalg.norm(p[x['ind'].clip(0)] - p[:, None, :], axis=2); bad2 = int((np.abs(real - x['dist']) > 1e-5 * np.maximum(real, 1e-12) + 1e-9).sum()); res.append((bad, bad2))
    print('| %s | %d | %s | %s | %s | %s | %d / %d | %d / %d |' % (nm, n, bool((a['ind'] == b['ind']).all()), bool((a['dist'] == b['dist']).all()), bool((a['isect'] == b['isect']).all()),
          bool(((a['nn'][:, 0] <= 0).all()) and ((b['nn'][:, 0] <= 0).all())), res[0][0], res[1][0], res[0][1], res[1][1]))
