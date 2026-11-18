"""Phase 11 B (fair RT vs CLOVER table): CPU preparation per dataset (2026-11-18). Writes fair_prep.json (to the given path, never overwritten) with
 N, TrueKNN start radius (smallest NONZERO distance among the 4 nearest neighbours of 100 random points, as F207), RTNN radius per K (smallest radius at which 99 percent of 3,000 sampled queries have K OTHER points inside, from the kd tree),
 and the sample of query rows used for exactness (5,000 rows, seed 11).  Usage: python3 -I fair_prep.py OUT.json
Data files: the same as F207 and F237 (3DRoad, Porto first 1M, uniform 1M, bunny, dragon, the three Arkade sets as the repo preprocessing gives them)."""
import sys, json, os, numpy as np
from scipy.spatial import cKDTree
D = '/local/scratch/a/cashman3/thirdparty/pw/data/'
SETS = [('3droad', '3droad.csv', 434874), ('porto1m', 'porto1m.csv', 1000000), ('uni1m', 'uni1m.csv', 1000000), ('bunny', 'bunny.csv', 362272), ('dragon', 'dragon.csv', 3609600),
        ('arkR_gowalla', 'arkR_gowalla.txt', 1280969), ('arkR_glove3d', 'arkR_glove3d.txt', 1193514), ('arkR_manuscript', 'arkR_manuscript.txt', 2155617)]
out = {}
for name, f, n in SETS:
    p = np.array(open(D + f).read().replace(',', ' ').split(), dtype=np.float64).reshape(-1, 3)[:n].astype(np.float32).astype(np.float64)
    assert len(p) == n, (name, len(p))
    t = cKDTree(p); rng = np.random.default_rng(5); q = p[rng.choice(n, 3000, replace=False)]
    d, _ = t.query(q, k=129, workers=-1)                      # column 0 is the query itself
    r99 = {K: float(np.quantile(d[:, K], 0.99)) for K in (5, 50, 128)}   # K other points inside
    rg = np.random.default_rng(3); q100 = p[rg.choice(n, 100, replace=False)]
    d4, _ = t.query(q100, k=5, workers=-1); nz = d4[:, 1:]; start = float(nz[nz > 0].min())
    out[name] = dict(file=f, N=n, start_radius=start, r99=r99)
    print(name, n, 'start', start, 'r99', r99, flush=True)
with open(sys.argv[1], 'x') as fo: json.dump(out, fo, indent=1)
