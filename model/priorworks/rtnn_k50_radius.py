# Radius at which 99 percent of the queries have at least 50 neighbours (self included), for the RTNN rows (bunny, dragon).
# CPU only, scipy kd tree, 3000 random queries of the data (seed 5). Output: one line per dataset.
import sys, numpy as np
from scipy.spatial import cKDTree
D = '/local/scratch/a/cashman3/thirdparty/pw/data/'
for name in sys.argv[1:]:
    P = np.loadtxt(D + name + '.csv', delimiter=',', dtype=np.float64)
    t = cKDTree(P); rng = np.random.default_rng(5); q = P[rng.choice(len(P), 3000, replace=False)]
    d, _ = t.query(q, k=50)
    kd = d[:, 49]
    print(name, 'N', len(P), 'r99', float(np.quantile(kd, 0.99)), 'r50pct', float(np.median(kd)), 'F207 r1/r2 see predictions_priorworks.md', flush=True)
