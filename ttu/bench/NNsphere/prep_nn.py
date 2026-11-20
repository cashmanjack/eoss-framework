"""NNsphere data and radii (predictions: model/predictions_5090_batch10.md). Writes /dev/shm/nn/data/{uni1m,clu1m}.{bin,csv} and OUT.json (radii per K, never overwritten).
uni1m: numpy default_rng(11).random((1e6,3)) float32 (the F245 recipe). clu1m: 50 Gaussian clusters x 20,000 points, centres uniform in [0.1,0.9]^3 (seed 7), sigma 0.02 (seed 8). Both shuffled (seed 9) so the first queries are a random subset.
r(K) = quantile 0.99 of the distance to the K-th nearest point (self included) over 3,000 sampled queries (seed 5): the F245 convention, ~1 percent of lists short."""
import sys, json, os, numpy as np
from scipy.spatial import cKDTree
O = '/dev/shm/nn/data/'; os.makedirs(O, exist_ok=True)
uni = np.random.default_rng(11).random((1000000, 3)).astype(np.float32)
c = np.random.default_rng(7).uniform(0.1, 0.9, (50, 3)); g = np.random.default_rng(8)
clu = (np.repeat(c, 20000, axis=0) + g.normal(0, 0.02, (1000000, 3))).astype(np.float32)
res = {}
for name, p in (('uni1m', uni), ('clu1m', clu)):
    p = p[np.random.default_rng(9).permutation(len(p))] if name == 'clu1m' else p   # uni1m is already i.i.d. (the F245 file order)
    p.tofile(O + name + '.bin'); np.savetxt(O + name + '.csv', p, fmt='%.9g', delimiter=',')
    pd = p.astype(np.float64); t = cKDTree(pd); q = pd[np.random.default_rng(5).choice(len(p), 3000, replace=False)]
    d, _ = t.query(q, k=51, workers=-1); res[name] = {'N': len(p), 'r': {str(K): float(np.quantile(d[:, K - 1], 0.99)) for K in (5, 50)}, 'min': p.min(0).tolist(), 'max': p.max(0).tolist()}
    print(name, res[name], flush=True)
with open(sys.argv[1], 'x') as f: json.dump(res, f, indent=1)
