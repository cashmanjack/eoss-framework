"""Independent check of CLOVER bitonic results (second implementation, scipy cKDTree in float64, no code shared with clover_check2.cu). 2026-11-17.
Reads the dumps written by clover_check2 (returned indices and distances, every point a query, N = 20000). A query is wrong if its returned set has a point whose true distance exceeds the true K-th distance by more than 1e-6 relative (squared), or repeats an index.
Usage: /usr/bin/python3 model/priorworks/clover_indep_check.py  (needs the dumps in /dev/shm/pw/clv2)"""
import numpy as np, glob
from scipy.spatial import cKDTree
N = 20000
bun = np.array([l.replace(',', ' ').split() for l in open('/dev/shm/pw/data/bunny20k_unique.csv')], dtype=np.float32).astype(np.float64)
# Only the bunny sets are checked here (the uniform points come from std::mt19937 inside the C++ checker and are not reproduced in python).
print('| set | K | wrong queries (this check) | mean recall | note |\n|---|---|---|---|---|')
for K in (32, 64, 128):
    f = sorted(glob.glob('/dev/shm/pw/clv2/d_clover2_bunny_a0_k%d_*_idx.bin' % K))[-1]; idx = np.fromfile(f, '<i4').reshape(N, K); dist = np.fromfile(f.replace('_idx', '_dist'), '<f4').reshape(N, K)
    t = cKDTree(bun); d2 = t.query(bun, k=K + 1)[0] ** 2; kth = d2[:, K - 1]                       # K-th squared distance, self counts as neighbour 1
    ret = ((bun[idx] - bun[:, None, :]) ** 2).sum(2)                                              # true squared distances of the returned points
    far = (ret > kth[:, None] * (1 + 1e-6) + 1e-12).any(1); dup = np.array([len(set(r)) < K for r in idx])
    rec = (ret <= kth[:, None] * (1 + 1e-6) + 1e-12).sum(1).mean() / K
    print('| bunny 20,000 | %d | %d | %.3f | repeated index in %d queries; returned distances vs true: max abs err %.3g (squared-distance units 1 for float) |' % (K, (far | dup).sum(), rec, dup.sum(), np.abs(dist - ret).max()))
