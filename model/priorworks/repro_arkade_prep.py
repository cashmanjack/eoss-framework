"""Arkade reproduction inputs (2026-11-17). Rule fixed before any run: a row reproduces the paper's Table 2 time (RTX 4070 Ti, search time, 10,000 queries) if our time on the RTX 6000 Ada
is within a factor 3 (different GPU, different radius procedure); otherwise reported as not reproduced, with the candidates per query as the first suspect.
Gowalla follows the repo's datasets/process_stanfordgowalla.py: distinct locations (location id), lat/lon to the unit sphere (x = cos lat cos lon, y = cos lat sin lon, z = sin lat).
Glove 3D and Manuscript as in F207 (the repo has no script for them). Queries: 10,000 random data points (seed 7), appended after the points. Program: the repo's s02-withTrueknn (pristine, KN 50, NORM 0 and NORM 1 builds), which doubles the radius and relaunches all queries each round;
its printed total search seconds is compared with Table 2. Start radius = initial_estimate.py: 100 random samples, nearest other SAMPLE in L1, smallest distance (seed 5).
Writes /dev/shm/pw/data/arkR_<ds>.txt and prints the table (radius per norm, candidates per query)."""
import numpy as np, gzip, json
from scipy.spatial import cKDTree
R = '/local/scratch/a/cashman3/rtdata/'; D = '/dev/shm/pw/data/'
def load(f): return np.array(open(D + f).read().split(), dtype=np.float64).reshape(-1, 3)
ids = {}; pts = []
for l in gzip.open(R + 'gowalla/loc-gowalla_totalCheckins.txt.gz', 'rt'):
    t = l.split('\t')
    if t[4] in ids: continue
    ids[t[4]] = 1; la, lo = np.deg2rad(float(t[2])), np.deg2rad(float(t[3])); pts.append((np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)))
gow = np.array(pts); print('gowalla distinct locations (by id):', len(gow))
out = {}
print('| dataset | N | start radius r0 |\n|---|---|---|')
for nm, p in (('gowalla', gow), ('glove3d', load('glove3d.txt')), ('manuscript', load('manuscript.txt'))):
    N = len(p); q = p[np.random.default_rng(7).choice(N, 10000, replace=False)]; t = cKDTree(p)
    np.savetxt(D + 'arkR_%s.txt' % nm, np.vstack([p, q]), fmt='%.9g')
    sm = p[np.random.default_rng(5).choice(N, 100, replace=False)]; dd = cKDTree(sm).query(sm, k=2, p=1)[0][:, 1]; r0 = float(dd.min())
    out['%s' % nm] = dict(N=N, r0=float('%.8g' % r0)); print('| %s | %d | %.6g |' % (nm, N, r0))
json.dump(out, open('/dev/shm/pw/repro_ark.json', 'w'))
