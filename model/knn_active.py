"""Active set of the growing radius kNN program per round with the program's own stop rule (queue 13, CPU only).
The program (patched copy /dev/shm/knn, ~/knn) builds one AABB (cube, half width = radius) per data point, its intersection program reports every box the ray origin is inside
(the heap starts at the largest float, so every cube candidate is inserted), and the host stops a query when its heap holds K entries: cube candidates >= K (the query itself counts, queries = data).
Finding 130 used the sphere rule (neighbours within the radius) and predicted 5,411 queries in round 5, the program ran 83.
    /usr/bin/python3 model/knn_active.py PTS.bin OUTDIR [K R0 MULT]     writes OUTDIR/active_r<round>.bin (int32 query indices, launch order = index order)"""
import sys, os, numpy as np
from scipy.spatial import cKDTree
pts = np.fromfile(sys.argv[1], np.float32).reshape(-1, 3).astype(np.float64); out = sys.argv[2]; os.makedirs(out, exist_ok=True)
K = int(sys.argv[3]) if len(sys.argv) > 3 else 16; r = float(sys.argv[4]) if len(sys.argv) > 4 else 0.004; mult = float(sys.argv[5]) if len(sys.argv) > 5 else 2.0
tree = cKDTree(pts); act = np.arange(len(pts), dtype=np.int32); rnd = 0
while len(act) and rnd < 15:
    rnd += 1; act.tofile('%s/active_r%d.bin' % (out, rnd))
    cube = np.array([len(x) for x in tree.query_ball_point(pts[act], r * (1 + 1e-6), p=np.inf)])   # Chebyshev ball = the cube
    sph = np.array([len(x) for x in tree.query_ball_point(pts[act], r * (1 + 1e-6))])
    nxt = act[cube < K]; print('round %d radius %.4g active %d cube candidates mean %.3f (sphere %.3f) stop (cube) %d stop (sphere) %d next active %d' % (rnd, r, len(act), cube.mean(), sph.mean(), (cube >= K).sum(), (sph >= K).sum(), len(nxt)))
    act = nxt; r *= mult
