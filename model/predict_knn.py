"""kNN with growing radius (our ~/knn program, patched copy /dev/shm/knn, diff in notes/q12): predictions BEFORE the search launches are measured (RTX 4070 Ti, io GPU 0).
For each round: the CPU replay (model/replay/replay_aabb.cpp) over the GAS of the round (built by OptiX, dumped, no search launch) and the round's rays gives nodes, leaf slots and primitives in the
entered slots per query; the hardware candidate count is taken as 1.006 x that (Finding 125); lane efficiency from the per query counts of the replay in launch order; time from the refit fixed part
(Finding 127, fitted on all 12 RTNN configurations, nodes variant) plus the share model miss term from the reuse histograms of the round.
    /usr/bin/python3 model/predict_knn.py TAG   (TAG rand|morton; needs /dev/shm/knn/dump_TAG, pts_TAG.bin, act_TAG)"""
import os, subprocess, sys, json
import numpy as np
sys.path.insert(0, 'model'); sys.path.insert(0, 'ttu/bench/Memory_4')
import rtnn_refit as RR, reuse_feature as RF
from scipy.optimize import least_squares
tag = sys.argv[1]; K = '/dev/shm/knn/'; W = K + 'work_' + tag; os.makedirs(W, exist_ok=True)
sys.stdout = open('/dev/null', 'w') if False else sys.stdout
trc = RR.CF
f = least_squares(lambda p: RR.err(trc, p, 'nodes', 'fixed'), [100, 10], bounds=(0, np.inf)).x
lo = [least_squares(lambda p: RR.err([c for c in trc if c[0] == s], p, 'nodes', 'fixed'), [100, 10], bounds=(0, np.inf)).x for s in ('u1e5', 'u1e6')]
print('# kNN (growing radius) predictions, %s order, 200,000 points = queries, K 16, initial radius 0.004, doubling; written before measuring\n' % tag)
print('fixed part fit on the 12 RTNN configurations: c0 %.0f, cq %.1f cycles; per point set fits: %s\n' % (f[0], f[1], ', '.join('c0 %.0f cq %.1f' % tuple(x) for x in lo)))
print('| round | active | radius | nodes / query | leaf slots / query | prims in slots / query | lane efficiency (replay) | share of fetches beyond 480 lines | predicted launch ms (all fit; range of the two fits) |')
print('|---|---|---|---|---|---|---|---|---|')
tot = 0
for rd in range(1, 16):
    if not os.path.exists(K + 'dump_%s/gas_r%d.bin' % (tag, rd)): break
    D = W + '/r%d' % rd; os.makedirs(D, exist_ok=True)
    for a, b in (('gas0.bin', 'dump_%s/gas_r%d.bin' % (tag, rd)), ('points.bin', 'pts_%s.bin' % tag), ('aabb_radius0.txt', 'dump_%s/radius_r%d.txt' % (tag, rd)), ('rays0.bin', 'dump_%s/rays_r%d.bin' % (tag, rd))):
        if os.path.lexists(D + '/' + a): os.remove(D + '/' + a)
        os.symlink(K + b, D + '/' + a)
    subprocess.run(['/usr/bin/python3', 'model/replay/rtnn_prep.py', D, D + '/rp'], check=True, capture_output=True)
    N = os.path.getsize(D + '/rays0.bin') // 32
    subprocess.run(['/dev/shm/rp/replay_aabb', D + '/rp.table', D + '/rays0.bin', str(N), '0', D + '/rep.bin'], check=True)
    np.arange(N, dtype=np.int32).tofile(D + '/order.bin')
    big = N >= 100000; env = dict(os.environ, NSM='60', **({'ONEWIN': '1'} if big else {}))
    args = [] if big else [str(max(1, min(13, (N // 32 + 59) // 60))), str(max(1, (N // 32 + 59) // 60)), '0']
    subprocess.run(['/dev/shm/rp/reuse_aabb', D + '/rp.table', D + '/rays0.bin', str(N), D + '/order.bin', D + '/reuse.txt'] + args, check=True, env=env, capture_output=True)
    r = np.fromfile(D + '/rep.bin', np.int32).reshape(-1, 8); cand = r[:, 2] * 1.006
    w = cand[: N // 32 * 32].reshape(-1, 32); eff = w.sum() / max(1e-9, 32 * w.max(1).sum()) if len(w) else float('nan')
    h = RR.rhist(D + '/reuse.txt', 'nodes'); win = 1 if big else 1; rays = (39 * 60 * 32) if big else max(1, (N // 32 // 60 - 0)) * 60 * 32
    S1 = RR.KK * h[0] / rays; beyond = (h[1] + h[2] + h[3][480:].sum()) / max(1, h[0])
    miss = RR.PAR[1] * S1 * (1 - RF.share(h, RR.PAR[2], RR.PAR[3])) * (h[4] / h[0]) ** (-RR.PAR[4])
    q = r[:, 2].mean() + 1; ms = lambda p: (p[0] + p[1] * q + miss) * N / 1.5e11 * 1e3
    rng_ = sorted([ms(lo[0]), ms(lo[1])]); tot += ms(f)
    rad = float(open(D + '/aabb_radius0.txt').read().split()[0])
    print('| %d | %d | %.4g | %.2f | %.2f | %.3f | %.3f | %.3f | %.3f (%.3f to %.3f) |' % (rd, N, rad, r[:, 0].mean(), r[:, 1].mean(), r[:, 2].mean(), eff, beyond, ms(f), rng_[0], rng_[1]))
print('\nTotal predicted search launch time over the rounds: %.2f ms (sum of the launches only; host work between rounds, BVH builds and copies excluded).' % tot)
print('What can fail: candidates per query beyond 1 x 1.006 (the hardware count is for entered leaf slots, here many slots hold 1 point and queries rarely fall in any box in the first rounds, so the fixed cost dominates: the c0 range 50 to 420 cycles is the whole uncertainty of the early rounds); lane efficiency within 0.1; the active counts per round (CPU exact k-th neighbour distance, the program\'s own count would follow after the heap reset I patched in).')
