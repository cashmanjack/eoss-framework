"""Prior works on their datasets: PREDICTIONS (written before any run), 2026-11-17, triton (RTX 6000 Ada, 142 SMs), GPU 1.
Forms are the ones already in the repo, no refit to any of these datasets:
 RTNN    (F116/F127 form): hardware candidates = 1.10 x exact cube count (F125), ray ends after K sphere hits: C_eff = K (C+1) / (S+1) if S > K (F116);
         time = (c0 + cq (C_eff + 1)) N / (142 x 2.5e9), c0 110, cq 4.9 (F127 fixed part, fitted on the 4070 Ti). No miss term, no host work.
 TrueKNN (F158 rule): round time = 6.297e-7 ms x hits, hits = sum over active queries of the exact Chebyshev box count (self included); a query stops when count - 1 >= k; radius doubles.
 Arkade  (F149/F151/F154 on triton): device buffer, original (double pow) time = max(A, B): A = 1.97 ms x C / C_cal (throughput, calibration 100,000 uniform 2D points and queries, r 0.005, 1.97 ms = F151 original with device buffer, C = cube candidates),
         B = latency bound: largest warp (sum of the largest lane count per warp... see code) x 21,000 cycles per warp call / 2.5e9 (F133 unloaded call, double pow); fixed (float square) = 0.09 x original (F151: 0.18 / 1.97), B scaled the same.
         pinned host output adds 0.33 ms per MB of result buffer (F145/F151 on triton: 6.2 ms pinned against 1.97 device for 12.8 MB).
Usage: /usr/bin/python3 model/priorworks/predict_priorworks.py > model/priorworks/predictions_priorworks.md   (also writes /dev/shm/pw/data/ark_*.txt and model/priorworks/pred.json)"""
import numpy as np, json, os, math, sys
from scipy.spatial import cKDTree
D = '/dev/shm/pw/data/'; SM, CLK = 142, 2.5e9
def load(f):
    return np.array(open(D + f).read().replace(',', ' ').split(), dtype=np.float64).reshape(-1, 3)
def cube(p, q, r):  # exact Chebyshev count (self included when q is p)
    return cKDTree(p).query_ball_point(q, r, p=np.inf, return_length=True, workers=-1)
pred = {}
print('# Prior works on their datasets: predictions BEFORE any run (2026-11-17, triton GPU 1)\n')
print('Parameters the papers do not state are marked CHOSEN and fixed here, before any run. Counts are exact CPU counts on the files we actually have (see the notes on each dataset). Units: ms.\n')
# ---------------- RTNN
print('## RTNN (range search, K = 50 repo default, queries = points, flags -p 0 -s 0 -m 0 (the validated configuration of F116, no partitioning, default Morton grid sort), RTNN_DRAM_GB 11.5958 as in F167)\n')
print('The paper states no radius, K or query count for any dataset (checked in the paper text 2026-11-17). CHOSEN: the only radius the paper mentions is r = 0.1 on Buddha, whose points lie in a 131 cube: r1 = 7.6e-4 x the largest bounding box side, and r2 = 4 x r1 ("above 0.1 the speedup falls"). Not our own scaled units of the Stanford files: bunny scans merged with bun.conf transforms, 362,272 points; Dragon = xyzrgb_dragon vertices, 3,609,600 points. Buddha (4.6M) and KITTI 1M/12M/25M, Millennium: not available.\n')
print('| dataset | N | r | cube cand / query (C) | sphere hits (S) | C_eff | predicted ms |\n|---|---|---|---|---|---|---|')
for nm, f in (('bunny', 'bunny.csv'), ('dragon', 'dragon.csv')):
    p = load(f).astype(np.float32).astype(np.float64); N = len(p); ext = (p.max(0) - p.min(0)).max(); t = cKDTree(p)
    for tag, r in (('r1', 7.6e-4 * ext), ('r2', 3.04e-3 * ext)):
        C = t.query_ball_point(p, r, p=np.inf, return_length=True, workers=-1).mean(); S = t.query_ball_point(p, r, return_length=True, workers=-1).mean()
        Ce = 50 * (C + 1) / (S + 1) if S > 50 else C; q = 1.10 * Ce + 1; ms = (110 + 4.9 * q) * N / (SM * CLK) * 1e3
        pred['rtnn_%s_%s' % (nm, tag)] = dict(N=N, r=r, C=C, S=S, ms=ms)
        print('| %s | %d | %.5g | %.2f | %.2f | %.2f | %.3f |' % (nm, N, r, C, S, Ce, ms))
print('\nWhat can fail: more than a factor 2 off (the model is fitted on the 4070 Ti with 1e5 to 1e6 point uniform clouds, no miss term: the dragon BVH is about 250 MB, beyond the 96 MB L2); surface data has far fewer candidates per box than the uniform clouds of F116.\n')
# ---------------- TrueKNN
print('## TrueKNN (OWL port in /dev/shm/pw, original = managed memory frame buffer read by the host, fixed = device buffer; the distance is already float: no float fix applies)\n')
print('Paper parameters: first n points, k = 5 and k = sqrt(n), start radius = the smallest distance among the 4 nearest neighbours of 100 random points (we take the smallest NONZERO distance, since duplicates give 0: CHOSEN), radius doubles, time per round. Files: 3DRoad (repo file, 434,874 lines, paper 430K), Porto first 1,000,000 points of train.csv (which 1M the paper used: not stated, CHOSEN), UniformDist 1M (seed 11). k = sqrt(n) only for 3DRoad (k = 659): at 1M points k = 1000 needs 16 GB per round copied to the host, not run.\n')
TK = 6.297e-7
print('| dataset | n | k | start radius | rounds | active per round | hits per round (M) | predicted total ms |\n|---|---|---|---|---|---|---|---|')
for nm, f, ks in (('road', '3droad.csv', (5, 659)), ('porto1m', 'porto1m.csv', (5,)), ('uni1m', 'uni1m.csv', (5,))):
    p = load(f); n = len(p); t = cKDTree(p); rng = np.random.default_rng(5); ids = rng.choice(n, 100, replace=False)
    d = t.query(p[ids], k=5)[0][:, 1:]; r0 = float(d[d > 0].min())
    for k in ks:
        active = np.ones(n, bool); r = r0; tot = 0; acts = []; hs = []
        for rd in range(60):
            c = np.zeros(n, np.int64); c[active] = t.query_ball_point(p[active], r, p=np.inf, return_length=True, workers=-1)
            h = float(c[active].sum()); tot += TK * h; acts.append(int(active.sum())); hs.append(h / 1e6)
            active = active & (c - 1 < k)
            if not active.any(): break
            r *= 2
        pred['trueknn_%s_k%d' % (nm, k)] = dict(n=n, k=k, r0=r0, rounds=len(acts), ms=tot, acts=acts)
        print('| %s | %d | %d | %.4g | %d | %s | %s | %.2f |' % (nm, n, k, r0, len(acts), ' '.join(map(str, acts)), ' '.join('%.2f' % x for x in hs), tot))
print('\nWhat can fail: the rule is fitted on uniform 3D, k = 16 only (F158 held out total within 1.5 on two sizes only); 2D data with duplicates and heavy clusters, k = 5 and 659 not seen; managed memory effect not in the rule (original) at all.\n')
# ---------------- Arkade
print('## Arkade (sample01-knn, KN = 50, NORM = 2, 10,000 queries = a random subset (seed 7) of the data points appended to the file, device buffer + float square = fixed; original = pinned host buffer + double pow)\n')
print('Paper parameters: 10,000 queries, k = 1, 50, 100 (main k not stated: CHOSEN 50), radius from the TrueKNN procedure (the code for it is not in the sample: CHOSEN, smallest r0 x 2^i such that 99 percent of the 10,000 queries have 50 neighbours within r in L2, r0 as in TrueKNN). Datasets we have: Gowalla (unique lat,lon of the check ins: 1,256,693 against 1,270,969 in the paper, outliers kept), Glove 3D (PCA of the 25 d Twitter GloVe, 1,193,514 points against 1,183,514: our inference of the vector set), Manuscript (2,155,617 against 2,145,617). Cali OSM, Kitti 4M, Randnet, Gbif: not available.\n')
pts = load('uni2d_cal.txt') if False else None
a = np.array(open('/home/min/a/cashman3/eoss/notes/q17/a1e5.txt').read().split(), float).reshape(-1, 2)[:100000]; ap = np.c_[a, np.zeros(len(a))]
Ccal = cube(ap, ap, 0.005).sum() * 1.10
print('Calibration: 100,000 uniform 2D points and queries, r 0.005: cube candidates per query %.2f (x 1.10 hardware), total %.0f.\n' % (Ccal / 1.1 / 1e5, Ccal))
print('| dataset | N | r | cand / query | A throughput ms | B latency ms | fixed device ms | original device ms | original pinned ms |\n|---|---|---|---|---|---|---|---|---|')
for nm, f in (('gowalla', 'gowalla.txt'), ('glove3d', 'glove3d.txt'), ('manuscript', 'manuscript.txt')):
    p = load(f); N = len(p); rng = np.random.default_rng(7); qi = rng.choice(N, 10000, replace=False); q = p[qi]; t = cKDTree(p)
    ids = np.random.default_rng(5).choice(N, 100, replace=False); d = t.query(p[ids], k=5)[0][:, 1:]; r0 = float(d[d > 0].min())
    kd = t.query(q, k=50)[0][:, -1]; r = r0
    while (kd <= r).mean() < 0.99: r *= 2
    c = 1.10 * t.query_ball_point(q, r, p=np.inf, return_length=True, workers=-1).astype(float); nw = (len(c) + 31) // 32
    A = 1.97 * c.sum() / Ccal; calls = np.sort(c.reshape(-1)[: nw * 32 - (nw * 32 - len(c)) if False else len(c)]); pad = np.zeros(nw * 32); pad[:len(c)] = c
    B = pad.reshape(-1, 32).max(1).max() * 21000 / CLK * 1e3  # longest warp: its warp calls = its largest lane count, one call each
    orig = max(A, B); fixed = 0.09 * orig; pin = orig + 0.33 * 10000 * 50 * 8 / 1e6
    pred['arkade_%s' % nm] = dict(N=N, r=r, cand=float(c.mean()), A=A, B=B, fixed=fixed, orig=orig, pinned=pin)
    print('| %s | %d | %.5g | %.1f | %.3f | %.3f | %.3f | %.3f | %.3f |' % (nm, N, r, c.mean(), A, B, fixed, orig, pin))
    with open(D + 'ark_%s.txt' % nm, 'w') as o:
        np.savetxt(o, p, fmt='%.9g'); np.savetxt(o, q, fmt='%.9g')
print('\nWhat can fail: 10,000 queries are 313 warps on 142 SMs, a launch size never used in the calibration (100,000 queries, one wave); B is a bound that was never tested alone; 0.09 for float is from one uniform 2D set; real data have heavy candidate skew between lanes. Success rule for all three programs (fixed before): within a factor 2 of measured per row; reported as ratios, no refit.')
def rd(x):  # 10 significant digits (the push scan flags long digit runs)
    if isinstance(x, float): return float('%.10g' % x)
    if isinstance(x, dict): return {k: rd(v) for k, v in x.items()}
    if isinstance(x, list): return [rd(v) for v in x]
    return x
json.dump(rd(pred), open('/home/min/a/cashman3/eoss/model/priorworks/pred.json', 'w'), indent=1)
