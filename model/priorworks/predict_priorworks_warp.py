"""Prior works, variant (a): per warp SLOWEST LANE counts instead of mean counts (2026-11-17, triton). CPU only, no refit, same constants as predict_priorworks.py.
NOT a blind prediction: the F207 measurements were already known when this was written. The form and the success rule (within a factor 2 per row; and whether the median log error shrinks)
are fixed here, before the comparison script is run.
Lane = one query in launch order. Warp = 32 consecutive lanes. Every lane is charged its warp's largest count (idle lanes charge nothing):
 RTNN    (-p 0 -s 0: queries are NOT sorted, launch order = file order, checked in sort.cpp): lane cost q_i = 1.10 C_eff_i + 1 -> warp max; time = (c0 + cq mean_i(max_w q)) N / (142 2.5e9), c0 110, cq 4.9.
 TrueKNN (one launch of ALL n points per round, 31,250 warps at 1M, inactive lanes exit): hits_eff = sum over warps (active lanes in warp x max box count among active lanes); round time = 6.297e-7 ms x hits_eff.
 Arkade  (queries in file order): A = 1.97 ms x sum_w(lanes x max_w c) / same sum for the calibration set (100,000 uniform 2D points, file order); B as before (largest warp max x 21,000 cycles); fixed = 0.09 x max(A,B); pinned adds 0.33 ms per MB.
Usage: python3 model/priorworks/predict_priorworks_warp.py > model/priorworks/predictions_priorworks_warp.md (also writes pred_warp.json)"""
import numpy as np, json
from scipy.spatial import cKDTree
D = '/dev/shm/pw/data/'; SM, CLK = 142, 2.5e9
def load(f): return np.array(open(D + f).read().replace(',', ' ').split(), dtype=np.float64).reshape(-1, 3)
def wmax(x, act=None):  # x per lane; returns sum over warps of (active lanes x max over active lanes)
    n = len(x); nw = (n + 31) // 32; v = np.zeros(nw * 32); v[:n] = x; a = np.zeros(nw * 32, bool); a[:n] = True if act is None else act
    v = np.where(a, v, 0).reshape(-1, 32); a = a.reshape(-1, 32); m = v.max(1); return float((a.sum(1) * m).sum()), m, a.sum(1)
pred = {}
print('# Prior works, variant (a): per warp slowest lane (post hoc recomputation, no refit)\n\nSame ingredients as predictions_priorworks.md; only the count that enters the model changes. Units ms.\n')
print('## RTNN\n\n| dataset | r | mean form ms (F207) | warp max form ms |\n|---|---|---|---|')
old = json.load(open('/home/min/a/cashman3/eoss/model/priorworks/pred.json'))
for nm, f in (('bunny', 'bunny.csv'), ('dragon', 'dragon.csv')):
    p = load(f).astype(np.float32).astype(np.float64); N = len(p); ext = (p.max(0) - p.min(0)).max(); t = cKDTree(p)
    for tag, r in (('r1', 7.6e-4 * ext), ('r2', 3.04e-3 * ext)):
        C = t.query_ball_point(p, r, p=np.inf, return_length=True, workers=-1).astype(float); S = t.query_ball_point(p, r, return_length=True, workers=-1).astype(float)
        Ce = np.where(S > 50, 50 * (C + 1) / (S + 1), C); q = 1.10 * Ce + 1; s, m, na = wmax(q); qe = s / N
        ms = (110 + 4.9 * qe) * N / (SM * CLK) * 1e3; pred['rtnn_%s_%s' % (nm, tag)] = dict(ms=ms, mean_q=q.mean(), warp_q=qe)
        print('| %s | %.5g | %.3f | %.3f |' % (nm, r, old['rtnn_%s_%s' % (nm, tag)]['ms'], ms))
print('\n## TrueKNN (fixed = device buffer)\n\n| dataset | k | mean form ms | warp max form ms | rounds |\n|---|---|---|---|---|')
TK = 6.297e-7
for nm, f, ks in (('road', '3droad.csv', (5, 659)), ('porto1m', 'porto1m.csv', (5,)), ('uni1m', 'uni1m.csv', (5,))):
    p = load(f); n = len(p); t = cKDTree(p); rng = np.random.default_rng(5); ids = rng.choice(n, 100, replace=False)
    d = t.query(p[ids], k=5)[0][:, 1:]; r0 = float(d[d > 0].min())
    for k in ks:
        active = np.ones(n, bool); r = r0; tot = 0; rds = []
        for rd in range(60):
            c = np.zeros(n); c[active] = t.query_ball_point(p[active], r, p=np.inf, return_length=True, workers=-1)
            s, _, _ = wmax(c, active); tot += TK * s; rds.append(s / 1e6)
            active = active & (c - 1 < k)
            if not active.any(): break
            r *= 2
        pred['trueknn_%s_k%d' % (nm, k)] = dict(ms=tot, rounds=len(rds), hits_eff_M=[float('%.6g' % x) for x in rds])
        print('| %s | %d | %.2f | %.2f | %d |' % (nm, k, old['trueknn_%s_k%d' % (nm, k)]['ms'], tot, len(rds)))
print('\n## Arkade (fixed = device buffer + float square; original pinned)\n')
a = np.array(open('/home/min/a/cashman3/eoss/notes/q17/a1e5.txt').read().split(), float).reshape(-1, 2)[:100000]; ap = np.c_[a, np.zeros(len(a))]
cc = cKDTree(ap).query_ball_point(ap, 0.005, p=np.inf, return_length=True, workers=-1) * 1.10; Ccal_w = wmax(cc.astype(float))[0]
print('Calibration warp max sum %.0f (mean form total %.0f).\n' % (Ccal_w, cc.sum()))
print('| dataset | mean fixed | warp fixed | mean original pinned | warp original pinned | A mean | A warp | B |\n|---|---|---|---|---|---|---|---|')
for nm, f in (('gowalla', 'gowalla.txt'), ('glove3d', 'glove3d.txt'), ('manuscript', 'manuscript.txt')):
    o = old['arkade_' + nm]; p = load(f); N = len(p); rng = np.random.default_rng(7); qi = rng.choice(N, 10000, replace=False); q = p[qi]; t = cKDTree(p)
    c = 1.10 * t.query_ball_point(q, o['r'], p=np.inf, return_length=True, workers=-1).astype(float)
    Aw = 1.97 * wmax(c)[0] / Ccal_w; B = o['B']; orig = max(Aw, B); fixed = 0.09 * orig; pin = orig + 0.33 * 10000 * 50 * 8 / 1e6
    pred['arkade_' + nm] = dict(A=Aw, B=B, fixed=fixed, orig=orig, pinned=pin)
    print('| %s | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f |' % (nm, o['fixed'], fixed, o['pinned'], pin, o['A'], Aw, B))
print('\nSuccess rule (fixed before): a row is closed if it is within a factor 2 of the F207 measurement; the variant helps a program if the median absolute log error over its rows falls. Not blind (see the header).')
json.dump(pred, open('/home/min/a/cashman3/eoss/model/priorworks/pred_warp.json', 'w'), indent=1)
