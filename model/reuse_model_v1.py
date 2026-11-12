"""Model v1 with its miss inputs (F: L1 fill sectors per ray, D: DRAM bytes per ray) taken from the CPU replay reuse distance histograms (queue 6).
    /usr/bin/python3 model/reuse_model_v1.py REUSE_DIR TIMING.json OUT.txt
REUSE_DIR = notes/q5/reuse (reuse_<scene>_<key>.txt from model/replay/reuse.cpp), TIMING = notes/q3/depth4_wset_23.json (measured speedups over random order, OptiX, 8.4M rays, 0 wrong).
Per ray inputs from the histograms (unit sm, nodes only, dedupe, C = 480 lines = 60 KB LRU): lines requested per ray a(key), misses per ray m(key) = (cold + overflow + distance >= C) / rays.
  rays in the stream = 142 SMs x 2 windows x 39 measured warps x 32 lanes = 354,432.
  S1(key) = k a(key)  (L1 sectors per ray), F(key) = k m(key)  (fill sectors per ray); k (sectors per line request) is ONE constant per run, set so that S1(random, 1e5) = 82 sectors per ray (ncu, OptiX, Finding 95).
  D(key) = 128 B x (cold + overflow)/rays x max(0, 1 - 96 MB / BVH bytes)  (crude: a request that leaves the SM's 1 MB horizon goes to DRAM with the share of the BVH that is not resident in L2); 0 for 1e5.
  q = queries per ray (+ idle lane queries) from Memory_4 (1e5: 1.876 + 0.34; 3e6: from the same file), W = 13 warps per SM.
(a) Model v1 as fitted (notes/model_v1.md parameters, no refit): predicted speedup(key) = c(random) / c(key), c = 1/predict_sm.
(b) The same model with its miss coefficients refit: c = A + B F + Cd D on one scene, tested on the other (both directions); NNLS, no intercept constraints beyond >= 0.
Compared with the LRU only model of Finding 98 (P0: ratio of miss rates; P1: one fitted constant, same cross scene protocol; P0 also at its best cache size), by rms log error against the measured speedups.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import rettu_model as M
from scipy.optimize import nnls
import run_model as RM
KEYS = ('random', 'd1', 'd2', 'd3', 'd4', 'full')
RAYS = 142 * 2 * 39 * 32
BVH_MB = {'1e5': 6.5, '3e6': 186.0}     # BVH sizes in MB: 1e5 about 6.5 MB, 3e6 186 MB


def load(d, scene, key):
    for l in open(os.path.join(d, 'reuse_%s_%s.txt' % (scene, key))):
        if l.startswith('sm nodes dedupe'):
            p = l.split(); return int(p[3]), int(p[4]), int(p[5]), np.array(p[6:], np.int64)


def main(d, timing, out):
    T = json.load(open(timing)); L = []
    def line(*a): s = ' '.join(str(x) for x in a); print(s); L.append(s)
    p = M.fit(RM.memory4_rows()); line('model v1 parameters (fit on Memory_4 one SM data):', {k: round(v, 3) for k, v in p.items()})
    mem = RM.memory4_rows()
    q = {s: np.mean([r['q'] for r in mem if r['scene'] == 'soup' + n and r['variant'] == 'interleaved']) for s, n in (('1e5', '100000'), ('3e6', '3000000'))}
    H = {s: {k: load(d, s, k) for k in KEYS} for s in ('1e5', '3e6')}
    a = {s: np.array([H[s][k][0] / RAYS for k in KEYS]) for s in H}
    m = {s: np.array([(H[s][k][1] + H[s][k][2] + H[s][k][3][480:].sum()) / RAYS for k in KEYS]) for s in H}
    ov = {s: np.array([(H[s][k][1] + H[s][k][2]) / RAYS for k in KEYS]) for s in H}
    kk = 82.0 / a['1e5'][0]; line('sectors per line request k = %.2f; lines requested per ray (random, d1..full): 1e5 %s, 3e6 %s' % (kk, ' '.join('%.1f' % v for v in a['1e5']), ' '.join('%.1f' % v for v in a['3e6'])))
    meas = {s: np.array([T[s]['timing'][k]['speedup'] for k in KEYS]) for s in H}
    feats = {}
    for s in H:
        share = max(0.0, 1 - 96.0 / BVH_MB[s])
        feats[s] = [dict(q=q[s], S1=kk * a[s][i], F=kk * m[s][i], D=128 * ov[s][i] * share, W=13) for i in range(len(KEYS))]
        line(s, 'q %.2f; S1 %s; F %s; D %s' % (q[s], ' '.join('%.1f' % r['S1'] for r in feats[s]), ' '.join('%.1f' % r['F'] for r in feats[s]), ' '.join('%.1f' % r['D'] for r in feats[s])))
    def rms(pred, s): return float(np.sqrt(np.mean(np.log(pred / meas[s]) ** 2)))
    res = {}
    # (a) v1 as fitted
    for s in H:
        c = np.array([1 / M.predict_sm(p, r) for r in feats[s]]); pa = c[0] / c; res[('v1', s)] = pa
        line('(a) v1 as fitted,', s, 'predicted', ' '.join('%.2f' % v for v in pa), '| measured', ' '.join('%.2f' % v for v in meas[s]), '| rms log error %.3f' % rms(pa, s))
    # (a2) v1 with F and D inputs from the histograms but also the measured S1 held at its random value (only misses vary)
    for s in H:
        fx = [dict(r, S1=feats[s][0]['S1']) for r in feats[s]]; c = np.array([1 / M.predict_sm(p, r) for r in fx]); pa = c[0] / c; res[('v1_S1fixed', s)] = pa
        line('(a2) v1 misses only (S1 fixed),', s, 'predicted', ' '.join('%.2f' % v for v in pa), '| rms log error %.3f' % rms(pa, s))
    # (b) refit miss coefficients: c = A + B F + Cd D/32 (cycles per ray), cross scene; the scene's absolute level is unknown for OptiX, so fit on speedups: c(key)/c(random) = 1 / speedup
    for tr, te in (('1e5', '3e6'), ('3e6', '1e5')):
        X = np.array([[1.0, r['F'], r['D'] / 32.0] for r in feats[tr]]); y = 1.0 / meas[tr]      # time relative to random, with A normalised by the random row
        # parametrise c(key) = A + B F + C D, normalised so c(random) = 1: fit (A, B, C) >= 0 to y, then rescale
        co, _ = nnls(X, y); Xt = np.array([[1.0, r['F'], r['D'] / 32.0] for r in feats[te]]); ct = Xt @ co; pb = ct[0] / ct
        # the cross scene transfer needs the same A scale: express in units of the random time of each scene: c_te = A_te + B F + C D with A_te unknown -> use A_te = A * (S1 random te / S1 random tr)
        sc = feats[te][0]['S1'] / feats[tr][0]['S1']; Xs = np.array([[sc, r['F'], r['D'] / 32.0] for r in feats[te]]); cs = Xs @ co; ps = cs[0] / cs
        line('(b) refit on %s (A %.3f B %.4f C %.4f), applied to %s: same A %s ; A scaled by S1 %s | measured %s | rms log error %.3f / %.3f' % (tr, *co, te, ' '.join('%.2f' % v for v in pb), ' '.join('%.2f' % v for v in ps), ' '.join('%.2f' % v for v in meas[te]), rms(pb, te), rms(ps, te)))
        res[('refit', te)] = pb; res[('refit_scaled', te)] = ps
    # LRU only (Finding 98): P0 = miss ratio of random over key (nodes only, dedupe, SM, 60 KB), also the best cache size over both scenes; P1 cross scene
    for s in H:
        p0 = m[s][0] / m[s]; res[('P0', s)] = p0; line('(c) LRU only P0 (60 KB),', s, 'predicted', ' '.join('%.2f' % v for v in p0), '| rms log error %.3f' % rms(p0, s))
    rate = {s: np.array([(H[s][k][1] + H[s][k][2] + H[s][k][3][480:].sum()) / H[s][k][0] for k in KEYS]) for s in H}
    for s in H:
        p0 = rate[s][0] / rate[s]; res[('P0rate', s)] = p0; line('(c) LRU only P0, miss RATE ratio as in Finding 98 (60 KB),', s, 'predicted', ' '.join('%.2f' % v for v in p0), '| rms log error %.3f' % rms(p0, s))
    line('combined rms log error over both scenes (12 points):')
    for nm in ('v1', 'v1_S1fixed', 'P0', 'P0rate'):
        line('   %-12s %.3f' % (nm, float(np.sqrt(np.mean([rms(res[(nm, s)], s) ** 2 for s in H])))))
    line('   refit cross scene (fit on one, test on the other): %.3f; with A scaled by S1: %.3f' % (float(np.sqrt(np.mean([rms(res[('refit', s)], s) ** 2 for s in H]))), float(np.sqrt(np.mean([rms(res[('refit_scaled', s)], s) ** 2 for s in H])))))
    best = None
    for C in (60, 120, 240, 480, 720, 960, 1440, 1920, 2880):
        e = 0
        mm = {s: np.array([(H[s][k][1] + H[s][k][2] + H[s][k][3][C:].sum()) / RAYS for k in KEYS]) for s in H}
        for s in H: e += np.sum(np.log((mm[s][0] / mm[s]) / meas[s]) ** 2)
        e = np.sqrt(e / 12); best = min(best or (9, 0), (e, C))
    line('(c) LRU only P0 at its best capacity over both scenes: C %d lines (%d KB), rms log error %.3f (in sample, one parameter)' % (best[1], best[1] * 128 // 1024, best[0]))
    fs = np.linspace(0, 0.9, 901)
    for tr, te in (('1e5', '3e6'), ('3e6', '1e5')):
        def p1(s, f): mm = m[s]; return 1 / (f + (1 - f) * mm / mm[0])
        err = [np.sqrt(np.mean(np.log(p1(tr, f) / meas[tr]) ** 2)) for f in fs]; f = fs[int(np.argmin(err))]
        line('(c) LRU only P1, f fit on %s = %.3f, applied to %s: %s | rms log error %.3f' % (tr, f, te, ' '.join('%.2f' % v for v in p1(te, f)), rms(p1(te, f), te)))
    open(out, 'w').write('\n'.join(L) + '\n')


if __name__ == '__main__': main(*sys.argv[1:4])
