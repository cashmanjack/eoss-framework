"""Prior works, critical path bound: PREDICTIONS (written before they are compared with the F207 measurements, 2026-11-17, triton GPU 1). No constant is fitted here.
Form: for every launch, T = max(T_throughput, T_slowest_warp).
 T_throughput = the F207 form unchanged (RTNN: (110 + 4.9 q) N / (142 2.5e9) with q = 1.10 C_eff + 1 mean over queries; TrueKNN: 6.297e-7 ms x box hits of the round; Arkade: A of F207).
 T_slowest_warp = (largest per lane candidate count in the launch) x L / 2.5e9: the warp that holds that lane makes that many callback round trips one after another, whatever else the chip does. L = the SMALLEST unloaded latency per warp call
   that an earlier finding measured for the same kind of program (so the term is a lower bound on the chain):
   RTNN 575 cycles (F133, sorted RTNN, one warp per SM), TrueKNN 5,800 cycles (F133, kNN round 4, one warp per SM, k = 16: NO term for k, so k = 659 is not corrected), Arkade 21,000 cycles (F133/F207, the B of F207, double pow; fixed = 0.09 x as in F207).
 Counts are the exact CPU Chebyshev box counts of F207 (hardware factor 1.10 on RTNN and Arkade as in F207; TrueKNN counts as in F207).
 TrueKNN: the max is taken per round (one launch per round) and summed. RTNN: one launch. Arkade: one launch; its F207 form already was max(A, B), so its rows do not change by construction (stated here, scored anyway).
Success rule (as F207): measured within a factor 2 of predicted (ratio 0.5 to 2); also the median absolute log error per program against F207's.
Usage: /usr/bin/python3 model/priorworks/predict_critical_path.py > model/priorworks/predictions_critical_path.md  (writes pred_cp.json)"""
import numpy as np, json
from scipy.spatial import cKDTree
D = '/dev/shm/pw/data/'; SM, CLK = 142, 2.5e9; L_RT, L_TK, L_AR = 575., 5800., 21000.
def load(f): return np.array(open(D + f).read().replace(',', ' ').split(), dtype=np.float64).reshape(-1, 3)
old = json.load(open('/home/min/a/cashman3/eoss/model/priorworks/pred.json')); pred = {}
print('# Prior works: critical path bound, predictions BEFORE scoring (2026-11-17, triton GPU 1)\n\nT = max(throughput, slowest warp), per launch. Constants: RTNN L = 575, TrueKNN L = 5,800, Arkade L = 21,000 cycles per warp call (smallest unloaded values of F133; none fitted here). Units ms.\n')
print('## RTNN\n\n| dataset | r | F207 throughput | max lane q | slowest warp | prediction | chain binds? |\n|---|---|---|---|---|---|---|')
for nm, f in (('bunny', 'bunny.csv'), ('dragon', 'dragon.csv')):
    p = load(f).astype(np.float32).astype(np.float64); N = len(p); ext = (p.max(0) - p.min(0)).max(); t = cKDTree(p)
    for tag, r in (('r1', 7.6e-4 * ext), ('r2', 3.04e-3 * ext)):
        C = t.query_ball_point(p, r, p=np.inf, return_length=True, workers=-1).astype(float); S = t.query_ball_point(p, r, return_length=True, workers=-1).astype(float)
        Ce = np.where(S > 50, 50 * (C + 1) / (S + 1), C); q = 1.10 * Ce + 1
        thr = old['rtnn_%s_%s' % (nm, tag)]['ms']; ch = q.max() * L_RT / CLK * 1e3; ms = max(thr, ch)
        pred['rtnn_%s_%s' % (nm, tag)] = dict(thr=thr, chain=ch, ms=ms, maxq=float(q.max()))
        print('| %s | %.5g | %.3f | %.1f | %.3f | %.3f | %s |' % (nm, r, thr, q.max(), ch, ms, 'yes' if ch > thr else 'no'))
print('\n## TrueKNN (fixed = device buffer; per round max, summed)\n\n| dataset | k | F207 throughput total | slowest warp per round (ms) | prediction total | rounds where chain binds |\n|---|---|---|---|---|---|')
TK = 6.297e-7
for nm, f, ks in (('road', '3droad.csv', (5, 659)), ('porto1m', 'porto1m.csv', (5,)), ('uni1m', 'uni1m.csv', (5,))):
    p = load(f); n = len(p); t = cKDTree(p); ids = np.random.default_rng(5).choice(n, 100, replace=False); d = t.query(p[ids], k=5)[0][:, 1:]; r0 = float(d[d > 0].min())
    for k in ks:
        active = np.ones(n, bool); r = r0; tot = 0; thr_tot = 0; chains = []; nb = 0
        for rd in range(60):
            c = np.zeros(n); c[active] = t.query_ball_point(p[active], r, p=np.inf, return_length=True, workers=-1)
            th = TK * float(c[active].sum()); ch = c[active].max() * L_TK / CLK * 1e3; tot += max(th, ch); thr_tot += th; chains.append(ch); nb += ch > th
            active = active & (c - 1 < k)
            if not active.any(): break
            r *= 2
        pred['trueknn_%s_k%d' % (nm, k)] = dict(thr=thr_tot, chain=[float('%.6g' % x) for x in chains], ms=tot, bind=int(nb))
        print('| %s | %d | %.2f | %s | %.2f | %d of %d |' % (nm, k, thr_tot, ' '.join('%.3f' % x for x in chains), tot, nb, len(chains)))
print('\n## Arkade (event time of the launch; F207 already was max(A, B) with B the slowest warp x 21,000 cycles, so these rows equal F207 by construction)\n\n| dataset | fixed | original pinned |\n|---|---|---|')
for nm in ('gowalla', 'glove3d', 'manuscript'):
    o = old['arkade_' + nm]; pred['arkade_' + nm] = dict(fixed=o['fixed'], pinned=o['pinned'], A=o['A'], B=o['B'])
    print('| %s | %.3f | %.3f |' % (nm, o['fixed'], o['pinned']))
print('\nWhat can fail: L is a lower bound from small launches of other data (k = 16 for TrueKNN), so rows where one lane has many candidates may stay under; k = 659 has no k term; the chain ignores traversal latency (not in the count data).')
def rd(x):
    if isinstance(x, float): return float('%.10g' % x)
    if isinstance(x, dict): return {k: rd(v) for k, v in x.items()}
    if isinstance(x, list): return [rd(v) for v in x]
    return x
json.dump(rd(pred), open('/home/min/a/cashman3/eoss/model/priorworks/pred_cp.json', 'w'), indent=1)
