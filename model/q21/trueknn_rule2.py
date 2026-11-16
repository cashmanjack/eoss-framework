"""TrueKNN second attempt (queue 21). Rule chosen from the 100,000 point calibration set A ONLY (notes/q20/tk/A_*.txt); nothing from the 200,000 / 400,000 measurements is used.
The point files of F156 are gone (/dev/shm), so the CPU features come from regenerated uniform points (numpy, fixed seed); the distribution is the same, the exact points are not.
Features per round r (Chebyshev box of half width radius, exact, self included, queries with count - 1 >= K stop):
  calls   = sum over warps of the largest lane count (F156's feature, times rho)       hits = sum over active queries of the box count (candidates reaching the program)
  inserts = sum over active queries of expected heap insertions: n if n <= K else K (1 + ln(n / K))   (random candidate order, a candidate is inserted when it is among the K closest so far)
Time of round r = tau x feature_r / (142 x clk), one pooled constant tau per rule (SM cycles per feature unit) from the A total. Rule selection: smallest rms log error of the per round time on A.
    /usr/bin/python3 model/q21/trueknn_rule2.py [select|predict]"""
import re, sys, math, numpy as np
from scipy.spatial import cKDTree
K = 16; R0 = 0.004; SM = 142; CLK = 2.5e9
def loadA():
    rows = {}
    for r in range(1, 6):
        for l in open('notes/q20/tk/A_%d.txt' % r):
            m = re.match(r'RND (\d+) radius (\S+) GT span_ms (\S+) CBK warps (\d+) total_warp_cycles (\d+) f0_cycles (\d+) f0_calls (\d+)', l)
            if m: rows.setdefault(int(m[1]), []).append([float(m[3]), float(m[5]), float(m[6]), float(m[7])])
    return {k: np.median(np.array(v), axis=0) for k, v in rows.items()}
def feats(n, seed):
    p = np.random.default_rng(seed).random((n, 3)); t = cKDTree(p); active = np.ones(n, bool); out = []; r = R0
    for rd in range(1, 12):
        c = np.zeros(n, np.int64); c[active] = t.query_ball_point(p[active], r, p=np.inf, return_length=True)
        nw = (n + 31) // 32; pad = np.zeros(nw * 32, np.int64); pad[:n] = c
        cc = c[active].astype(float)
        ins = np.where(cc <= K, cc, K * (1 + np.log(np.maximum(cc, 1) / K))).sum()
        out.append(dict(rd=rd, r=r, act=int(active.sum()), calls=float(pad.reshape(-1, 32).max(1).sum()), hits=float(cc.sum()), ins=float(ins), awarps=int((pad.reshape(-1, 32) > 0).any(1).sum())))
        active = active & (c - 1 < K)
        if not active.any(): break
        r *= 2
    return out
if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'select'
    A = loadA(); fA = feats(100000, 1)
    T = np.array([A[i][0] for i in range(1, 6)]); calls_meas = np.array([A[i][3] for i in range(1, 6)]); cyc = np.array([A[i][2] for i in range(1, 6)])
    print('A per round: radius, active, calls model / measured, hits, inserts, active warps, measured span ms, callback cycles per call')
    for i, f in enumerate(fA): print(f['rd'], f['r'], f['act'], '%.0f / %.0f' % (f['calls'], calls_meas[i]), '%.0f' % f['hits'], '%.0f' % f['ins'], f['awarps'], T[i], '%.0f' % (cyc[i] / calls_meas[i]))
    rules = {'calls': np.array([f['calls'] for f in fA]), 'hits': np.array([f['hits'] for f in fA]), 'inserts': np.array([f['ins'] for f in fA])}
    best = None
    for nm, x in rules.items():
        tau = T.sum() / x.sum(); pred = tau * x; e = np.log(pred / T)
        print('%-8s tau %.4g ms per unit  per round log error %s  rms %.3f' % (nm, tau, ' '.join('%+.2f' % v for v in e), math.sqrt((e ** 2).mean())))
    if mode == 'predict':
        tau = T.sum() / rules['hits'].sum()
        print('\n# TrueKNN SECOND ATTEMPT: rule and per round predictions, committed BEFORE comparing with the 200,000 / 400,000 measurements (2026-11-16)\n')
        print('Label: second attempt. F156 (first attempt) stays as reported. Honest status: the F156 comparison tables were already seen, and one raw 200,000 point run file (notes/q20/tk/200000_1.txt, round spans and callback counts) was printed before the rule was chosen, so the held out sets are held out only in the sense that no number from them was used to build the rule (rule from the 100,000 point set A alone, three candidate rules compared on A, the one with the smallest in-sample rms chosen). Point files of F156 are gone, so the CPU features use regenerated uniform points (same distribution, fixed seeds); F156 predicted 5,879 warp calls in round 1 of A, this regeneration 5,792.\n')
        print('Rule (one): the launch time of a round is proportional to the number of candidates the intersection program sees, hits = sum over active queries of the exact Chebyshev box count (self included): T = %.4g ms x hits. That is %.0f SM cycles per candidate (142 SMs, 2.5 GHz). Equivalent per call statement: callback cost per warp call grows with the active lanes per call (hits per call), not with the call count; calls alone (F156 style, one pooled cost per call) gives rms log error 1.00 on A against 0.38 for hits and 0.60 for expected heap inserts. Active rule, stop rule, rounds: unchanged from F156.\n' % (tau, tau * SM * CLK * 1e-3))
        print('Calibration (in sample, shown for the shape only): log error of the per round time on A, rounds 1 to 5: %s (round 1 is the first launch, cold; round 5 has 2,700 active warps with about 2 active lanes)\n' % ' '.join('%+.2f' % v for v in np.log(tau * rules['hits'] / T)))
        for n, seed, nm in ((200000, 2, 'B'), (400000, 3, 'C')):
            f = feats(n, seed); print('## %d points (held out)\n' % n)
            print('| %s round | radius | active (predicted) | hits (predicted) | hits per active query | launch ms (predicted) |\n|---|---|---|---|---|---|' % nm); tot = 0
            for x in f:
                ms = tau * x['hits']; tot += ms
                print('| %d | %g | %d | %.0f | %.1f | %.4f |' % (x['rd'], x['r'], x['act'], x['hits'], x['hits'] / x['act'], ms))
            print('| total | | | | | %.4f |\n' % tot)
        print('What can fail: the cost per candidate differs between rounds by 2 times already on A (161 to 446 SM cycles per candidate by the callback probe), cold first launch rounds, heap insertion cost (F138) at larger sizes. Success rule (set before): total within a factor 1.5 of measured, per round log error reported, rounds equal. No refitting after the comparison.')
