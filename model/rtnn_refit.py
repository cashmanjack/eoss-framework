"""Refit of the model's fixed part for custom primitives on RTNN radii (queue 12). CPU only.  /usr/bin/python3 model/rtnn_refit.py
Data: 12 configurations (100,000 points radius 2, 4, 6; 1,000,000 points radius 1, 2, 3; sorted and unsorted), RTX 4070 Ti, launch times by CUDA events (median of 5), replay inputs (prims in entered leaf slots, reuse
histograms, nodes variant and nodes+leaf variant), recorder (candidates, lane efficiency).
Cost per query in cycles on one SM: c = fixed + miss.  Before: fixed = A (a_q q + b_q q/W + c_qW q W/16) of the triangle fit (Finding 123), miss = share model (Finding 123), q = primitives in the entered leaf slots + 1.
After: fixed = c0 + cq q (two parameters, least squares on log of the measured cycles), miss unchanged (and, as a second variant, with a free miss scale m).
Splits: fit on two radii of one point set, test on the third (all three choices), plus fit on 100,000 points (3 radii), test on 1,000,000 points and the reverse."""
import json, os, re, statistics as st, sys, numpy as np
from scipy.optimize import least_squares
sys.path.insert(0, 'model'); sys.path.insert(0, 'ttu/bench/Memory_4')
import reuse_feature as RF
PAR = json.load(open('notes/q11/reuse_feature.json'))['share_k1']['par']
OUT = 'notes/q12/rtnn'; D = '/dev/shm/rt/dump/'
KK = 82.0 / (RF.hist('notes/q5/reuse/reuse_1e5_random.txt')[0] / (142 * 2 * 39 * 32))
CF = [(s, r, o) for s, rs in (('u1e5', (2, 4, 6)), ('u1e6', (1, 2, 3))) for r in rs for o in ('sorted', 'unsorted')]
def name(c): return '%s_r%d_%s' % c
def rhist(path, variant):
    raw = None
    for l in open(path):
        p = l.split()
        if p[0] == 'sm' and p[1] == variant and p[2] == 'raw': raw = int(p[3])
        if p[0] == 'sm' and p[1] == variant and p[2] == 'dedupe': return int(p[3]), int(p[4]), int(p[5]), np.array(p[6:], np.float64), raw
data = {}
for c in CF:
    n = name(c); N = 1e5 if c[0] == 'u1e5' else 1e6; rays = (1 if N == 1e5 else 2) * 39 * 60 * 32
    ms = st.median(float(re.search(r'KERNEL queries \d+ ms ([\d.]+)', open('%s/%s_plain%d.log' % (OUT, n, i)).read()).group(1)) for i in range(1, 6))
    rec = re.search(r'candidates_per_query ([\d.]+) lane_efficiency ([\d.]+)', open('%s/%s_rec.log' % (OUT, n)).read()); rep = np.fromfile(D + n + '/rep.bin', np.int32).reshape(-1, 8)
    miss = {}
    for v in ('nodes', 'nodes+leaf'):
        h = rhist('notes/q12/rtnn_replay/reuse_%s.txt' % n, v); S1 = KK * h[0] / rays
        miss[v] = PAR[1] * S1 * (1 - RF.share(h, PAR[2], PAR[3])) * (h[4] / h[0]) ** (-PAR[4])
    data[c] = dict(cyc=ms * 1e-3 * 2.5e9 * 60 / N, ms=ms, q=rep[:, 2].mean() + 1, cand=float(rec.group(1)), eff=float(rec.group(2)), miss=miss, nodes=rep[:, 0].mean(), leaves=rep[:, 1].mean())
def fixed_old(q): return PAR[0] * (RF.p0['a_q'] * q + RF.p0['b_q'] * q / 13 + RF.p0['c_qW'] * q * 13 / 16.0)
def pred(c, p, variant, kind):
    d = data[c]
    if kind == 'old': return fixed_old(d['q']) + d['miss'][variant]
    if kind == 'fixed': return p[0] + p[1] * d['q'] + d['miss'][variant]
    return p[0] + p[1] * d['q'] + p[2] * d['miss'][variant]
def ratio(c, p, variant, kind):
    return pred((c[0], c[1], 'unsorted'), p, variant, kind) / pred((c[0], c[1], 'sorted'), p, variant, kind)
def err(cfgs, p, variant, kind): return [np.log(pred(c, p, variant, kind) / data[c]['cyc']) for c in cfgs]
def rerr(rads, s, p, variant, kind): return [np.log(ratio((s, r, 's'), p, variant, kind) / (data[(s, r, 'unsorted')]['cyc'] / data[(s, r, 'sorted')]['cyc'])) for r in rads]
L = []
def P(*a):
    t = ' '.join(str(x) for x in a); print(t); L.append(t)
P('%-18s | cycles per query per SM measured | q | cand (hw) | lane eff | miss cycles (nodes / nodes+leaf)' % 'config')
for c in CF: d = data[c]; P('%-18s | %7.0f | %6.1f | %6.1f | %.3f | %6.0f / %6.0f' % (name(c), d['cyc'], d['q'], d['cand'], d['eff'], d['miss']['nodes'], d['miss']['nodes+leaf']))
rms = lambda x: float(np.sqrt(np.mean(np.square(x))))
splits = [(s, tr, te) for s, rs in (('u1e5', (2, 4, 6)), ('u1e6', (1, 2, 3))) for te in rs for tr in [tuple(r for r in rs if r != te)]]
res = {}
for variant in ('nodes', 'nodes+leaf'):
    for kind, npar, x0 in (('fixed', 2, [100, 10]), ('fixed+m', 3, [100, 10, 1.0])):
        rows = []
        for s, tr, te in splits:
            trc = [(s, r, o) for r in tr for o in ('sorted', 'unsorted')]; tec = [(s, te, o) for o in ('sorted', 'unsorted')]
            f = least_squares(lambda p: err(trc, p, variant, kind), x0, bounds=(0, np.inf)).x
            rows.append((s, te, f, rms(err(tec, f, variant, 'fixed' if kind == 'fixed' else kind)), rms(err(tec, None, variant, 'old')), abs(rerr([te], s, f, variant, kind)[0]), abs(rerr([te], s, None, variant, 'old')[0])))
        for (s, tr_, te) in [(a, b, c) for a, b, c in [('u1e5', None, 'u1e6'), ('u1e6', None, 'u1e5')]]:
            trc = [c for c in CF if c[0] == s]; tec = [c for c in CF if c[0] == te]
            f = least_squares(lambda p: err(trc, p, variant, kind), x0, bounds=(0, np.inf)).x
            rows.append((s + '->' + te, 'all', f, rms(err(tec, f, variant, kind)), rms(err(tec, None, variant, 'old')), rms([abs(rerr([r], te, f, variant, kind)[0]) for r in ((2, 4, 6) if te == 'u1e5' else (1, 2, 3))]), rms([abs(rerr([r], te, None, variant, 'old')[0]) for r in ((2, 4, 6) if te == 'u1e5' else (1, 2, 3))])))
        P('\nvariant %s, %s: held out rms log error of the absolute cycles (before = original fixed part, after = refit), and |log error| of the unsorted over sorted ratio' % (variant, kind))
        for s, te, f, a, b, ra, rb in rows: P('  %-12s test %-3s  fit %s | cycles before %.3f after %.3f | ratio before %.3f after %.3f' % (s, te, np.round(f, 2), b, a, rb, ra))
        res[(variant, kind)] = rows
        P('  mean over the 6 radius splits: cycles before %.3f after %.3f; ratio before %.3f after %.3f; the two size transfers: after %.3f %.3f' % (np.mean([r[4] for r in res[(variant, kind)][:6]]), np.mean([r[3] for r in res[(variant, kind)][:6]]), np.mean([r[6] for r in res[(variant, kind)][:6]]), np.mean([r[5] for r in res[(variant, kind)][:6]]), res[(variant, kind)][6][3], res[(variant, kind)][7][3]))
open('notes/q12/rtnn_refit.txt', 'w').write('\n'.join(L) + '\n')
