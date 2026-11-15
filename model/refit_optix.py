"""Refit the miss and locality coefficients of the performance model on OptiX data (2026-11-15), then test on scenes not used in the fit.
Cost of an order c = 1 / predict_sm(p, feats). Only a_S1, a_F, a_D32 (the L1 sector, L1 fill and DRAM terms) are refitted; every other coefficient stays as fitted on the RTX 6000 Ada.
Fit: least squares on log(predicted speedup over the natural order / measured speedup), coefficients >= 0.
Train: soup 300,000 random rays, bounce rays on the 100,000 soup. Test: soup 1,500,000 and Bistro (random rays), bounce rays on the 3 million soup. (RTNN has no replay inputs: not rechecked.)"""
import json, sys, numpy as np
from scipy.optimize import least_squares
sys.path.insert(0, 'model'); sys.path.insert(0, 'ttu/bench/Memory_4')
import performance_model as PM, run_model as RM
p0 = PM.fit(RM.memory4_rows())
def load_random(t):
    f = json.load(open('notes/q10/check/predict_%s.json' % t))['feats']; m = {r['order']: r['ms'] for r in json.load(open('notes/q10/check/measure_%s.json' % t))['results']}
    return t, f, m
def load_bounce(n):
    f = json.load(open('notes/io/predict/predict_bounce_%d.json' % n))['feats']; m = {r['order']: r['ms'] for r in json.load(open('notes/io/bounce/bounce_oracle_s%d.json' % n))['results']}
    return 'bounce%d' % n, f, m
sets = {s[0]: s for s in [load_random('s3e5'), load_random('s15e5'), load_random('bistro'), load_bounce(100000), load_bounce(3000000)]}
TRAIN, TEST = ['s3e5', 'bounce100000'], ['s15e5', 'bistro', 'bounce3000000']
def errs(coef, names):
    p = dict(p0); p.update(zip(('a_S1', 'a_F', 'a_D32'), coef)); out = {}
    for n in names:
        _, f, m = sets[n]; c = {k: 1 / PM.predict_sm(p, v) for k, v in f.items()}
        out[n] = [(k, c['natural'] / c[k], m['natural'] / m[k]) for k in f if k != 'natural' and k in m]
    return out
rms = lambda rows: float(np.sqrt(np.mean([np.log(a / b) ** 2 for _, a, b in rows])))
def report(tag, coef):
    e = errs(coef, list(sets)); print(tag, 'coef', np.round(coef, 4))
    for n in sets: print('  %-13s %s rms log error %.3f (%d orders)' % (n, 'train' if n in TRAIN else 'test ', rms(e[n]), len(e[n])))
    return e
c0 = [p0['a_S1'], p0['a_F'], p0['a_D32']]; report('before (Ada fit)', c0)
def res(c):
    e = errs(c, TRAIN); return [np.log(a / b) for n in TRAIN for _, a, b in e[n]]
r = least_squares(res, c0, bounds=(0, np.inf)); e = report('after (refit on OptiX train set)', r.x)
for n in ('s15e5', 'bistro'):
    print(n, [(k, round(a, 2), round(b, 2)) for k, a, b in e[n]])
