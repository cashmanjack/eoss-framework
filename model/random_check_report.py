"""Report of the random ray pre registered check (model/predictions_random_check_4070ti.md vs notes/q10/check/measure_*.json) plus a post hoc diagnostic.
Diagnostic (not a prediction): the model's time per ray is c = c_fixed + c_miss, where c_miss holds every term that contains S1, F or D (the order dependent ones). For each measured speedup find the
factor s on c_miss that reproduces it (c_miss -> s c_miss for every order, natural included). nan means no factor exists: the model's own structure cannot reach the measured speedup
(with s -> infinity the speedup tends to the ratio of the miss terms alone). Second diagnostic: the same predictions with the S1 term (L1 sectors requested, nearly independent of the order) set to zero."""
import json, os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, '..')
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, 'ttu', 'bench', 'Memory_4'))
import performance_model as PM, run_model as RM
from scipy.optimize import brentq
p = PM.fit(RM.memory4_rows())


def split(f):
    w = f['W']
    miss = p['a_S1'] * f['S1'] + p['a_F'] * f['F'] + p['a_D32'] * f['D'] / 32.0 + (p['b_F'] * f['F'] + p['b_D32'] * f['D'] / 32.0) / w
    fixed = p['a_q'] * f['q'] + p['b_q'] * f['q'] / w + p['c_qW'] * f['q'] * w / 16.0
    return fixed, miss


alle, spreads = [], []
print('%-9s %-12s %9s %9s %8s %8s %12s' % ('scene', 'order', 'predicted', 'measured', 'log err', 'factor s', 'a_S1 = 0'))
for tag in ('s3e5', 's15e5', 'bistro'):
    pr = json.load(open(os.path.join(ROOT, 'notes/q10/check/predict_%s.json' % tag))); me = json.load(open(os.path.join(ROOT, 'notes/q10/check/measure_%s.json' % tag)))
    nat = split(pr['feats']['natural']); errs = []
    for name, ps, ms in me['rows']:
        o = split(pr['feats'][name]); f = lambda s: (nat[0] + s * nat[1]) / (o[0] + s * o[1]) - ms
        try: s = brentq(f, 0.1, 50)
        except Exception: s = float('nan')
        q = dict(p, a_S1=0.0); n0 = 1 / PM.predict_sm(q, pr['feats']['natural']); o0 = 1 / PM.predict_sm(q, pr['feats'][name])
        print('%-9s %-12s %9.2f %9.2f %+8.3f %8.2f %12.2f' % (tag, name, ps, ms, np.log(ps / ms), s, n0 / o0)); errs.append(np.log(ps / ms)); alle.append(np.log(ps / ms)); spreads.append(np.log(ms))
    print('  %s: rms log error %.3f, rms of the measured log speedups %.3f, miss share of the natural order in the model %.2f' % (tag, np.sqrt(np.mean(np.square(errs))), np.sqrt(np.mean(np.square([np.log(r[2]) for r in me['rows']]))), nat[1] / (nat[0] + nat[1])))
for k in ('a_q', 'a_S1', 'a_F', 'a_D32', 'b_q', 'b_F', 'b_D32', 'c_qW'): print('  fitted %s = %.4g' % (k, p[k]))
print('all 18: rms log error %.3f, rms of the measured log speedups %.3f' % (np.sqrt(np.mean(np.square(alle))), np.sqrt(np.mean(np.square(spreads)))))
