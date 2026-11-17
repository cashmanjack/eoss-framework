"""Rungholt residual against DRAM traffic (plan and pass rule: model/predictions_rungholt_dram.md, committed before this ran). CPU only, no refit.
    /usr/bin/python3 model/analyze_rungholt_dram.py"""
import json, numpy as np
from scipy.stats import spearmanr
P = json.load(open('notes/hs/predict_rungholt.json'))['feats']; M = {r['order']: r['ms'] for r in json.load(open('notes/hs/measure_rungholt.json'))['results']}
F = json.load(open('notes/hs/f123_predictions_sponza_conference_rungholt.json'))['rungholt']['f123']
NR = 8388608; BW = 1008e9
orders = ['oracle d1', 'oracle d2', 'oracle d3', 'oracle d4', 'oracle full']; allo = ['natural'] + orders + ['morton-o k5']
print('T1 bandwidth floor (D bytes per ray x 8,388,608 rays / 1008 GB/s) against measured ms')
for k in allo:
    fl = P[k]['D'] * NR / BW * 1e3; print('  %-12s D %6.1f B/ray  floor %.3f ms  measured %.3f ms  floor/measured %.2f' % (k, P[k]['D'], fl, M[k], fl / M[k]))
r = {k: np.log((M['natural'] / M[k]) / F[k]) for k in orders + ['morton-o k5']}
print('residual r = log(measured speedup / F123 predicted):', {k: round(v, 3) for k, v in r.items()})
reg = {'D (DRAM, node lines)': {k: np.log(P['natural']['D'] / P[k]['D']) for k in r}, 'F (L1 miss sectors)': {k: np.log(P['natural']['F'] / P[k]['F']) for k in r},
       'order index': {k: float(i + 1) for i, k in enumerate(orders)} | {'morton-o k5': None}}
for name, x in reg.items():
    xs = np.array([x[k] for k in orders]); ys = np.array([r[k] for k in orders]); b = (xs * ys).sum() / (xs * xs).sum()
    ins = np.sqrt(np.mean((ys - b * xs) ** 2)); loo = []
    for i in range(5):
        m = np.arange(5) != i; bi = (xs[m] * ys[m]).sum() / (xs[m] ** 2).sum(); loo.append(ys[i] - bi * xs[i])
    loo = float(np.sqrt(np.mean(np.square(loo)))); rho = spearmanr(xs, ys)[0]
    mo = (r['morton-o k5'] - b * x['morton-o k5']) if x['morton-o k5'] is not None else None
    print('%-22s x = %s  slope %.3f  in sample rms %.3f  leave one out rms %.3f  Spearman %.2f  Morton residual after %s (measured r %.3f)' % (name, np.round(xs, 3), b, ins, loo, rho, 'n/a' if mo is None else '%.3f' % mo, r['morton-o k5']))
print('rms of r itself (no explanation):', round(float(np.sqrt(np.mean([r[k] ** 2 for k in orders]))), 3))
