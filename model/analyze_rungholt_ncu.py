"""Rungholt ncu DRAM bytes against the committed expectations (model/predictions_rungholt_ncu.md) and the F209 rule. CPU only.
    /usr/bin/python3 model/analyze_rungholt_ncu.py CSV_DIR   (reads ncu_<order>_<ts>.csv; F205 / F209 inputs from notes/hs and notes/hs2)"""
import csv, glob, io, json, os, sys
import numpy as np
from scipy.stats import spearmanr
D = sys.argv[1]; NR = 8388608; BW = 1008e9
names = ['natural', 'oracle d1', 'oracle d2', 'oracle d3', 'oracle d4', 'oracle full', 'morton-o k5']
P = json.load(open('notes/hs/predict_rungholt.json'))['feats']; M = {r['order']: r['ms'] for r in json.load(open('notes/hs/measure_rungholt.json'))['results']}
F = json.load(open('notes/hs/f123_predictions_sponza_conference_rungholt.json'))['rungholt']['f123']
res = {}
for n in names:
    f = sorted(glob.glob(os.path.join(D, 'ncu_%s_2*.csv' % n.replace(' ', '_'))))[-1]; txt = ''.join(l for l in open(f) if l.startswith('"'))
    res[n] = {r['Metric Name']: float(r['Metric Value'].replace(',', '')) for r in csv.DictReader(io.StringIO(txt))}
print('%-12s %9s %9s %8s %8s %8s %9s %9s %9s %8s' % ('order', 'rd MB', 'wr MB', 'dur ms', 'F205 ms', 'DRAM %pk', 'L2 hit %', 'feat MB', 'rd/feat', 'rd/(BW t)'))
for n in names:
    r = res[n]; t = r['gpu__time_duration.sum'] / 1e6; feat = P[n]['D'] * NR / 1e6
    print('%-12s %9.0f %9.0f %8.3f %8.3f %8.1f %9.1f %9.0f %9.2f %8.2f' % (n, r['dram__bytes_read.sum'] / 1e6, r['dram__bytes_write.sum'] / 1e6, t, M[n], r['dram__throughput.avg.pct_of_peak_sustained_elapsed'], r['lts__t_sector_hit_rate.pct'], feat, r['dram__bytes_read.sum'] / 1e6 / feat, r['dram__bytes_read.sum'] / (BW * t * 1e-3)))
orders = names[1:6]
rd = {n: res[n]['dram__bytes_read.sum'] for n in names}; wr = {n: res[n]['dram__bytes_write.sum'] for n in names}
print('\nE3: natural over oracle full DRAM read bytes %.2f; monotone fall with depth: %s' % (rd['natural'] / rd['oracle full'], all(rd[a] >= rd[b] for a, b in zip(orders, orders[1:]))))
print('ray input + output traffic is the same for every order: writes %s MB (rays read 268 MB: 32 B per ray)' % sorted({round(wr[n] / 1e6) for n in names}))
rr = {k: np.log((M['natural'] / M[k]) / F[k]) for k in orders + ['morton-o k5']}
print('F209 residual r:', {k: round(v, 3) for k, v in rr.items()})
def rule(name, x):
    xs = np.array([x[k] for k in orders]); ys = np.array([rr[k] for k in orders]); b = (xs * ys).sum() / (xs * xs).sum(); ins = np.sqrt(np.mean((ys - b * xs) ** 2)); loo = []
    for i in range(5):
        m = np.arange(5) != i; bi = (xs[m] * ys[m]).sum() / (xs[m] ** 2).sum(); loo.append(ys[i] - bi * xs[i])
    loo = float(np.sqrt(np.mean(np.square(loo)))); rho = spearmanr(xs, ys)[0]; mo = rr['morton-o k5'] - b * x['morton-o k5'] if x.get('morton-o k5') is not None else None
    print('%-34s x %s slope %.3f in sample %.3f LOO %.3f Spearman %.2f Morton resid %s' % (name, np.round(xs, 3), b, ins, loo, rho, 'n/a' if mo is None else '%.3f' % mo)); return loo, rho, mo
a = rule('E4 x = log(DRAM read nat / order)', {k: np.log(rd['natural'] / rd[k]) for k in orders + ['morton-o k5']})
b = rule('(extra) x = log(DRAM total nat/order)', {k: np.log((rd['natural'] + wr['natural']) / (rd[k] + wr[k])) for k in orders + ['morton-o k5']})
print('\nE4 pass: Spearman>=0.8 %s, LOO<=0.06 %s, LOO below order index 0.089 and miss 0.107 %s, Morton within 0.04 %s' % (a[1] >= 0.8, a[0] <= 0.06, a[0] < 0.089, abs(a[2]) <= 0.04))
print('(extra, not pre registered) fixed ray I/O traffic 538 MB at 1008 GB/s = %.3f ms; measured full order %.3f ms; DRAM time share of each order (all bytes / (BW t)):' % (538e6 / BW * 1e3, M['oracle full']),
      {n: round((rd[n] + wr[n]) / (BW * res[n]['gpu__time_duration.sum'] * 1e-9), 2) for n in names})
