"""Phase 10i test 1 (b): is the RT ratio on the TPC siblings one function of the stream's miss line rate m = lines/s x (1 - L2 hit rate)? CPU only, existing Conc_4 data (F254) and the ncu hit rates of ttu/bench/Conc_5/l2probe.cu (raw notes/p10i/ncu_raw6000).
Rule (model/predictions_phase10i.md): HOLDS if pooled Spearman(r, m) <= -0.9 over the sib arms and every pair of arms with m within 1.5 times has r within 0.10 (m floored at 0.1 G lines/s).
/usr/bin/python3 model/phase10i_sibling.py [PNG]"""
import csv, glob, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import conc4_analyze as C4
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
hit = {}
for f in glob.glob(ROOT + '/notes/p10i/ncu_raw6000/l2probe_*_p0_*.csv'):
    fp = int(os.path.basename(f).split('_')[1])
    for r in csv.reader(open(f)):
        if len(r) >= 15 and r[-3] == 'lts__t_sector_hit_rate.pct': hit[fp] = float(r[-1]) / 100
print('ncu hit rate by footprint (alone):', {k: round(v, 3) for k, v in sorted(hit.items())})
out = [o for o in C4.main(os.path.join(ROOT, 'notes/conc4')) if o['pl'] == 'sib' and o['F'] in hit]
rows = [(o['name'], o['F'], o['thr'], o['r'], o['lam'] / 1e9, o['lam'] / 1e9 * (1 - hit[o['F']])) for o in out]
print('%-18s %5s %5s %7s %9s %9s' % ('arm', 'F', 'thr', 'r', 'G lines/s', 'm G/s'))
for x in rows: print('%-18s %5d %5s %7.3f %9.2f %9.2f' % x)
mm = [max(x[5], 0.1) for x in rows]; rr = [x[3] for x in rows]
rho = C4.spearman(rr, mm); print('pooled Spearman(r, m) = %.3f over %d arms' % (rho, len(rows)))
pairs = bad = 0
for i in range(len(rows)):
    for j in range(i + 1, len(rows)):
        if max(mm[i], mm[j]) / min(mm[i], mm[j]) < 1.5:
            pairs += 1; b = abs(rr[i] - rr[j]) > 0.10; bad += b
            if b: print('  pair differs: %s r %.3f m %.2f | %s r %.3f m %.2f' % (rows[i][0], rr[i], mm[i], rows[j][0], rr[j], mm[j]))
print('pairs with m within 1.5x: %d, r apart by more than 0.10: %d -> %s' % (pairs, bad, 'HOLDS' if rho <= -0.9 and bad == 0 else 'FAILS'))
