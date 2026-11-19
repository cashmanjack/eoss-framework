"""Scores model/predictions_5090_batch8.md (commit a7009db, before the run): tethys's L2rw benchmark on the 5090. python3 model/score_5090_batch8.py (reads notes/pc14_core/l2rw/*.jsonl, private raw)"""
import json, glob, re, statistics as st
f = sorted(glob.glob('notes/pc14_core/l2rw/l2rw_*.jsonl'))[-1]; R = [json.loads(l) for l in open(f)]
assert all(r['rc'] == 0 for r in R), 'a run failed'
D = {}; N = {}
for r in R:
    kv = dict(re.findall(r'(\w+)=("[^"]*"|\S+)', r['line'])); k = (float(kv['mb']), int(kv['pat']), int(kv['wq']))
    assert kv['err'] == '0'; D.setdefault(k, []).append((float(kv['ns_per_sector']), float(kv['TBps'])))
    if r.get('ncu'): N[k] = r['ncu']
med = lambda k, i: st.median(x[i] for x in D[k])
PN = ['coalesced 128 B', 'lane4 (4 B writes)', 'lane32 (32 B)']
print('runs', len(R), f)
print('| footprint MB | pattern | TB/s at write quarters 0 / 1 / 2 / 3 / 4 | ns per sector (x1000) 0 / 4 | cost ratio c_w / c_r | wq 2 against the linear mean of the ends | rep spread (max/min TBps, wq 0 and 4) |\n|---|---|---|---|---|---|---|')
for mb in (8, 32, 80):
    for p in (0, 1, 2):
        t = [med((mb, p, w), 1) for w in range(5)]; n = [med((mb, p, w), 0) for w in range(5)]
        sp = [max(x[1] for x in D[(mb, p, w)]) / min(x[1] for x in D[(mb, p, w)]) for w in (0, 4)]
        print('| %g | %s | %s | %.2f / %.2f | %.2f | %.2f | %.3f, %.3f |' % (mb, PN[p], ' / '.join('%.2f' % x for x in t), n[0] * 1e3, n[4] * 1e3, n[4] / n[0], n[2] / ((n[0] + n[4]) / 2), sp[0], sp[1]))
print('\nncu (repetition 0): hit rate pct, lts pct of peak, DRAM MB, read / write sectors M')
for mb in (8, 32, 80):
    for p in (0, 1, 2):
        row = []
        for w in (0, 4):
            x = N.get((mb, p, w))
            if x: row.append('wq%d hit %s lts %s dram %.1f rd %.0f wr %.0f' % (w, x['lts__t_sector_hit_rate.pct'], x['lts__throughput.avg.pct_of_peak_sustained_elapsed'], float(x['dram__bytes.sum'].replace(',', '')) / 1e6 if 'dram__bytes.sum' in x else -1, float(x['lts__t_sectors_op_read.sum'].replace(',', '')) / 1e6, float(x['lts__t_sectors_op_write.sum'].replace(',', '')) / 1e6))
        print(mb, PN[p], ' | '.join(row))

# F425 method on the 5090 with the 5090's own l2rw costs: time = read sectors x c_r + write sectors x c_w (ncu sector counts by operation, results of model/score_5090_batch7a.py inputs)
import csv, io
def load(fn):
    t = open(fn).read(); rows = list(csv.DictReader(io.StringIO(t[t.find('"ID"'):]))); d = {}
    for r in rows: d.setdefault(int(r['ID']), {'n': r['Kernel Name'][:30]})[r['Metric Name']] = float(r['Metric Value'].replace(',', ''))
    return [v for k, v in sorted(d.items()) if v['n'] == 'optixLaunch'][:4]
O = load(sorted(glob.glob('notes/pc14_workloads/ncu_knn2_orig_2*.csv'))[-1]); G = load(sorted(glob.glob('notes/pc14_workloads/ncu_knn2_reg_2*.csv'))[-1])
meas = [2.99, 3.34, 4.00, 2.98]; mtot = 4.5603 / 1.4364
for name, p in (('lane32 costs (32 MB)', 2), ('lane4 costs (32 MB)', 1), ('coalesced costs (32 MB)', 0)):
    cr, cw = med((32, p, 0), 0), med((32, p, 4), 0); pr = []; to = tg = 0
    for o, g in zip(O, G):
        a = o['lts__t_sectors_op_read.sum'] * cr + o['lts__t_sectors_op_write.sum'] * cw; b = g['lts__t_sectors_op_read.sum'] * cr + g['lts__t_sectors_op_write.sum'] * cw; pr.append(a / b); to += a; tg += b
    print('%s: c_w/c_r %.2f; predicted speedup per round %s total %.2f; measured %s total %.2f; error per round %s' % (name, cw / cr, ' '.join('%.2f' % x for x in pr), to / tg, ' '.join('%.2f' % x for x in meas), mtot, ' '.join('%+.0f%%' % (100 * (x / m - 1)) for x, m in zip(pr, meas))))
sr = [(o['lts__t_sectors_op_read.sum'] + o['lts__t_sectors_op_write.sum']) / (g['lts__t_sectors_op_read.sum'] + g['lts__t_sectors_op_write.sum']) for o, g in zip(O, G)]
print('unweighted sector ratio per round', ' '.join('%.2f' % x for x in sr), 'errors', ' '.join('%+.0f%%' % (100 * (x / m - 1)) for x, m in zip(sr, meas)))
