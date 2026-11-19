"""Scores model/predictions_5090_batch6.md (commit 9083baa, before the runs): ncu csv from notes/pc14_workloads (private raw). python3 model/score_5090_batch6.py"""
import csv, glob, io
def load(pat):
    f = sorted(glob.glob('notes/pc14_workloads/' + pat))[-1]; t = open(f).read(); rows = list(csv.DictReader(io.StringIO(t[t.find('"ID"'):]))); d = {}
    for r in rows: d.setdefault((int(r['ID']), r['Kernel Name'][:30]), {})[r['Metric Name']] = float(r['Metric Value'].replace(',', ''))
    return f, d
for p in ('ncu_rtnnC_r05_*.csv', 'ncu_rtnnC_r025_*.csv'):
    f, d = load(p); k = [v for (i, n), v in d.items() if n == 'optixLaunch'][0]; print(p, 'optixLaunch ms', k['gpu__time_duration.sum'] / 1e6)
for v in ('orig', 'reg'):
    f, d = load('ncu_knn_%s_*.csv' % v); print(f)
    L = [(i, n, x) for (i, n), x in sorted(d.items()) if n == 'optixLaunch']
    for r, (i, n, x) in enumerate(L, 1):
        print(v, 'round', r, 'ms %.4f' % (x['gpu__time_duration.sum'] / 1e6), 'grid', int(x['launch__grid_size']), 'L2 MB %.1f' % (x['lts__t_bytes.sum'] / 1e6), 'DRAM MB %.2f' % (x['dram__bytes.sum'] / 1e6), 'warp inst M %.2f' % (x['smsp__inst_executed.sum'] / 1e6),
              'lts %.1f l1tex %.1f dram %.1f sm %.1f' % tuple(x['%s__throughput.avg.pct_of_peak_sustained_elapsed' % a] for a in ('lts', 'l1tex', 'dram', 'sm')))
        print('     stalls pct:', ' '.join('%s %.2f' % (k.split('stalled_')[1].split('_per')[0], x[k]) for k in x if 'stalled' in k))
