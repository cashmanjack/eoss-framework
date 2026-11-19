"""Scores part A of model/predictions_4090_batch7.md (L2 write vs read cost). python3 model/score_4090_batch7a.py [jsonl]"""
import json, glob, re, sys, statistics as st
f = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob('notes/l2rw/l2rw_*.jsonl'))[-1]
D = {}; N = {}
for l in open(f):
    r = json.loads(l); m = dict(x.split('=', 1) for x in re.findall(r'(\w+=(?:"[^"]*"|\S+))', r['line'])); k = (float(m['mb']), int(m['pat']), int(m['wq'])); D.setdefault(k, []).append(float(m['ns_per_sector']) / (2 if (k[1] == 2 and 'v' not in m) else 1))  # v1 files counted 32 sectors per lane32 access, it is 64
    if 'ncu' in r and r['ncu']: N[k] = r['ncu']
med = lambda k: st.median(D[k]); spread = lambda k: (max(D[k]) - min(D[k])) / med(k)
print(f, 'configs', len(D), 'max rep spread', round(max(spread(k) for k in D), 3))
PN = {0: 'coal', 1: 'lane4', 2: 'lane32'}
print('| MB | pattern | ns/sector wq 0,1,2,3,4 (write share 0, .25, .5, .75, 1) | TB/s wq 0 / 4 | c_w / c_r | linearity (wq 2 against mean of ends) | ncu hit rate min | write sector share (ncu) wq 4 |')
for mb in sorted({k[0] for k in D}):
    for p in (0, 1, 2):
        v = [med((mb, p, w)) for w in range(5)]; cw = v[4] / v[0]; lin = v[2] / ((v[0] + v[4]) / 2)
        hr = min(float(N[(mb, p, w)]['lts__t_sector_hit_rate.pct']) for w in range(5) if (mb, p, w) in N) if N else float('nan')
        n4 = N.get((mb, p, 4)); ws = float(n4['lts__t_sectors_op_write.sum'].replace(',', '')) / (float(n4['lts__t_sectors_op_write.sum'].replace(',', '')) + float(n4['lts__t_sectors_op_read.sum'].replace(',', ''))) if n4 else float('nan')
        print('| %g | %s | %s | %.2f / %.2f | %.2f | %.2f | %.1f | %.2f |' % (mb, PN[p], ' '.join('%.5f' % x for x in v), 0.032 / v[0], 0.032 / v[4], cw, lin, hr, ws))
print('\nfootprint spread (8 against 56 MB, ns per sector ratio) per pattern and wq:')
for p in (0, 1, 2): print(PN[p], ' '.join('%.2f' % (med((56.0, p, w)) / med((8.0, p, w))) for w in range(5)))
print('\nncu sectors per config (32 MB):')
for p in (0, 1, 2):
    for w in range(5):
        n = N.get((32.0, p, w))
        if n: print(PN[p], 'wq', w, 'read', n['lts__t_sectors_op_read.sum'], 'write', n['lts__t_sectors_op_write.sum'], 'hit', n['lts__t_sector_hit_rate.pct'], 'lts pct', n['lts__throughput.avg.pct_of_peak_sustained_elapsed'], 'dram bytes', n['dram__bytes.sum'])
