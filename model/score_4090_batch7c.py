"""Part C of model/predictions_4090_batch7.md: speedup of the register heap from the byte mix (ncu2 sectors by operation) and the measured per sector costs (l2rw). python3 model/score_4090_batch7c.py [l2rw.jsonl] [CHIP]
Primary (committed method): T = R c_r + W c_w with lane4 at 32 MB; secondary: coal. Variant (post hoc, after the mixes were seen to be sub additive, labeled): time per sector interpolated on the write share from the measured curve."""
import csv, glob, io, json, re, sys, statistics as st
import numpy as np
f = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].endswith('.jsonl') else sorted(glob.glob('notes/l2rw/l2rw_*.jsonl'))[-1]
D = {}
for l in open(f):
    r = json.loads(l); m = dict(x.split('=', 1) for x in re.findall(r'(\w+=(?:"[^"]*"|\S+))', r['line'])); k = (float(m['mb']), int(m['pat']), int(m['wq'])); D.setdefault(k, []).append(float(m['ns_per_sector']) / (2 if (k[1] == 2 and 'v' not in m) else 1))
cost = lambda pat, w, mb=32.0: st.median(D[(mb, pat, w)])
def load(p, f):
    t = open(f).read(); rows = list(csv.DictReader(io.StringIO(t[t.find('"ID"'):]))); d = {}
    for r in rows: d.setdefault(int(r['ID']), {'n': r['Kernel Name'][:30]})[r['Metric Name']] = float(r['Metric Value'].replace(',', ''))
    return [v for k, v in sorted(d.items()) if v['n'] == 'optixLaunch'][:4]
def RW(x): return x['lts__t_sectors_op_read.sum'], x['lts__t_sectors_op_write.sum']
fo = sorted(glob.glob('notes/ncu4090/ncu2_knn_orig_2*.csv')); fr = sorted(glob.glob('notes/ncu4090/ncu2_knn_reg_2*.csv')); k1o = sorted(glob.glob('notes/ncu4090/ncu_knn_orig_2*.csv'))[:3]; k1r = sorted(glob.glob('notes/ncu4090/ncu_knn_reg_2*.csv'))
k1o = [f for f in k1o if 'smoke' not in f]; k1r = [f for f in k1r if 'smoke' not in f]
def curve(pat):  # time per sector as a function of write share w, piecewise linear on wq/4 (measured)
    xs = [q / 4 for q in range(5)]; ys = [cost(pat, q) for q in range(5)]; return lambda w: float(np.interp(w, xs, ys))
for rep in (0, 1):
    O, G = load('o', fo[rep]), load('r', fr[rep]); KO, KG = load('o', k1o[rep]), load('r', k1r[rep]); print('repetition', rep + 1)
    tot = {}
    for name, fn in (('lane4 linear (primary)', lambda R, W: R * cost(1, 0) + W * cost(1, 4)), ('coal linear (secondary)', lambda R, W: R * cost(0, 0) + W * cost(0, 4)),
                     ('lane4 curve (post hoc)', lambda R, W: (R + W) * curve(1)(W / (R + W))), ('byte ratio only', lambda R, W: R + W)):
        s = []; To = Tg = 0
        for r in range(4):
            ro, wo = RW(O[r]); rg, wg = RW(G[r]); a, b = fn(ro, wo), fn(rg, wg); s.append(a / b); To += a; Tg += b
        tot[name] = (s, To / Tg)
    ms = [KO[r]['gpu__time_duration.sum'] / KG[r]['gpu__time_duration.sum'] for r in range(4)]; mt = sum(x['gpu__time_duration.sum'] for x in KO) / sum(x['gpu__time_duration.sum'] for x in KG)
    print('| model | round 1 | round 2 | round 3 | round 4 | total | per round error vs measured (pct) | total error |\n|---|---|---|---|---|---|---|---|')
    print('| measured kernel speedup | %s | %.2f | | |' % (' | '.join('%.2f' % x for x in ms), mt))
    for n, (s, t) in tot.items(): print('| %s | %s | %.2f | %s | %+.0f%% |' % (n, ' | '.join('%.2f' % x for x in s), t, ' '.join('%+.0f' % (100 * (a / b - 1)) for a, b in zip(s, ms)), 100 * (t / mt - 1)))
    wo = [RW(O[r])[1] / sum(RW(O[r])) for r in range(4)]; wg = [RW(G[r])[1] / sum(RW(G[r])) for r in range(4)]; print('write share orig', ' '.join('%.2f' % x for x in wo), 'reg', ' '.join('%.2f' % x for x in wg))
