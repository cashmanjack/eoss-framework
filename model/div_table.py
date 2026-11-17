"""Div_1 table: model/div_table.py notes/div1/raw. Whole run totals per workload (all rounds / launches pooled, intersection callbacks), 3 repeats, and probe overhead (probe vs plain run time, medians)."""
import glob, os, re, sys, statistics as st
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import div_summ as D
d = sys.argv[1]
def runs(wl, mode='probe'):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, '%s_%s_*.bin' % (wl, mode)))):
        m = re.match(r'%s_%s_(\d+T\d+)_(\d)' % (wl, mode), os.path.basename(f)); out.setdefault(m[1] + m[2], []).append(f)
    return out
def tm(wl, mode):
    v = []
    for f in sorted(glob.glob(os.path.join(d, '%s_%s_*_?.txt' % (wl, mode)))):
        t = open(f).read()
        m = (re.findall(r'Total Search Time: (\d+)ms', t) or re.findall(r'time total search time: ([\d.]+) ms', t) or re.findall(r'KERNEL queries \d+ ms ([\d.]+)', t) or re.findall(r'Total time: ([\d.]+)', t))
        if wl.startswith('rtnn'): m = re.findall(r'KERNEL queries \d+ ms ([\d.]+)', t)
        if wl in ('mochi', 'knn', 'tk100k', 'tk400k'): m = []
        if wl == 'mochi-': m = re.findall(r'^([\d.]+)$', t, re.M)[:1]
        if m: v.append(float(m[-1]))
    return v
print('workload | repeats | callbacks | mean lanes/callback | share of callbacks <=8 lanes | share =32 | callbacks per handoff | time with <=8 lanes (share of warp time) | <=16 lanes | callback cycles share | spread of mean lanes, <=8 time (3 repeats) | time probe/plain')
for wl in ['mochi', 'knn', 'rtnn_r025', 'rtnn_r05', 'ark_r005', 'ark_r0112', 'tk100k', 'tk400k']:
    R = runs(wl); res = []
    for k, fl in R.items():
        if wl == 'mochi': fl = [f for f in fl if 'launch1' in f]
        a = np.concatenate([D.load(f, 0 if wl == 'knn' else None) for f in fl]); o, cb, iv = D.summ(a); res.append(o)
    if not res: continue
    g = lambda key: st.median(o[key] for o in res); sp = lambda key: 100 * (max(o[key] for o in res) - min(o[key] for o in res)) / g(key)
    tp, tl = tm(wl, 'probe'), tm(wl, 'plain')
    print('%s | %d | %.4g | %.2f | %.3f | %.3f | %.2f | %.3f | %.3f | %.3f | %.1f%%, %.1f%% | %s' % (wl, len(res), g('callbacks'), g('mean_lanes'), g('cb_le8'), g('cb_32'), g('cb_per_ho'), g('t_le8'), g('t_le16'), g('cb_cyc_share'), sp('mean_lanes'), sp('t_le8'),
          '%.3f (%s vs %s)' % (st.median(tp) / st.median(tl), ','.join('%g' % x for x in tp), ','.join('%g' % x for x in tl)) if tp and tl else 'n/a'))
