"""Conc_2 analysis (Phase 10c). conc2_analyze.py DIR [DIR ...]: reads every conc_*.json (Conc_1 style rows) in the directories.
RT ratio = rays/us of the row over rays/us of the alone32 row of the same repetition file (same process group). Prints per scene and config: median ratio (min to max), compute GB/s with RT (alone),
compute with RT over alone, number of runs, wrong records. Compute GB/s = iterations per second * 196,608 bytes (384 threads x 32 loads x 16 B) of useful data per block iteration.
Clock group (clk_*.csv + rows with launch stamps t0, t1 in the raw per-run json): clk_summary(csv, rawdir)."""
import glob, json, os, statistics as st, sys, csv
import numpy as np
BYTES = 196608

def rows_of(dirs):
    out = []
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, 'conc_*.json'))):
            rows = json.load(open(f))
            for r in rows: r['file'] = os.path.basename(f)
            out += rows
    return out

def table(rows):
    alone = {(r['file'], r['scene'], r['rep']): r['rays_per_us'] for r in rows if r['config'].startswith('alone')}
    res = {}
    for r in rows:
        if r['config'].startswith('alone'): continue
        k = (r['scene'], r['config']); a = alone.get((r['file'], r['scene'], r['rep']))
        if a is None: continue
        d = res.setdefault(k, dict(ratio=[], gb=[], gba=[], cr=[], wrong=0))
        d['ratio'].append(r['rays_per_us'] / a); d['gb'].append(r['compute_with_rt'] * BYTES / 1e9); d['gba'].append(r['compute_alone'] * BYTES / 1e9)
        d['cr'].append(r['compute_with_rt'] / r['compute_alone']); d['wrong'] += r['wrong'] + r['unstored']
    return res

def show(res, scenes=('soup100000', 'tri1')):
    for sc in scenes:
        print('scene', sc)
        for (s, c), d in sorted(res.items(), key=lambda x: x[0]):
            if s != sc: continue
            print('  %-18s RT %.3f (%.3f to %.3f)  compute %5.1f GB/s (alone %5.1f, %.2f of alone)  n %d wrong %d' % (c, st.median(d['ratio']), min(d['ratio']), max(d['ratio']), st.median(d['gb']), st.median(d['gba']), st.median(d['cr']), len(d['ratio']), d['wrong']))

def clk_summary(csvf, rawdir):
    a = np.genfromtxt(csvf, delimiter=',', names=True)
    out = {}
    for f in sorted(glob.glob(os.path.join(rawdir, '*_r*_2*.json'))):
        if os.path.basename(f).startswith(('s0_', 's1_')): continue
        j = json.load(open(f))
        if 'launches' not in j: continue
        name = os.path.basename(f).split('_r')[0]
        for L in j['launches'][1:]:
            if 't0' not in L: continue
            m = (a['t'] >= L['t0']) & (a['t'] <= L['t1'])
            if m.sum() == 0: continue
            d = out.setdefault(name, dict(sm=[], mem=[], mw=[], n=0, ms=[]))
            d['sm'] += list(a['sm'][m]); d['mem'] += list(a['mem'][m]); d['mw'] += list(a['mw'][m] / 1000.0); d['n'] += int(m.sum()); d['ms'].append(L['ms'])
    return out

if __name__ == '__main__':
    show(table(rows_of(sys.argv[1:])))
