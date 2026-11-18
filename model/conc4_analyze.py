"""Conc_4 analysis (Phase 10e, F243 close out). conc4_analyze.py DIR [out.png]  (python3 with numpy; reads the latest DIR/c4_sweep_*.json and the matching c4clk csv).
r = RT rays/us with the stream over the preceding alone run (same repetition), medians over 3 repetitions; lambda = stream lines/s; latency = mean timed load latency (SM cycles) / sampled SM clock of the arm; O = lambda x latency (lines in flight), per streaming SM = O / 32 (sib) or / 64 (high).
Rule (predictions_phase10e.md): HOLDS if on sib (a) pooled Spearman(r, O) <= -0.9 and (b) every L2 arm (F <= 72) and DRAM arm (F >= 96 or dramfew) pair with O within 1.5x has r within 0.10."""
import csv, glob, json, os, statistics as st, sys
import numpy as np
def rank(x): return np.argsort(np.argsort(x)).astype(float)
def spearman(a, b): return float(np.corrcoef(rank(a), rank(b))[0, 1])
def main(d, png=None):
    jf = sorted(glob.glob(os.path.join(d, 'c4_sweep_*.json')))[-1]; rows = json.load(open(jf)); ck = jf.replace('c4_sweep_', 'c4clk_sweep_').replace('.json', '.csv')
    clk = {}
    for r in csv.DictReader(open(ck)): clk.setdefault(r['key'], []).append(float(r['sm']))
    bykey = {r['seq']: r for r in rows}; res = {}
    for r in rows:
        if not r['compute'] or not r['lat_cyc']: continue   # dramfew (type 11) has no latency sampling: not on the O axis
        al = bykey[r['seq'] - 1]; ratio = st.median(r['ms'][1:] and [1] ) if False else r['rays_per_us'] / al['rays_per_us']
        lam = st.median(r['lines_per_s']); lat = st.median(r['lat_cyc']); mhz = st.median(clk[r['key']]) if r['key'] in clk else 2730.0
        res.setdefault(r['config'], []).append((ratio, lam, lat / (mhz * 1e6), lat, mhz))
    out = []
    for name, v in res.items():
        pl = 'sib' if name.startswith('sib') else 'high'; ns = 32 if pl == 'sib' else 64
        r, lam, lt, lc, mh = [st.median(x[i] for x in v) for i in range(5)]
        F = 0 if 'dramfew' in name else int(name.split('l2fp')[1].split('_')[0][:-1] if False else ''.join(c for c in name.split('l2fp')[1].split('_')[0][:3] if c.isdigit()) if 'l2fp' in name else 0)
        thr = '_d150' in name; out.append(dict(name=name, pl=pl, F=F, thr=thr, r=r, lam=lam, lat_us=lt * 1e6, lat_cyc=lc, O=lam * lt, Os=lam * lt / ns, rmin=min(x[0] for x in v), rmax=max(x[0] for x in v), n=len(v)))
    out.sort(key=lambda o: (o['pl'], o['thr'], o['F'] if o['F'] else 999))
    print('%-22s %6s %8s %9s %9s %9s %8s %s' % ('arm', 'r', 'r range', 'G lines/s', 'lat cyc', 'lat us', 'O lines', 'O per SM'))
    for o in out: print('%-22s %6.3f %4.3f-%4.3f %9.2f %9.0f %9.2f %8.0f %8.1f' % (o['name'], o['r'], o['rmin'], o['rmax'], o['lam'] / 1e9, o['lat_cyc'], o['lat_us'], o['O'], o['Os']))
    for pl in ('sib', 'high'):
        s = [o for o in out if o['pl'] == pl]; print('%s: Spearman(r, O) = %.3f over %d arms' % (pl, spearman([o['r'] for o in s], [o['O'] for o in s]), len(s)))
    s = [o for o in out if o['pl'] == 'sib']; L2 = [o for o in s if o['F'] and o['F'] <= 72]; DR = [o for o in s if not o['F'] or o['F'] >= 96]; pairs = fail = 0
    for a in L2:
        for b in DR:
            if max(a['O'], b['O']) / min(a['O'], b['O']) < 1.5:
                pairs += 1; bad = abs(a['r'] - b['r']) > 0.10; fail += bad; print('pair %-20s r %.3f O %.0f | %-20s r %.3f O %.0f %s' % (a['name'], a['r'], a['O'], b['name'], b['r'], b['O'], 'FAIL' if bad else 'ok'))
    print('matched O pairs (within 1.5x): %d, outside 0.10: %d' % (pairs, fail))
    if png:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4.2))
        for pl, mk in (('sib', 'o'), ('high', 's')):
            for kind, col in (('L2 (F <= 72 MB)', 'tab:blue'), ('DRAM (F >= 96 MB, dramfew)', 'tab:red')):
                s = [o for o in out if o['pl'] == pl and ((o['F'] and o['F'] <= 72) if kind[0] == 'L' else (not o['F'] or o['F'] >= 96))]
                ax.scatter([o['O'] for o in s], [o['r'] for o in s], marker=mk, c=col, label='%s, %s' % (pl, kind))
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('stream lines in flight (lines per s x latency)'); ax.set_ylabel('RT rays/us with stream over alone'); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(png, dpi=130)
if __name__ == '__main__': main(*sys.argv[1:3])
