"""Omm_1 analysis: model/omm_analyze.py notes/omm1 [res dir]. Medians over the 3 outer repeats (each the median of 20 launches), spread = (max - min) / median of the 3 medians,
invocations per ray (counting launch), ratio to plain any hit A, time ratios, correctness against A (hit, primitive, t) from the .res dumps (repeat 1), ncu."""
import glob, json, os, re, statistics as st, sys
import numpy as np
d = sys.argv[1]; PFX = sys.argv[2] if len(sys.argv) > 2 else 'omm'; resd = '/dev/shm/om1/res'
rows = {}
for f in sorted(glob.glob(os.path.join(d, PFX + '_N*_r[1-9]_*.json'))):
    m = re.match(PFX + r'_N(\d+)_(\w+)_r(\d)_', os.path.basename(f)); N, cfg, rep = int(m[1]), m[2], int(m[3]); j = json.load(open(f))
    rows.setdefault((N, cfg), []).append((rep, j, os.path.basename(f)[:-5]))
def key(c): return (c[0], 'A0 A D0 D B2 B4 B6 B8 C2 C4 C6 C8'.split().index(c[1]))
res = {}
for (N, cfg), L in rows.items():
    meds = [st.median(j['ms']) for _, j, _ in L]; med = st.median(meds); spread = (max(meds) - min(meds)) / med; within = max((max(j['ms']) - min(j['ms'])) / st.median(j['ms']) for _, j, _ in L)
    j0 = L[0][1]; res[(N, cfg)] = dict(ms=med, spread=spread, within=within, calls=j0.get('calls_mean'), hit=j0.get('hit_frac'), gas=j0['gas_bytes'], omm=j0['omm_bytes'], unk=j0['unknown_frac'], hole=j0['hole_frac'], n=len(L), tag=L[0][2])
def load(tag):
    f = os.path.join(resd, tag + '.res')
    return np.fromfile(f, np.uint32).reshape(-1, 4) if os.path.exists(f) else None
print('N | cfg | ms (median of %d) | spread | within-run | calls/ray | calls vs A | time vs A | time vs D | hit frac | GAS MB | OMM KB | rays differing from A (hit, prim, t)' % 3)
for N in (100000, 1000000):
    A = res.get((N, 'A')); D = res.get((N, 'D')); ra = load(A['tag']) if A else None
    for c in sorted([k for k in res if k[0] == N], key=key):
        r = res[c]; x = ''
        if ra is not None:
            rr = load(r['tag'])
            if rr is not None:
                bad = (rr[:, 0] != ra[:, 0]) | ((rr[:, 0] == 1) & ((rr[:, 2] != ra[:, 2]) | (np.abs(rr[:, 1].view(np.float32) - ra[:, 1].view(np.float32)) > 1e-5 * np.abs(ra[:, 1].view(np.float32)))))
                x = '%d (%.3f%%)' % (bad.sum(), 100 * bad.mean())
        print('%d | %s | %.4f | %.1f%% | %.1f%% | %s | %s | %s | %s | %.4f | %.2f | %.2f | %s%s' % (N, c[1], r['ms'], 100 * r['spread'], 100 * r['within'], '%.4f' % r['calls'] if r['calls'] is not None else '-',
              '%.3f' % (r['calls'] / A['calls']) if A and r['calls'] is not None else '-', '%.3f' % (r['ms'] / A['ms']) if A else '-', '%.3f' % (r['ms'] / D['ms']) if D else '-', r['hit'], r['gas'] / 1e6, r['omm'] / 1e3, x, '' if r['spread'] < 0.03 else '  NOISY'))
    print('unknown fractions per pattern (B level 2,4,6,8):', {c[1]: [round(u, 3) for u in res[c]['unk']] for c in res if c[0] == N and c[1].startswith('B')}); print('hole fractions', res[(N, 'A')]['hole'] if (N, 'A') in res else '')
print('\nncu (per optixLaunch, 2,097,152 rays):')
for f in sorted(glob.glob(os.path.join(d, 'ncu_N*.ncu.csv'))):
    t = open(f).read().splitlines(); hdr = None
    for i, l in enumerate(t):
        if l.startswith('"ID"'): hdr = [c.strip('"') for c in l.split('","')]; vals = [c.strip('"') for c in t[i + 2].split('","')] if i + 2 < len(t) else None; break
    if hdr and vals:
        g = dict(zip(hdr, vals)); n = lambda k: float(g[k].replace(',', '')) if k in g else float('nan')
        print(os.path.basename(f).split('_2026')[0], 'warp inst %.4g thread inst %.4g sectors %.4g time_ns %.4g' % (n('smsp__inst_executed.sum'), n('smsp__thread_inst_executed.sum'), n('l1tex__t_sectors.sum'), n('gpu__time_duration.sum')))
