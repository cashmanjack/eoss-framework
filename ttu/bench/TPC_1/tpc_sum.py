"""Summarise tpc4 / tpc5 json: RT speed against the same build alone (split: alone71, shared: alone142), median (min..max) over repetitions, per scene.
    tpc_sum.py JSON [JSON...]      rows with scene, [build], config, rep, rays_per_us, compute_with_rt, compute_alone
"""
import json, sys, statistics as st
from collections import defaultdict
rows = []
for f in sys.argv[1:]:
    r = json.load(open(f)); b = f.split('tpc5_')[1].split('_')[0] if 'tpc5_' in f else None
    for x in r: x.setdefault('build', b); rows.append(x)
R = defaultdict(list); C = defaultdict(list); A = {}
for x in rows: R[(x['scene'], x['build'], x['config'])].append(x['rays_per_us']); C[(x['scene'], x['build'], x['config'])].append(x['compute_with_rt'] / x['compute_alone'] if x.get('compute_alone') else None)
for k in R:
    if k[2].startswith('alone'): A[k] = st.median(R[k])
ORDER = ['base', 'lean', 'lean_ld', 'lean_st']
for sc in sorted({k[0] for k in R}):
    print('== scene', sc)
    for b in sorted({k[1] for k in R if k[0] == sc}, key=lambda s: ORDER.index(s) if s in ORDER else 9):
        cf = [k[2] for k in R if k[0] == sc and k[1] == b]; seen = []
        print(' build %-8s alone142 %s alone71 %s rays/us' % (b, '%.0f' % A[(sc, b, 'alone142')] if (sc, b, 'alone142') in A else '-', '%.0f' % A[(sc, b, 'alone71')]))
        for c in cf:
            if c in seen or c.startswith('alone'): continue
            seen.append(c); ref = A[(sc, b, 'alone142' if c.startswith('shared') else 'alone71')]; v = [x / ref for x in R[(sc, b, c)]]; cc = [x for x in C[(sc, b, c)] if x]
            print('   %-18s RT %.2f (%.2f..%.2f)   compute %.2f' % (c, st.median(v), min(v), max(v), st.median(cc) if cc else float('nan')))
