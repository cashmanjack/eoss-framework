"""Conc_1 analysis: model/conc_analyze.py notes/conc1/conc_main_<TS>.json   (the final json written at the end of the run; the .tmp is the same rows while running)
RT speed against the same RT alone (alone32 for G 32 configs, alone64 for the split configs) and compute speed against its own alone rate, medians over repetitions, spread = (max - min) / median."""
import json, statistics as st, sys
rows = json.load(open(sys.argv[1]))
by = {}
for r in rows: by.setdefault((r['scene'], r['config']), []).append(r)
for scene in ('soup100000', 'tri1'):
    print(scene)
    print('config | G | RT rays/us (median) | RT vs alone | RT spread | compute vs alone | compute spread | reps | wrong/unstored')
    base = {g: st.median(r['rays_per_us'] for r in by[(scene, 'alone%d' % g)]) for g in sorted({r['G'] for r in rows})}
    for (s, c), L in by.items():
        if s != scene: continue
        rate = [r['rays_per_us'] for r in L]; m = st.median(rate); g = L[0]['G']
        cr = [r['compute_with_rt'] / r['compute_alone'] for r in L if 'compute_with_rt' in r]
        print('%-10s | %d | %8.1f | %.3f | %.1f%% | %s | %s | %d | %d/%d' % (c, g, m, m / base[g], 100 * (max(rate) - min(rate)) / m, '%.3f' % st.median(cr) if cr else '-', '%.1f%%' % (100 * (max(cr) - min(cr)) / st.median(cr)) if cr else '-', len(L), sum(r['wrong'] for r in L), sum(r['unstored'] for r in L)))
