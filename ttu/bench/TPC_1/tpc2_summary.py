"""Summarize tpc2_run.py output: per scene and config, across repetitions: RT rate (median, min, max), RT speed against the same repetition's RT alone reference
(alone71 for split, alone142 for shared; median, min, max over the repetitions) and compute speed against compute alone (median, min..max).

    tpc2_summary.py tpc2_GROUP.json [...]
"""
import json, statistics as st, sys
rows = []
for f in sys.argv[1:]: rows += json.load(open(f))
by = {}
for r in rows: by.setdefault((r['scene'], r['config']), {})[r['rep']] = r
ref = lambda sc, rep, cfg: by[(sc, 'alone71' if cfg.startswith('split') else 'alone142')][rep]['rays_per_us']
def fmt(v): return '%.2f (%.2f..%.2f)' % (st.median(v), min(v), max(v))
for sc in ('soup100000', 'tri1'):
    print('scene', sc)
    print('%-26s %-22s %-20s %-20s %s' % ('config', 'RT rays/us med (min..max)', 'RT / alone', 'compute / alone', 'reps'))
    for (s, cfg), d in by.items():
        if s != sc: continue
        reps = sorted(d); rt = [d[k]['rays_per_us'] for k in reps]
        rr = [d[k]['rays_per_us'] / ref(sc, k, cfg) for k in reps]
        cc = [d[k]['compute_with_rt'] / d[k]['compute_alone'] for k in reps if 'compute_with_rt' in d[k]]
        print('%-26s %-22s %-20s %-20s %d' % (cfg, '%.0f (%.0f..%.0f)' % (st.median(rt), min(rt), max(rt)), fmt(rr), fmt(cc) if cc else '-', len(reps)))
