"""analyze3.py DIR : per (tag, WPS) medians over processes of the single pass arms, and ratios to the best baseline (min of nat_R, e_mo4, e_mo5)."""
import glob, json, os, re, statistics as st, sys
G = {}
for f in sorted(glob.glob(sys.argv[1] + '/run_*.json')):
    try: j = json.load(open(f))
    except Exception: continue
    m = re.match(r'run_(.+?)_(grid\d+|chunk\d+)_w(\d+)_', os.path.basename(f))
    if any(a['R'] > 1 for a in j['arms']): continue
    G.setdefault((m.group(1), int(m.group(3))), []).append(j)
for k in sorted(G):
    js = G[k]; arms = {}
    for j in js:
        for a in j['arms']: arms.setdefault(a['arm'], []).append(a)
    med = {a: st.median(x['median_ms'] for x in v) for a, v in arms.items()}
    best = min(med.get(a, 1e9) for a in ('nat_R', 'e_mo4', 'e_mo5'))
    print('== %s WPS %d, %d processes' % (k[0], k[1], len(js)))
    for a, v in arms.items():
        bad = sum(x['bad'] + x['tbad'] for x in v); ov = sum((x.get('cnt') or [0] * 8)[4] for x in v)
        print('  %-8s %8.3f ms  vs best baseline %.2fx  wrong %d overflow %d %s' % (a, med[a], best / med[a], bad, ov, ('stages ' + str([round(st.median(c), 2) for c in zip(*[x['stage_median_ms'] for x in v])])) if 'stage_median_ms' in v[0] else ''))
    if 'f_pin' in arms:
        c = [x['cyc'] for x in arms['f_pin']]; t = [sum(z[:5]) for z in c]; m = st.median(range(len(c)))
        z = c[len(c) // 2]; tot = sum(z[:5]); print('  f_pin warp cycle shares: pop/probe %.2f trace %.2f inspector %.2f requeue %.2f idle %.2f' % tuple(x / tot for x in z[:5]), ' visits/ray %.2f' % (arms['f_pin'][0]['cnt'][3] / js[0]['rays']))
