"""analyze2.py DIR : per (tag, WPS) medians over processes (each process: median of its 10 rounds) and speed ratios. Ratio(A over B) = time B / time A, median and range over processes."""
import glob, json, os, re, statistics as st, sys
d = sys.argv[1]; G = {}
for f in sorted(glob.glob(d + '/run_*.json')):
    m = re.match(r'run_(.+)_(grid\d+|chunk\d+)_w(\d+)_(\d+T\d+)\.json', os.path.basename(f))
    if m.group(4) < (sys.argv[2] if len(sys.argv) > 2 else '0'): continue
    j = json.loads(open(f).read()); G.setdefault((m.group(1), int(m.group(3))), []).append(j)
PAIRS = [('pp_T', 'grp_T', 'pinned vs grouped normal (presorted)'), ('pp_T', 'pg_T', 'pinned vs unpinned queue (presorted)'), ('grp_R', 'nat_R', 'oracle grouping root start vs natural'), ('grp_T', 'grp_R', 'treelet start vs root start, grouped'),
         ('e_mo4', 'nat_R', 'E2E Morton4 vs natural'), ('e_mo5', 'nat_R', 'E2E Morton5 vs natural'), ('e_ogR', 'nat_R', 'E2E oracle grouped root vs natural'), ('e_ogT', 'nat_R', 'E2E oracle grouped T vs natural'), ('e_opT', 'nat_R', 'E2E oracle pinned T vs natural'),
         ('e_opT', 'e_mo5', 'E2E oracle pinned T vs Morton5'), ('e_opT', 'e_ogT', 'E2E oracle pinned T vs oracle grouped T'), ('e_igT', 'nat_R', 'E2E inspector grouped vs natural'), ('e_ipT', 'nat_R', 'E2E inspector pinned vs natural'),
         ('e_ipT', 'e_mo5', 'E2E inspector pinned vs Morton5'), ('e_ipT', 'e_igT', 'E2E inspector pinned vs inspector grouped')]
for k in sorted(G):
    js = G[k]; arms = {a['arm']: [] for a in js[0]['arms'] if 'arm' in a}
    for j in js:
        for a in j['arms']:
            if 'arm' in a: arms[a['arm']].append(a)
    print('== set %s WPS %d: %d processes, %d rays, treelets %d, treelet GAS median %d B, full GAS %d B' % (k[0], k[1], len(js), js[0]['rays'], js[0]['treelets'], js[0]['treelet_gas_median'], js[0]['full_gas_bytes']))
    for j in js[:1]:
        for a in j['arms']:
            if 'info' in a: print('   ', a)
    tm = {a: [x['median_ms'] for x in v] for a, v in arms.items()}
    print('   median ms (min..max over processes): ' + ', '.join('%s %.3f (%.3f..%.3f)' % (a, st.median(v), min(v), max(v)) for a, v in tm.items()))
    for f, s, lab in PAIRS:
        if f in tm and s in tm: r = [b / a for a, b in zip(tm[f], tm[s])]; print('   %-46s %.3f (%.3f..%.3f)' % (lab, st.median(r), min(r), max(r)))
    for a in ('e_igT', 'e_ipT'):
        if a in arms:
            sm = [x['stage_median_ms'] for x in arms[a]]; print('   %s stages (ms: inspector box trace, key+sort+gather(+cuts), treelet trace, verify trace, fallback, scatter): %s | flagged %.3f key accuracy %.3f' % (a, [round(st.median(c), 3) for c in zip(*sm)], st.median([x['flagged_frac'] for x in arms[a]]), st.median([x['key_accuracy'] for x in arms[a]])))
    bad = sum(x['bad'] + x['tbad'] for v in arms.values() for x in v); ties = sum(x.get('ties', 0) for v in arms.values() for x in v); print('   wrong rays + t mismatches (all arms, all processes):', bad, '| ties (other primitive, same t):', ties, '| stolen share pp_T %.1f%% e_opT %.1f%% e_ipT %.1f%%' % tuple(100. * st.median([x['stolen_chunks'] / x['chunks'] for x in arms[a]]) if a in arms else -1 for a in ('pp_T', 'e_opT', 'e_ipT')))
