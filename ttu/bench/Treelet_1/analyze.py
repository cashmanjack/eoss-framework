"""analyze.py RUNS_DIR MIN_TS : speed ratios per (set, K, WPS) over all valid main_*.json with timestamp >= MIN_TS (e.g. 20261124T101521).
Ratio = time(arm B) / time(arm A) = speed of A over B, per process (median of its 10 rounds), then median and range over processes."""
import glob, json, os, re, statistics as st, sys
d, mints = sys.argv[1], sys.argv[2]
G = {}
for f in sorted(glob.glob(d + '/main_*.json')):
    m = re.match(r'main_(\w+)_k(\d+)_w(\d+)_(\d+T\d+)\.json', os.path.basename(f))
    if not m or m.group(4) < mints: continue
    j = json.load(open(f)); G.setdefault((m.group(1), int(m.group(2)), int(m.group(3))), []).append(j)
PAIRS = [('grp_R', 'nat_R', 'grouping, root start (nat/grp)'), ('grp_T', 'nat_T', 'grouping, treelet start'), ('grp_T', 'grp_R', 'treelet start vs root start, grouped'),
         ('pg_R', 'grp_R', 'persistent machinery root: grp/pg'), ('pn_R', 'nat_R', 'machinery natural root: nat/pn'),
         ('pp_R', 'grp_R', 'PIN vs normal grouped, root (grp/pp)'), ('pp_R', 'pg_R', 'PIN vs unpinned queue, root (pg/pp)'),
         ('pp_T', 'grp_T', 'PIN vs normal grouped, treelet (grp/pp)'), ('pp_T', 'pg_T', 'PIN vs unpinned queue, treelet (pg/pp)')]
for k in sorted(G):
    js = G[k]; print('set %s K %d WPS %d: %d processes, %d rays, treelet GAS median %d B' % (k[0], k[1], k[2], len(js), js[0]['rays'], js[0]['treelet_gas_median']))
    tm = {a: [x['median_ms'] for j in js for x in j['arms'] if x['arm'] == a] for a in [x['arm'] for x in js[0]['arms']]}
    print('   median ms (min..max over processes): ' + ', '.join('%s %.3f (%.3f..%.3f)' % (a, st.median(v), min(v), max(v)) for a, v in tm.items()))
    for fast, slow, lab in PAIRS:
        if fast in tm and slow in tm:
            r = [s / f for s, f in zip(tm[slow], tm[fast])]; print('   %-42s %.3f (%.3f..%.3f)' % (lab, st.median(r), min(r), max(r)))
    bad = sum(x['bad'] + x['tbad'] for j in js for x in j['arms']); print('   wrong rays + t mismatches over all arms and processes:', bad,
          '| stolen share pp_T %.1f%% pp_R %.1f%%' % tuple(100. * st.median([x['stolen_chunks'] / x['chunks'] for j in js for x in j['arms'] if x['arm'] == a]) for a in ('pp_T', 'pp_R')))
