"""Program arm analysis (F171): python3 model/program_arm.py notes/q30/prog_<ts>.json"""
import json, sys, statistics as st
R = [r for r in json.load(open(sys.argv[1])) if r['nq'] == 100000]
print('| arm | program | instance level | span us (median of 20) | cold us | calls per query | warp cycles mean |'); print('|---|---|---|---|---|---|---|')
for name, (p, i) in zip('ABCD', ((0, 0), (1, 0), (0, 1), (1, 1))):
    r = [x for x in R if x['prog'] == p and x['ias'] == i][0]; L = [x for x in r['launches'] if not x['cold']]
    print('| %s | %s | %s | %.2f | %.2f | %.3f | %.0f |' % (name, 'Mochi' if p else 'trivial', 'yes' if i else 'no', st.median(x['span_ns'] for x in L) / 1e3, r['launches'][0]['span_ns'] / 1e3, L[0]['calls'] / 100000, st.median(x['warp_cyc_mean'] for x in L)))
