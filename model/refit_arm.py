"""Refit arm analysis (F170): python3 model/refit_arm.py notes/q29/refit_<ts>.json   (median span of 20, calls per query, R/F)."""
import json, sys, statistics as st
R = [r for r in json.load(open(sys.argv[1])) if r['nq'] == 100000]; by = {}
for r in R:
    L = [x for x in r['launches'] if not x['cold']]; by.setdefault(r['step'], {})[r['tree']] = (r['f'], st.median(x['span_ns'] for x in L) / 1e3, L[0]['calls'] / r['nq'], st.median(x['warp_cyc_mean'] for x in L))
print('| step | f (squeeze factor or scramble cells) | R span us (calls/query) | U span us (calls) | F span us (calls) | R / F span | R warp cycles | F warp cycles |'); print('|---|---|---|---|---|---|---|---|')
for s, d in sorted(by.items()):
    print('| %d | %.2f | %.2f (%.3f) | %.2f (%.3f) | %.2f (%.3f) | %.2f | %.0f | %.0f |' % (s, d['R'][0], d['R'][1], d['R'][2], d['U'][1], d['U'][2], d['F'][1], d['F'][2], d['R'][1] / d['F'][1], d['R'][3], d['F'][3]))
