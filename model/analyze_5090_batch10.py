"""Tables for model/results_5090_batch10.md from the raw files in notes/pc14_nn (read only). usage: python3 model/analyze_5090_batch10.py"""
import json, glob, re, statistics as st
D = 'notes/pc14_nn/'; med = st.median
def rj(prefix, arm, d, k):
    f = sorted(glob.glob('%s%s_%s_%s_k%s_q1000000_*.json' % (D, prefix, arm, d, k)))
    return json.load(open(f[-1]))
def hubs(d, k):
    f = sorted(glob.glob('%shubsjit_%s_k%s_*.txt' % (D, d, k)))[-1]; t = [int(x.split(',')[0]) / 1e6 for x in open(f) if re.match(r'^\d+, rep', x)]; return med(t), min(t), max(t)
def rtnn(d, k):
    rows = []
    for f in sorted(glob.glob('%srtnn_%s_k%s_*.txt' % (D, d, k))):
        s = open(f).read(); g = lambda p: float(re.search(p, s).group(1)); rows.append((g(r'time sort and/or partition queries: ([\d.]+)'), g(r'time create and upload geometry: ([\d.]+)'), g(r'time search compute: ([\d.]+)')))
    return [med(x[i] for x in rows) for i in range(3)], len(rows)
print('| data | K | r | ball mean (sample) | SPH cand | CUS cand | cand/ball SPH | cand/ball CUS | SPH/CUS cand |'); print('|---|---|---|---|---|---|---|---|---|')
BALL = {('uni1m', 5): 11.511, ('uni1m', 50): 96.183, ('clu1m', 5): 248.052, ('clu1m', 50): 1847.133}   # from check_nn.py output (20,000 sampled queries)
R = json.load(open(glob.glob(D + 'radii_*.json')[0]))
for d in ('uni1m', 'clu1m'):
    for k in (5, 50):
        s = rj('sorted', 'sph', d, k); c = rj('sorted', 'cus', d, k); b = BALL[(d, k)]
        print('| %s | %d | %.5f | %.1f | %.2f | %.2f | %.3f | %.3f | %.3f |' % (d, k, R[d]['r'][str(k)], b, s['mean_cand'], c['mean_cand'], s['mean_cand'] / b, c['mean_cand'] / b, s['mean_cand'] / c['mean_cand']))
print('\nTimes in ms, 1M queries, median of 5 (SPH, CUS: Morton sort + GAS build + search, device scope; unsorted = no sort, input order); hubs = CLOVER own span (median of 5, min to max); RTNN = sort + geometry + search of its own timers (median of 3), search alone in brackets')
print('| data | K | SPH sorted | CUS sorted | SPH / CUS | SPH unsorted | CUS unsorted | hubs | RTNN (sort+geom+search) [search] | SPH count only | CUS count only | SPH/CUS count only | ns per candidate per SM, SPH | CUS |'); print('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
for d in ('uni1m', 'clu1m'):
    for k in (5, 50):
        s = rj('sorted', 'sph', d, k); c = rj('sorted', 'cus', d, k); su = rj('full', 'sph', d, k); cu = rj('full', 'cus', d, k); so = rj('countonly', 'sph', d, k); co = rj('countonly', 'cus', d, k)
        h = hubs(d, k); (rs, rg, rq), n = rtnn(d, k); ts, tc = med(s['total_ms']), med(c['total_ms'])
        nsc = lambda t, j: t * 1e-3 * 170 / (j['mean_cand'] * 1e6) * 1e9
        print('| %s | %d | %.2f | %.2f | %.2f | %.2f | %.2f | %.1f (%.1f to %.1f) | %.1f [%.2f] | %.2f | %.2f | %.2f | %.1f | %.1f |' % (d, k, ts, tc, ts / tc, med(su['total_ms']), med(cu['total_ms']), h[0], h[1], h[2], rs + rg + rq, rq, med(so['total_ms']), med(co['total_ms']), med(so['total_ms']) / med(co['total_ms']), nsc(ts, s), nsc(tc, c)))
