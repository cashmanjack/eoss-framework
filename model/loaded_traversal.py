"""Load_1 analysis (F169): python3 model/loaded_traversal.py notes/q28/load1_<ts>.json  (prints the markdown table and the decision rule inputs)."""
import json, sys, statistics as st
import numpy as np
R = json.load(open(sys.argv[1])); SM = 142
def med(v): return st.median(v)
for order in (0, 1):
    rows = [r for r in R if r['order'] == order]; tab = []
    # clock (GHz): warp cycles over warp ns at the largest launch (ns resolution 1024 so only big launches)
    big = rows[-1]; f = np.mean([x['warp_cyc_mean'] / x['warp_ns_mean'] for x in big['launches'] if not x['cold']])
    print('\norder', 'lattice' if order == 0 else 'random', 'SM clock from the largest launch %.3f GHz' % f)
    print('| NQ | span us (median of 20) | cold us | warp cycles mean (median of 20) | warps/SM | W in flight | c cycles per query per SM | queries per ns | calls per query |'); print('|---|---|---|---|---|---|---|---|---|')
    for r in rows:
        L = [x for x in r['launches'] if not x['cold']]; S = med([x['span_ns'] for x in L]); cyc = med([x['warp_cyc_mean'] for x in L]); nw = r['nq'] / 32
        W = nw * cyc / (SM * S * f); c = S * f * SM / r['nq']
        tab.append((r['nq'], S, cyc, c)); print('| %d | %.2f | %.2f | %.0f | %.2f | %.2f | %.1f | %.2f | %.3f |' % (r['nq'], S / 1e3, r['launches'][0]['span_ns'] / 1e3, cyc, nw / SM, W, c, r['nq'] / S, L[0]['calls'] / r['nq']))
    big = [t for t in tab if t[0] >= 4544]; X = np.array([t[0] for t in big], float); Y = np.array([t[1] for t in big])
    b, a = np.polyfit(X, Y, 1); s100 = [t[1] for t in tab if t[0] == 100000][0]
    L1 = [t[2] for t in tab if t[0] == 4544][0]; L100 = [t[2] for t in tab if t[0] == 100000][0]; L8 = [t[2] for t in tab if t[0] == 36352][0]
    cs = [t[3] for t in tab if t[0] >= 18176]; c4544 = [t[3] for t in tab if t[0] == 4544][0]; c100 = [t[3] for t in tab if t[0] == 100000][0]
    print('fit S = a + b NQ (NQ >= 4544): a %.2f us, b %.4f ns per query (%.2f queries per ns); intercept share a / S(100,000) = %.2f' % (a / 1e3, b, 1 / b, a / s100))
    print('L(100,000) / L(4,544) = %.2f; L(36,352; 8 warps per SM) / L(4,544) = %.2f; c max/min over NQ >= 18,176 = %.2f; c(4,544) / c(100,000) = %.1f' % (L100 / L1, L8 / L1, max(cs) / min(cs), c4544 / c100))
