"""Knee blind test results vs model/predictions_knee2.md.
 /usr/bin/python3 model/analyze_knee2.py notes/unit3g/k_summary_<F200>.json notes/unit3g/k2_summary_<TS>.json notes/unit3g/knee2_inputs.json"""
import json, sys
import numpy as np
G = (1, 2, 4, 8, 16, 32)
P = {}
for r in json.load(open(sys.argv[1])): P[(r['size'], r['w'], int(r['rays'][3:]), 'plain')] = r['c']
S = {}
for r in json.load(open(sys.argv[2])): P[(r['size'], r['w'], int(r['rays'][3:]), r['kind'])] = r['c']
INP = {int(k): v for k, v in json.load(open(sys.argv[3])).items()}
def knee(c):
    top, fl = c[0], c[-1]
    if top / fl < 1.5: return None
    mid = (top + fl) / 2
    for i in range(1, 6):
        if c[i] <= mid: return float(2 ** (np.log2(G[i - 1]) + (c[i - 1] - mid) / (c[i - 1] - c[i])))
ladder = lambda n, W: [P[(n, W, g, 'plain')] for g in G]
cp = lambda n, W, g: float(np.exp(np.interp(np.log2(g), np.log2(G), np.log(ladder(n, W)))))
print('(1) new sizes: knee (measured) vs K prediction 2.69 (30,000) / 2.81 (300,000); factor 2 rule')
pred = {30000: 2.69, 300000: 2.81}; ok1 = 0
for n in (30000, 300000):
    for W in (16, 32):
        c = ladder(n, W); k = knee(c); good = k is not None and 0.5 <= k / pred[n] <= 2; ok1 += good
        print('   %7d W%-2d c %s top/floor %.1f knee %s  %s' % (n, W, ' '.join('%.1f' % x for x in c), c[0] / c[-1], 'none' if k is None else '%.2f' % k, 'ok' if good else 'MISS'))
print('   held (>= 3 of 4):', ok1 >= 3, ok1)
print('(2) W = 12: top / floor per size (no knee if < 1.5; held if >= 4 of 6 without knee)')
nk = 0
for n in (1000, 10000, 30000, 100000, 300000, 1000000):
    c = ladder(n, 12); k = knee(c); nk += k is None
    print('   %7d c %s top/floor %.2f knee %s' % (n, ' '.join('%.1f' % x for x in c), c[0] / c[-1], 'none' if k is None else '%.2f' % k))
print('   no knee at', nk, 'of 6; held:', nk >= 4)
print('(3) shifted variant: c_shift against c_plain(g_eq) (K) and c_plain(g) (distinct rays); scored: 100,000 300,000 1,000,000 x g 4 8 16 x W 16 32 where the two differ by > 25 percent')
sc = ok_k = closer = 0
for n in (1000, 10000, 30000, 100000, 300000, 1000000):
    for W in (16, 32):
        for g in (2, 4, 8, 16, 32):
            d = [INP[n]['plain'][str(x)]['mean'] for x in G]; t = INP[n]['shifted'][str(g)]['mean']
            geq = float(2 ** np.interp(-np.log(t), -np.log(d), np.log2(G))); cs = P[(n, W, g, 'shift')]; ck, ca = cp(n, W, geq), cp(n, W, g)
            inf = abs(ck / ca - 1) > 0.25; scored = n >= 100000 and g in (4, 8, 16) and inf
            if scored: sc += 1; ok_k += abs(cs / ck - 1) <= 0.25; closer += abs(np.log(cs / ck)) < abs(np.log(cs / ca))
            print('   %7d W%-2d g=%-2d shifted %5.1f | K c_plain(%.2f)=%5.1f | rays c_plain(%d)=%5.1f | %s %s' % (n, W, g, cs, geq, ck, g, ca, 'informative' if inf else '', 'SCORED' if scored else ''))
print('   scored %d: within 25 percent of K %d, closer to K than to the distinct ray alternative %d; K held: %s' % (sc, ok_k, closer, sc > 0 and ok_k >= 0.7 * sc and closer > sc / 2))
