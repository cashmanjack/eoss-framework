"""Plateau (b): SM -> GPC map from the per SM clock64 counters.

    gpcmap.py RUN.txt...     RUN.txt = output of clkmap (smid c0 g0 c1 g1 rate)

The cycle counter read by clock64() is one counter per GPC: all TPCs of a GPC read the same value (within about 30
cycles of read skew); different GPCs differ by 1e6..1e9 cycles. Group TPCs by that value at the same instant.
"""
import sys
def groups(path):
    d = {}
    for l in open(path):
        r = l.split(); d.setdefault(int(r[0]) // 2, []).append(int(r[1]))
    t = sorted((sum(v) / len(v), k) for k, v in d.items()); gs = [[t[0][1]]]
    for (a, _), (b, k) in zip(t, t[1:]):
        if b - a > 1000: gs.append([])
        gs[-1].append(k)
    return sorted(sorted(g) for g in gs)
if __name__ == '__main__':
    res = [groups(p) for p in sys.argv[1:]]
    for g in res[0]: print('GPC TPCs', g, 'SMs', [x for t in g for x in (2 * t, 2 * t + 1)])
    print('groups equal in all runs:', all(r == res[0] for r in res), '| number of GPCs', len(res[0]), '| sizes', [len(g) for g in res[0]])
    print('every group = TPC index mod 12:', all(len({t % 12 for t in g}) == 1 for g in res[0]))
