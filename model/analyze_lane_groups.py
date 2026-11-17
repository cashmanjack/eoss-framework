"""Lane group results vs predictions (model/predictions_lane_groups.md). /usr/bin/python3 model/analyze_lane_groups.py notes/unit3g/g_summary_*.json"""
import json, sys
sys.path.insert(0, 'model')
import predict_lane_groups as P, audit_lane_sharing as L
res = json.load(open(sys.argv[1])); G = (1, 2, 4, 8, 16, 32)
name = lambda g: 'miss' if g == 1 else 'grp%d' % g
M = {(r['size'], r['rays'], r['w']): r for r in res}
print('RATE c (32 warps), measured | A | B | H     [rule: within 30 percent]')
ok = {'A': {}, 'B': {}, 'H': {}}
for n, d in L.SIZES.items():
    row = []
    for g in G:
        m = M[(n, name(g), 32)]['c']; a, b, h = P.A_rate(d, g, 32), P.B_rate(d, g), P.H(d['c_coh'], d['c_inc'], g)
        row.append((g, m, a, b, h))
    print('%8d | ' % n + ' | '.join('g%-2d %5.1f %4.0f %3.0f %3.0f' % r for r in row), '| coh(F197 rep ray) %.1f' % M[(n, 'coh', 32)]['c'])
    for k, ix in (('A', 2), ('B', 3), ('H', 4)):
        ok[k][n] = sum(abs(r[ix] / r[1] - 1) <= 0.30 for r in row if r[0] in (2, 4, 8, 16))
print('groups within 30 percent (of g = 2, 4, 8, 16):', ok)
print('\nratios to g = 1 (measured c(g) / c(1)); A and B predicted c(16) / c(1)')
for n, d in L.SIZES.items():
    c1 = M[(n, 'miss', 32)]['c']
    print('%8d | meas %s | A c16/c1 %.2f  B %.2f  H %.2f | meas c16/c1 %.2f, c32/c1 %.2f' % (n, ' '.join('%.2f' % (M[(n, name(g), 32)]['c'] / c1) for g in G), P.A_rate(d, 16, 32) / P.A_rate(d, 1, 32), P.B_rate(d, 16) / P.B_rate(d, 1), P.H(d['c_coh'], d['c_inc'], 16) / d['c_inc'], M[(n, 'grp16', 32)]['c'] / c1, M[(n, 'grp32', 32)]['c'] / c1))
print('\nONE WARP latency, measured | A | H')
for n, d in L.SIZES.items():
    print('%8d | ' % n + ' | '.join('g%-2d %6.0f %5.0f %5.0f' % (g, M[(n, name(g), 1)]['lat'], 32 * P.A_rate(d, g, 1), P.H(d['L_coh'], d['L_inc'], g)) for g in G), '| coh %.0f' % M[(n, 'coh', 1)]['lat'])
print('\nF197 controls in this session: g = 1 (new rays) vs F197 incoherent, coh (rep ray) vs F197 coherent: c, then one warp latency')
for n, d in L.SIZES.items():
    print('%8d | c inc %.1f vs %.1f, coh %.1f vs %.1f | lat inc %.0f vs %d, coh %.0f vs %d' % (n, M[(n, 'miss', 32)]['c'], d['c_inc'], M[(n, 'coh', 32)]['c'], d['c_coh'], M[(n, 'miss', 1)]['lat'], d['L_inc'], M[(n, 'coh', 1)]['lat'], d['L_coh']))
