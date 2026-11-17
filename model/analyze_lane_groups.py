"""Lane group results vs predictions (model/predictions_lane_groups.md and its addendum). From the repo root:
    /usr/bin/python3 model/analyze_lane_groups.py A notes/unit3g/g_summary_<TS>.json      design A (first rays of each file, counts not fixed)
    /usr/bin/python3 model/analyze_lane_groups.py B notes/unit3g/g_summaryB_<TS>.json     design B (32 rays of one counts class)
    /usr/bin/python3 model/analyze_lane_groups.py slow                                    one warp latency of design A against F67 of the slowest lane (needs /dev/shm/t9u/cand_N.out)"""
import json, sys
sys.path.insert(0, 'model')
import numpy as np
import predict_lane_groups as P, audit_lane_sharing as L
G = (1, 2, 4, 8, 16, 32)
mode = sys.argv[1]
if mode == 'slow':
    f67 = lambda c: 934 + 343 * c[..., 0] + 113 * c[..., 1] + 52 * c[..., 2]; meas = {1000: 3944, 10000: 11943, 100000: 21027, 1000000: 17853}
    for n in meas:
        r = np.fromfile('/dev/shm/t9u/cand_%d.out' % n, np.int32).reshape(-1, 8); c = r[np.where(r[:, 5] < 0)[0][:1024]][:32, :3].astype(float)
        print(n, 'measured %d | F67 mean counts %.0f | F67 slowest lane %.0f | meas/mean %.2f meas/slowest %.2f' % (meas[n], f67(c.mean(0)), f67(c).max(), meas[n] / f67(c.mean(0)), meas[n] / f67(c).max()))
    sys.exit()
res = json.load(open(sys.argv[2])); M = {(r['size'], r['rays'], r['w']): r for r in res}
nm = (lambda g: 'miss' if g == 1 else 'grp%d' % g) if mode == 'A' else (lambda g: 'cls%d' % g)
print('DESIGN %s, rate c (32 warps) per g: measured | model A | model B | H (F197 ends) | H (own ends, post hoc)' % mode)
ok = {k: {} for k in ('A', 'B', 'H', 'Hown')}
for n, d in L.SIZES.items():
    c1, c32 = M[(n, nm(1), 32)]['c'], M[(n, nm(32), 32)]['c']; row = []
    for g in G:
        m = M[(n, nm(g), 32)]['c']; row.append((g, m, P.A_rate(d, g, 32), P.B_rate(d, g), P.H(d['c_coh'], d['c_inc'], g), P.H(c32, c1, g)))
    print('%8d | ' % n + ' | '.join('g%-2d %5.1f %4.0f %3.0f %3.0f %3.0f' % r for r in row))
    for k, ix in (('A', 2), ('B', 3), ('H', 4), ('Hown', 5)): ok[k][n] = sum(abs(r[ix] / r[1] - 1) <= 0.30 for r in row if r[0] in (2, 4, 8, 16))
print('groups of g = 2, 4, 8, 16 within 30 percent, per size:', ok)
print('measured c(g) / c(1) [g = 1 2 4 8 16 32] | predicted c(16) / c(1): A, B, H | rule 4: c(16) / c(32)')
for n, d in L.SIZES.items():
    c1 = M[(n, nm(1), 32)]['c']
    print('%8d | %s | %.2f %.2f %.2f | %.2f' % (n, ' '.join('%.2f' % (M[(n, nm(g), 32)]['c'] / c1) for g in G), P.A_rate(d, 16, 32) / P.A_rate(d, 1, 32), P.B_rate(d, 16) / P.B_rate(d, 1), P.H(d['c_coh'], d['c_inc'], 16) / d['c_inc'], M[(n, nm(16), 32)]['c'] / M[(n, nm(32), 32)]['c']))
print('one warp latency per g (measured) and lat(g) / lat(1)')
for n, d in L.SIZES.items():
    l1 = M[(n, nm(1), 1)]['lat']; print('%8d | %s | %s' % (n, ' '.join('%5.0f' % M[(n, nm(g), 1)]['lat'] for g in G), ' '.join('%.2f' % (M[(n, nm(g), 1)]['lat'] / l1) for g in G)))
