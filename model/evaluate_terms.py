"""Checks of the terms added to the performance model on 2026-11-15 against measured data. CPU only.  /usr/bin/python3 model/evaluate_terms.py [OUT.md]"""
import json, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import performance_model as M
ROOT = os.path.join(os.path.dirname(__file__), '..')
out = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); out.append(s)

# ---- (1) TPC / short query term: set on the 4070 Ti (Finding 105), tested on the RTX 6000 Ada (Throughput_6, never used to set it) and the other way round
P('## Short query (TPC unit capacity) term')
rep = open(os.path.join(ROOT, 'notes/setup3/runs/throughput6/report.txt')).read().splitlines()
def tri(sc):
    d = {}
    for l in rep:
        m = re.match(r'%s\s*\|\s*(\d+)\s*\|\s*([\d.]+)\s+[\d.]+\s*\|\s*([\d.]+)' % sc, l)
        if m: d[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
    return d
for sc, long_q in (('tri1', False), ('par64', True)):
    d = tri(sc); x1 = d[1][0]
    # 4070 Ti ratio U / x1 from Finding 105 (41.3 / 30.0 at 60 and 30 SMs, short queries): 1.377; long queries: 1.04
    ratio4070 = 41.3 / 30.0
    U = ratio4070 * x1
    rows = []
    for G in (32, 71, 100, 120, 142):
        meas = d[G][1]; pred = M.tpc_aggregate(x1, G, 71, U, long_q); rows.append((G, pred, meas, pred / meas - 1))
    P('RTX 6000 Ada, %s, one SM %.4f rays/cycle, U set from the 4070 Ti ratio (%s): ' % (sc, x1, '1.04 x one SM' if long_q else '%.3f x one SM' % ratio4070) + ', '.join('G %d pred %.3f meas %.3f (%+.0f%%)' % (g, p, m, e * 100) for g, p, m, e in rows))
    if not long_q:
        U2 = d[142][1] / 71
        P('  triton alone: U = %.4f per TPC = %.3f x one SM (4070 Ti: %.3f x): unit capacity ratio differs by %.0f%% between the two chips' % (U2, U2 / x1, ratio4070, (U2 / x1 / ratio4070 - 1) * 100))
# 4070 Ti side: x1 = 0.1 and measured 30.0 / 41.3 (short), 29.8 / 31.1 (long), U set from triton's 1.527
x1 = 0.1; d = tri('tri1'); r_tri = d[142][1] / d[71][1]
for sc, meas, long_q in (('tri1', (30.0, 41.3), False), ('par64', (29.8, 31.1), True)):
    U = r_tri * x1
    p30 = M.tpc_aggregate(x1, 30, 30, U, long_q) / x1; p60 = M.tpc_aggregate(x1, 60, 30, U, long_q) / x1
    P('4070 Ti, %s, U set from triton (%.3f x one SM): G 30 pred %.1f meas %.1f, G 60 pred %.1f meas %.1f (%+.0f%%)' % (sc, r_tri, p30, meas[0], p60, meas[1], (p60 / meas[1] - 1) * 100))

# ---- (2) payload and attribute term: fit k on the triangle runs, test on custom primitives (not used for k or for the register formula)
P('\n## Payload / attribute term (OptiX kernel on the 4070 Ti, Finding 112)')
rows = json.load(open(os.path.join(ROOT, 'notes/io/payload/payload.json')))
def get(p, a, c, rs): return next(r for r in rows if r['npay'] == p and r['nattr'] == a and r['custom'] == c and r['rays'] == rs)
errs = []
for (p, a, c) in [(p, a, 0) for a in (2, 8) for p in (0, 2, 8, 16, 32)] + [(p, a, 1) for p in (2, 8, 32) for a in (0, 1, 2, 4, 8)]:
    r = get(p, a, c, 'rand'); errs.append(M.optix_registers(p, a) - r['ncu']['launch__registers_per_thread'])
P('registers per thread, formula minus ncu over all %d configurations: mean %+.1f, max abs %.1f' % (len(errs), np.mean(errs), np.max(np.abs(errs))))
res = {}
for rs in ('rand', 'coh'):
    a, b = get(0, 2, 0, rs), get(32, 2, 0, rs)
    t_a, t_b = a['ms'], b['ms']; wa, wb = M.resident_warps(M.optix_registers(0, 2)), M.resident_warps(M.optix_registers(32, 2))
    k = np.log(t_b / t_a) / np.log(wa / wb); res[rs] = k
    P('triangles %s: payload 0 -> 32 took %.3f -> %.3f ms, warps %.1f -> %.1f, exponent k = %.2f (set from this pair)' % (rs, t_a, t_b, wa, wb, k))
P('test on the other triangle configurations (k fixed): ' + ', '.join('%s p%d a%d pred %+.1f%% meas %+.1f%%' % (rs, p, a, (M.payload_factor(M.optix_registers(0, 2), M.optix_registers(p, a), res[rs]) - 1) * 100, (get(p, a, 0, rs)['ms'] / get(0, 2, 0, rs)['ms'] - 1) * 100) for rs in ('rand', 'coh') for p, a in ((16, 2), (16, 8), (32, 8))))
for rs in ('rand', 'coh'):
    t0, t1 = get(2, 2, 1, rs)['ms'], get(32, 2, 1, rs)['ms']
    P('custom primitives %s (k from the triangle run of the same ray set %.2f): payload 2 -> 32 predicted %+.0f%%, measured %+.0f%%' % (rs, res[rs], (M.payload_factor(M.optix_registers(2, 2), M.optix_registers(32, 2), res[rs]) - 1) * 100, (t1 / t0 - 1) * 100))
P('Reading: the register and occupancy part (registers -> resident warps) is predicted; the slowdown needs the regime exponent k (0.1 random triangles, 0.7 coherent), set from one pair, and the custom primitive random case (intersection program on the SM, issue bound) is NOT predicted by the triangle k.')
if len(sys.argv) > 1: open(sys.argv[1], 'w').write('\n'.join(out) + '\n')
