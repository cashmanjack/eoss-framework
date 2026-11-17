"""Lane groups sharing a ray (g = 1, 2, 4, 8, 16, 32 lanes per ray): the existing model's predictions, no refit, written BEFORE any run (2026-11-17).
Run from the repo root: /usr/bin/python3 model/predict_lane_groups.py > model/predictions_lane_groups_table.txt
Model A = performance_model.predict_sm (Memory_4 fit); with g lanes per ray the lane requests merge: S1 = k x nodes / g, fills per lane ray F = F(g=1) / g (assumption: the same lines are needed by g lanes), q = 1.
Model B = F123 'share' form as written: A x fixed + b1 x S1 x (1 - hit) x (raw / dedupe)^-1 with S1 dedupe level = k x nodes / g, (1 - hit) = F(g=1) / S1(g=1), raw / dedupe = 1.1 g (1.1 = random rays F123). The term as written divides by g twice (S1 and ratio); reported as written.
H = my hypothesis, NOT the model: the unit's cost scales with the number of distinct rays in a warp: c(g) = c_coh + (c_inc - c_coh) x (32/g - 1) / 31 (g = 1 gives c_inc, g = 32 gives c_coh); same form for one warp latency."""
import sys
sys.path.insert(0, 'model'); sys.path.insert(0, 'model')
import audit_lane_sharing as L
PM, p, A, b1, K = L.PM, L.p, L.A, L.b1, L.K
G = (1, 2, 4, 8, 16, 32)


def A_rate(d, g, W):
    if g == 32: return L.modelA(d['nodes'], 0.0, W, True)
    return 1.0 / PM.predict_sm(p, dict(q=1.0, S1=K * d['nodes'] / g, F=d['F'] / g, D=0.0, W=W))


def B_rate(d, g):
    if g == 32: return L.modelB(d['nodes'], 0.0, 32, 32.0)
    S1 = K * d['nodes'] / g; miss = d['F'] / (K * d['nodes'])
    return A * (p['a_q'] + p['b_q'] / 32 + p['c_qW'] * 2) + b1 * S1 * miss * (1.1 * g) ** -1


def H(lo, hi, g): return lo + (hi - lo) * (32.0 / g - 1) / 31.0


print('RATE, 32 warps, cycles per ray per SM (A / B / H); measured g = 1 and g = 32 are the F197 values')
print('%8s | ' % 'soup' + ' | '.join('g=%-2d  A    B    H ' % g for g in G))
for n, d in L.SIZES.items():
    print('%8d | ' % n + ' | '.join('%5.0f %4.0f %4.0f' % (A_rate(d, g, 32), B_rate(d, g), H(d['c_coh'], d['c_inc'], g)) for g in G))
print('\nONE WARP latency, cycles (A / H)')
for n, d in L.SIZES.items():
    print('%8d | ' % n + ' | '.join('%6.0f %6.0f' % (32 * A_rate(d, g, 1), H(d['L_coh'], d['L_inc'], g)) for g in G))
