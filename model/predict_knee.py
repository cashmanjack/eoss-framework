"""Knee map predictions (2026-11-17), written BEFORE any run. Hypothesis H: per ray cost depends on the distinct node lines a warp requests per step against a capacity C.
    /usr/bin/python3 model/predict_knee.py notes/unit3g/knee_inputs.json
Knee (measured, also used to calibrate): for a configuration (soup size n, warps W) with c(g), g = 1 2 4 8 16 32 lanes per ray: top = c(1), floor = c(32), a knee exists if top / floor >= 1.5;
g_knee = where c(g) crosses the midpoint (top + floor) / 2, linear in c, log2 g axis, between the last g above it and the first below.
Calibration: C of every statistic from ONE configuration only, soup:100,000 at 32 warps (knee from the F199 design B numbers 85.9 76.3 44.0 21.1 15.6 18.0), as D(g_knee) with log D linear in log g.
Prediction for any other (n, W): g_pred = g where the statistic D(g) (times W for the xW variants, C scaled to 32 warps) crosses C; below g = 1 = no knee (demand under C everywhere), above 32 = no knee (over C everywhere)."""
import json, sys
import numpy as np
G = np.array([1, 2, 4, 8, 16, 32.])
INP = {int(k): {int(g): v for g, v in d.items()} for k, d in json.load(open(sys.argv[1])).items()}
STAT = {'H step_max': ('step_max', False), 'step_mean': ('step_mean', False), 'total': ('total', False), 'step_max x W': ('step_max', True), 'total x W': ('total', True)}


def knee(c):
    c = np.asarray(c, float); top, fl = c[0], c[-1]
    if top / fl < 1.5: return None
    mid = (top + fl) / 2
    for i in range(1, 6):
        if c[i] <= mid: return float(2 ** (np.log2(G[i - 1]) + (c[i - 1] - mid) / (c[i - 1] - c[i]) * (np.log2(G[i]) - np.log2(G[i - 1]))))
    return None


def D_at(n, stat, g, W, xW):
    d = np.array([INP[n][int(x)][stat] for x in G], float) * (W if xW else 1.0)
    return float(np.exp(np.interp(np.log2(g), np.log2(G), np.log(d))))


def g_cross(n, stat, W, xW, C):
    d = np.array([INP[n][int(x)][stat] for x in G], float) * (W if xW else 1.0)
    if d[0] <= C: return 0.0          # demand under C even with 32 distinct rays: no knee
    if d[-1] >= C: return 99.0        # over C even with one ray per warp: no knee
    i = int(np.where(d > C)[0].max()); y0, y1 = np.log(d[i]), np.log(d[i + 1])
    return float(2 ** (np.log2(G[i]) + (y0 - np.log(C)) / (y0 - y1) * 1.0))


CAL = [85.9, 76.3, 44.0, 21.1, 15.6, 18.0]; gk = knee(CAL)
print('calibration soup:100,000 W=32: g_knee %.2f' % gk)
C = {k: D_at(100000, s, gk, 32, x) / (1.0) for k, (s, x) in STAT.items()}
print('C per statistic (x W variants: demand x 32 at calibration):', {k: round(v, 1) for k, v in C.items()})
print('\nD(g) per statistic and size (g = 1 2 4 8 16 32):')
for k, (s, x) in STAT.items():
    if x: continue
    for n in INP: print('%-14s %8d | %s' % (k, n, ' '.join('%6.1f' % INP[n][int(g)][s] for g in G)))
print('\nPREDICTED g_knee (0 = no knee, demand under C; 99 = no knee, over C), configurations n x W = 4 8 16 32:')
out = {}
for k, (s, x) in STAT.items():
    for n in INP:
        out[(k, n)] = [g_cross(n, s, W, x, C[k]) for W in (4, 8, 16, 32)]
        print('%-14s %8d | %s' % (k, n, ' '.join('%5.2f' % v if 0 < v < 99 else ('none' if v else 'none') for v in out[(k, n)])))
json.dump({'C': C, 'pred': {'%s|%d' % k: v for k, v in out.items()}, 'g_cal': gk}, open('notes/unit3g/knee_pred.json', 'x'))
