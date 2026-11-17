"""Knee blind test predictions (2026-11-17), BEFORE any run. Hypothesis K: the knee is where the MEAN distinct node lines per step per warp (step_mean, F200) equals C = 2^3.35 = 10.2 (F200, not refitted),
in the unit bound regime only (16 or more warps).  /usr/bin/python3 model/predict_knee2.py notes/unit3g/knee2_inputs.json"""
import json, sys
import numpy as np
INP = {int(k): v for k, v in json.load(open(sys.argv[1])).items()}; G = np.array([1, 2, 4, 8, 16, 32.]); C = 2 ** 3.35
def cross(d, C):                         # g where the decreasing D(g) crosses C, log-log
    d = np.array(d, float)
    if d[0] <= C: return 0.0
    if d[-1] >= C: return 99.0
    i = int(np.where(d > C)[0].max()); return float(2 ** (np.log2(G[i]) + (np.log(d[i]) - np.log(C)) / (np.log(d[i]) - np.log(d[i + 1]))))
def g_eq(n, g):                          # the plain g with the same mean distinct lines as the shifted variant of g
    d = [INP[n]['plain'][str(int(x))]['mean'] if isinstance(list(INP[n]['plain'])[0], str) else INP[n]['plain'][int(x)]['mean'] for x in G]
    target = INP[n]['shifted'][str(g)]['mean']
    return float(2 ** np.interp(-np.log(target), -np.log(d), np.log2(G)))
print('C = %.2f. (1) knee g predicted at W = 16 and 32 (same g at both): ' % C)
for n in INP:
    d = [INP[n]['plain'][str(int(x))]['mean'] for x in G]; print('   %8d  D_mean(g) %s -> g_knee %.2f' % (n, ' '.join('%.1f' % v for v in d), cross(d, C)))
print('(2) W = 12: no knee (top / floor < 1.5) at every size')
print('(3) shifted variant: g, plain D_mean, shifted D_mean, g_eq (predicted equivalent plain g under K; the distinct ray alternative says g itself)')
for n in INP:
    for g in (2, 4, 8, 16, 32):
        print('   %8d g=%-2d  D_mean plain %.1f shifted %.1f  g_eq %.2f' % (n, g, INP[n]['plain'][str(g)]['mean'], INP[n]['shifted'][str(g)]['mean'], g_eq(n, g)))
