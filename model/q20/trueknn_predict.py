"""TrueKNN (vani-nag/OWLRayTracing s01-trueknn, patched copy /dev/shm/tk/p, device buffer output, per round probes) on triton GPU 1: predictions BEFORE the held out runs.
Calibration set A = 100,000 uniform points (5 runs, notes/q20/tk/A_*.txt): three constants, nothing else fitted.
  rho  = measured warp calls of the intersection program / sum over warps of the largest exact box hit count among the warp's active lanes (pooled over rounds)
  c    = measured callback cycles per warp call (pooled over rounds)      g = total warp cycles / callback cycles (pooled)      W = implied warps in flight per SM (pooled)
Prediction for a set X (200,000 and 400,000 uniform points, same start radius 0.004, k 16, radius doubling): per round exact box hit counts by CPU (Chebyshev, includes self), active set rule: a query
stops when its exact count minus self reaches k; time of the round = warp_calls x c x g / (142 x W x clk) (clk cancels); no launch floor, no per round parameter.
    /usr/bin/python3 model/q20/trueknn_predict.py > model/predictions_trueknn.md"""
import re, statistics as st, numpy as np
from scipy.spatial import cKDTree
K = 16; R0 = 0.004; SM = 142; CLK = 2.5e9
def load_A():
    rows = {}
    for r in range(1, 6):
        for l in open('notes/q20/tk/A_%d.txt' % r):
            m = re.match(r'RND (\d+) radius (\S+) GT span_ms (\S+) CBK warps (\d+) total_warp_cycles (\d+) f0_cycles (\d+) f0_calls (\d+)', l)
            if m: rows.setdefault(int(m[1]), []).append([float(m[3]), float(m[5]), float(m[6]), float(m[7])])
            m = re.match(r'RND (\d+) points left (\d+)', l)
            if m: rows.setdefault(('left', int(m[1])), []).append(int(m[2]))
    return {k: (np.median(np.array(v), axis=0) if not isinstance(k, tuple) else int(np.median(v))) for k, v in rows.items()}
def counts(path, n):
    p = np.loadtxt(path, delimiter=',', dtype=np.float32)[:n].astype(np.float64); t = cKDTree(p)
    active = np.ones(n, bool); out = []; r = R0
    for rd in range(1, 12):
        c = np.zeros(n, np.int64); c[active] = t.query_ball_point(p[active], r, p=np.inf, return_length=True)
        nw = (n + 31) // 32; pad = np.zeros(nw * 32, np.int64); pad[:n] = c
        out.append((rd, r, int(active.sum()), float(pad.reshape(-1, 32).max(1).sum()), float(c.sum())))
        active = active & (c - 1 < K)
        if not active.any(): break
        r *= 2
    return out
A = load_A(); cA = counts('/dev/shm/tk/u100000.csv', 100000)
calls = sum(A[i][3] for i in range(1, 6)); rho = calls / sum(x[3] for x in cA)
c = sum(A[i][2] for i in range(1, 6)) / calls; g = sum(A[i][1] for i in range(1, 6)) / sum(A[i][2] for i in range(1, 6))
W = sum(A[i][1] for i in range(1, 6)) / (SM * CLK * sum(A[i][0] for i in range(1, 6)) * 1e-3)
print('# TrueKNN predictions on triton GPU 1 (142 SMs), written BEFORE the 200,000 and 400,000 point runs (2026-11-16)\n')
print('Program: OWLRayTracing s01-trueknn, patched copy (frame buffer in device memory, per round probes of Finding 133 type, global timer span of each launch). Data: uniform random 3D points in the unit cube, k = 16, start radius 0.004, doubling, one launch per round, refit between rounds.\n')
print('Constants from the calibration set A (100,000 points, medians of 5 runs): hardware warp calls / exact warp max box hits rho = %.3f, callback cycles per warp call c = %.0f, total warp cycles / callback cycles g = %.3f, warps in flight per SM W = %.2f.\n' % (rho, c, g, W))
print('Calibration check on A (these constants were taken from A, so only the SHAPE per round is a test): the rule predicts active queries per round and the time per round pooled-constant.\n')
def table(name, cc, meas=None):
    print('| %s round | radius | active (predicted) | %swarp calls (predicted) | launch ms (predicted) |' % (name, 'active (measured) | warp calls (measured) | launch ms (measured) | ' if meas else ''))
    print('|---|---|---|%s---|---|' % ('---|---|---|' if meas else '')); tot = 0
    for rd, r, act, wc, ex in cc:
        pc = wc * rho; ms = pc * c * g / (SM * W * CLK) * 1e3; tot += ms
        m = ''
        if meas:
            left = meas[('left', rd - 1)] if rd > 1 else 100000
            m = '%d | %.0f | %.4f | ' % (left, meas[rd][3], meas[rd][0])
        print('| %d | %g | %d | %s%.0f | %.4f |' % (rd, r, act, m, pc, ms))
    print('| total | | | %s| %.4f |\n' % ('| | %.4f ' % sum(meas[i][0] for i in meas if not isinstance(i, tuple)) + '| ' if meas else '', tot))
table('A', cA, A)
for n in (200000, 400000):
    cc = counts('/dev/shm/tk/u%d.csv' % n, n); print('## %d points (held out)\n' % n); table('B' if n == 200000 else 'C', cc)
print('What can fail: rounds where the per call latency differs from the pooled c (Finding 133 saw 4 to 5 times between rounds of kNN); the active rule (hardware candidates beyond the exact box hits could stop queries earlier, Finding 132); the warp max model for lane efficiency. Success rule (set before): total launch time within a factor 1.5 and the number of rounds equal; per round error reported as log(predicted / measured). No refitting after the comparison.')
