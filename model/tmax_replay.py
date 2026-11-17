"""tmax replay check vs F182 (predictions: model/predictions_tmax_replay.md). usage: tmax_replay.py RAYS.bin TABLE.bin REPLAY WORK"""
import subprocess, sys, numpy as np
rays_f, table, replay, work = sys.argv[1:5]
R = np.fromfile(rays_f, np.float32).reshape(-1, 8); n = len(R)
TM = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 1e16]
req = [20.7, 22.9, 25.2, 31.5, 41.8, 56.8, 75.2, 80.2, 80.7, 80.8]; hitf = [0, .038, .075, .175, .314, .509, .754, .823, .829, .829]; ns = [.188, .188, .219, .246, .289, .371, .469, .496, .507, .508]   # hit fractions: 262,144 ray launch (F182 summary.json; corrected 2026-11-17, the first version used the 32 ray probe launch); ns = [.188, .188, .219, .246, .289, .371, .469, .496, .507, .508]
rows = []
for t in TM:
    r = R.copy(); r[:, 7] = t; r.tofile(work + '/r.bin')
    subprocess.run([replay, table, work + '/r.bin', str(n), '0', work + '/o.bin'], check=True)
    o = np.fromfile(work + '/o.bin', np.int32).reshape(n, 8)
    rows.append(dict(t=t, nodes=o[:, 0].mean(), leaves=o[:, 1].mean(), tris=o[:, 2].mean(), hit=(o[:, 5] >= 0).mean()))
u = rows[-1]; unl = u['nodes'] + u['leaves']; un = u['nodes']
print('tmax      nodes  leaves  tris   n+l   | ratio(n+l) meas ratio  err%% | ratio(n) | hit  meas  | time pred  meas  err%%')
for i, x in enumerate(rows):
    rr = (x['nodes'] + x['leaves']) / unl; mr = req[i] / req[-1]
    tp = ns[0] + (ns[-1] - ns[0]) * (req[0] + (req[-1] - req[0]) * 0 + 0) * 0  # placeholder replaced below
    # time line through the measured endpoints, requests from the replay scaled to the measured unlimited value
    rq = rr * req[-1]; tpred = ns[0] + (ns[-1] - ns[0]) * (rq - req[0]) / (req[-1] - req[0])
    print('%-8g %6.1f %6.1f %6.1f %6.1f | %.3f %.3f %6.1f | %.3f | %.3f %.2f | %.3f %.3f %6.1f' % (x['t'], x['nodes'], x['leaves'], x['tris'], x['nodes'] + x['leaves'], rr, mr, 100 * (rr / mr - 1), x['nodes'] / un, x['hit'], hitf[i], tpred, ns[i], 100 * (tpred / ns[i] - 1)))
