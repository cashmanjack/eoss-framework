"""Phase 15 item 2 scoring: Arkade Glove 3D in real 3D against the 2D build, same session (predictions model/priorworks/predictions_phase15.md, e57d74e).
Usage: /usr/bin/python3 model/priorworks/analyze_ark3d.py TS    (TS = the timestamp of notes/q45/ark3d_run.sh, e.g. 20261119T101720) > model/priorworks/results_ark3d.md"""
import numpy as np, json, glob, re, sys, statistics
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
TS = sys.argv[1]; R = '/home/min/a/cashman3/eoss/notes/q45/raw/'; D = '/local/scratch/a/cashman3/thirdparty/pw/data/'; CLK = 2.5e9
old = json.load(open('/home/min/a/cashman3/eoss/model/priorworks/pred.json'))['arkade_glove3d']; N = old['N']; r = old['r']; B = old['B'] * 0.09
a = np.array(open(D + 'ark_glove3d.txt').read().replace(',', ' ').split(), dtype=np.float64).reshape(-1, 3); p = a[:N].astype(np.float32).astype(np.float64); q = a[N:N + 10000].astype(np.float32).astype(np.float64)
t3 = cKDTree(p); c3 = t3.query_ball_point(q, r, p=np.inf, return_length=True, workers=-1).astype(float)
p2 = p.copy(); p2[:, 2] = 0; q2 = q.copy(); q2[:, 2] = 0; c2 = cKDTree(p2).query_ball_point(q2, r, p=np.inf, return_length=True, workers=-1).astype(float)
def wmax(x): nw = len(x) // 32; return x[:nw * 32].reshape(nw, 32).max(1)
def kern(tag): return [float(re.search(r'KERNEL queries \d+ ms (\S+)', open(f).read())[1]) for f in sorted(glob.glob(R + 'ark3d_%s_%s_*.txt' % (tag, TS)))]
print('# Arkade Glove 3D in real 3D against the 2D build (2026-11-19, triton GPU 1, session %s)\n' % TS)
print('Queries = the 10,000 appended rows of ark_glove3d.txt; replay = Chebyshev box count of the 10,000 queries on the N = %d points, x 1.10 (F207 convention); recorder = counters / 2 (two launches).\n' % N)
t2d, t3d = kern('time2d'), kern('time3d')
print('## Launch time (CUDA event, fixed build, no probes)\n\n| build | runs ms | median ms | F207 prediction (model B) | measured / predicted |\n|---|---|---|---|---|')
for n, L in (('2D (F207 build)', t2d), ('3D (z read from the file)', t3d)): m = statistics.median(L); print('| %s | %s | %.1f | %.2f | %.2f |' % (n, ', '.join('%.1f' % x for x in L), m, B, m / B))
print('\n3D / 2D launch time: %.3f (A3 prediction 0.45 to 0.65)\n' % (statistics.median(t3d) / statistics.median(t2d)))
print('## Counts (recorder from the probe build, 2 runs each)\n\n| build | run | recorder mean | 3D replay x1.1 mean | recorder / 3D replay | 2D replay x1.1 mean | recorder / 2D replay | Spearman vs 3D | Spearman vs 2D | recorder mean / exact 3D count (no 1.10) |\n|---|---|---|---|---|---|---|---|---|---|')
rec = {}
for tag in ('rec2d', 'rec3d'):
    for f in sorted(glob.glob(R + 'ark3d_%s_%s_*_cnt.bin' % (tag, TS))):
        hc = np.fromfile(f, '<u4').astype(float) / 2; rec[(tag, f)] = hc
        print('| %s | %s | %.1f | %.1f | %.3f | %.1f | %.3f | %.4f | %.4f | %.3f |' % (tag, f.split('_')[-2], hc.mean(), 1.1 * c3.mean(), hc.mean() / (1.1 * c3.mean()), 1.1 * c2.mean(), hc.mean() / (1.1 * c2.mean()), spearmanr(c3, hc)[0], spearmanr(c2, hc)[0], hc.mean() / c3.mean()))
print('\n## Slowest warp (probe runs; definitions of F220: calls per step = warp calls / its largest lane count; life cycles per call over the model\'s 21,000 x 0.09; its largest lane count / the replay\'s largest warp count)\n')
print('| build | run | probe launch ms | slowest warp life ms | slowest / launch | warp calls / its max lane count | life cycles per call / model | its max lane count / replay max (3D) | product | callback share | sms used | callback cycles per call |\n|---|---|---|---|---|---|---|---|---|---|---|---|')
for tag in ('rec2d', 'rec3d'):
    for f in sorted(glob.glob(R + 'ark3d_%s_%s_*_warp.bin' % (tag, TS))):
        b = np.fromfile(f, '<u8').reshape(-1, 16).astype(np.float64); tot = b[:, 11] - b[:, 8]; tr = b[:, 10] - b[:, 9]; life = (b[:, 13] - b[:, 12]) / 1e6; i = int(np.argmax(life))
        hc = rec[(tag, f.replace('_warp.bin', '_cnt.bin'))]; wm = wmax(hc); wr = wmax(1.1 * c3); K_ = float(re.search(r'KERNEL queries \d+ ms (\S+)', open(f.replace('_warp.bin', '.txt')).read())[1])
        Fc = b[i, 1] / wm[i]; Fl = life[i] * 1e-3 * CLK / b[i, 1] / (21000 * 0.09); Fn = wm[i] / wr.max(); sms = re.search(r'sms_used (\d+)', open(f.replace('_warp.bin', '.txt')).read())[1]
        print('| %s | %s | %.1f | %.1f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %s | %.0f |' % (tag, f.split('_')[-2], K_, life.max(), life.max() / K_, Fc, Fl, Fn, Fc * Fl * Fn, b[:, 0].sum() / tot.sum(), sms, b[:, 0].sum() / b[:, 1].sum()))
print('\nLane statistics: per warp max count, replay 3D: mean over warps %.0f, max %.0f; replay 2D: mean %.0f, max %.0f; recorder 3D run 1: mean %.0f, max %.0f.' % (wmax(1.1 * c3).mean(), wmax(1.1 * c3).max(), wmax(1.1 * c2).mean(), wmax(1.1 * c2).max(), wmax(rec[[k for k in rec if k[0] == 'rec3d'][0]]).mean(), wmax(rec[[k for k in rec if k[0] == 'rec3d'][0]]).max()))
