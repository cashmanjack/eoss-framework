"""RTNN and Arkade on the F207 datasets: (A) replay counts against the recorder's exact counts, (B) warp time split into traversal window and callback by clock64 probes (as F168). 2026-11-17, triton GPU 1.
Raw: notes/q36/raw (private): rtnn_event (launch event time, no recorder, no probes), rtnn_split and ark_split (recorder per query counts + per warp probe records, 2 runs each; 'time search compute' of the split runs is NOT a launch time: it includes the recorder's host copies).
Per warp record (16 u64): [0] callback cycles, [1] callback calls, [8] raygen start clock, [9] before trace, [10] after trace, [11] raygen end clock; Arkade also [12] global timer start, [13] global timer end.
Replay = the F207 CPU counts (exact Chebyshev box count including the point itself, x 1.10 hardware factor; RTNN with the K early termination rule). Recorder = intersection program calls per query (Arkade counts in two launches, halved).
Usage: /usr/bin/python3 model/priorworks/split_rtnn_arkade.py > model/priorworks/results_split_rtnn_arkade.md"""
import numpy as np, json, glob, re
from scipy.spatial import cKDTree
R = '/home/min/a/cashman3/eoss/notes/q36/raw/'; M = '/home/min/a/cashman3/eoss/model/priorworks/'; D = '/dev/shm/pw/data/'; SM, CLK = 142, 2.5e9
old = json.load(open(M + 'pred.json')); med = lambda x: float(np.median(x))
def load(f): return np.array(open(D + f).read().replace(',', ' ').split(), dtype=np.float64).reshape(-1, 3)
def wmax_sum(x): n = len(x); nw = n // 32; v = x[:nw * 32].reshape(nw, 32); return float(v.max(1).sum() * 32), v.max(1)
def probes(f):
    b = np.fromfile(f, '<u8').reshape(-1, 16).astype(np.float64); tot = b[:, 11] - b[:, 8]; tr = b[:, 10] - b[:, 9]; return b, tot, tr
print('# RTNN and Arkade on the F207 datasets: replay counts against recorder counts, and warp time split (2026-11-17, triton GPU 1)\n')
print('## RTNN counts (replay = F207 CPU count x 1.10 with the K = 50 stop rule; recorder = intersection program calls per query)\n')
print('| dataset | r | replay mean | recorder mean | recorder / replay | replay warp max sum / lanes | recorder warp max sum / lanes | recorder max lane | per query median ratio |\n|---|---|---|---|---|---|---|---|---|')
rt = {}
for k, v in old.items():
    if not k.startswith('rtnn'): continue
    _, ds, r = k.split('_'); p = load(ds + '.csv').astype(np.float32).astype(np.float64); N = len(p); t = cKDTree(p)
    C = t.query_ball_point(p, v['r'], p=np.inf, return_length=True, workers=-1).astype(float); S = t.query_ball_point(p, v['r'], return_length=True, workers=-1).astype(float)
    rep = 1.10 * np.where(S > 50, 50 * (C + 1) / (S + 1), C)
    hc = np.fromfile(sorted(glob.glob(R + 'rtnn_split_%s_%s_*_1_cnt.bin' % (ds, r)))[-1], '<u4').astype(float)
    rs, _ = wmax_sum(rep); hs, _ = wmax_sum(hc); nz = (rep > 0) & (hc > 0)
    rt[k] = dict(rep=rep, hc=hc, N=N)
    print('| %s | %.5g | %.3f | %.3f | %.2f | %.2f | %.2f | %d | %.2f |' % (ds, v['r'], rep.mean(), hc.mean(), hc.mean() / rep.mean(), rs / (len(rep) // 32 * 32) / 1, hs / (len(hc) // 32 * 32), hc.max(), np.median(hc[nz] / rep[nz])))
print('\n## RTNN time: launch event time (no probes, no recorder, 3 runs) against the F207 prediction, and what is inside "time search compute"\n')
print('| dataset | r | F207 predicted | launch event ms (rt_rec build) | event / predicted | "search compute" ms (rt_rec build, same runs) | not in the launch ms (host: allocate and fill the output buffer) | output buffer MB | host us per MB | "search compute" ms (F207 build) | launch estimate for the F207 build = that minus host | estimate / predicted | prediction with recorder counts | event / that |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
ev = {}
for k, v in old.items():
    if not k.startswith('rtnn'): continue
    _, ds, r = k.split('_'); E = med([float(re.search(r'KERNEL queries \d+ ms (\S+)', open(f).read())[1]) for f in glob.glob(R + 'rtnn_event_%s_%s_*.txt' % (ds, r))])
    sc0 = med([float(re.search(r'time search compute: (\S+)', open(f).read())[1]) for f in glob.glob(R + 'rtnn_event_%s_%s_*.txt' % (ds, r))])
    sc = med([float(re.search(r'time search compute: (\S+)', open(f).read())[1]) for f in glob.glob('/home/min/a/cashman3/eoss/notes/q33/raw/rtnn_%s_%s_*.txt' % (ds, r))])
    N = v['N']; mb = N * 50 * 4 / 1e6; q = rt[k]['hc'].mean() + 1; pr = (110 + 4.9 * q) * N / (SM * CLK) * 1e3; ev[k] = E; host = sc0 - E; est = sc - host
    print('| %s | %.5g | %.3f | %.4f | %.2f | %.3f | %.3f | %.0f | %.1f | %.3f | %.3f | %.2f | %.3f | %.2f |' % (ds, v['r'], v['ms'], E, E / v['ms'], sc0, host, mb, host * 1e3 / mb, sc, est, est / v['ms'], pr, E / pr))
print('\nThe rt_rec build carries the (disabled) probe code, so its launch is longer than the F207 build\'s (dragon r2: 10.48 against 8.97 ms of search compute); the estimate column removes the same host part from the F207 build\'s search compute and is the better guess of that build\'s launch.')
print('\n## RTNN warp time split (probe runs, mean of 2; the probes lengthen the launch, see the probe launch ms)\n')
print('| dataset | r | probe launch ms | probe cost | callback share of warp cycles | traversal window share (trace - callback) | outside trace share | warps in flight per SM | slowest warp ms | slowest warp / launch | measured cycles per query per SM | model c0 + cq q | callback part: measured / model (4.9 q) | traversal part: measured / model (110) |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
for k, v in old.items():
    if not k.startswith('rtnn'): continue
    _, ds, r = k.split('_'); fs = sorted(glob.glob(R + 'rtnn_split_%s_%s_*T164852_1_warp.bin' % (ds, r))); 
    b, tot, tr = probes(fs[0]); K_ = float(re.search(r'KERNEL queries \d+ ms (\S+)', open(fs[0].replace('_warp.bin', '.txt')).read())[1]); E = ev[k]; N = v['N']; q = rt[k]['hc'].mean() + 1
    cb = b[:, 0].sum() / tot.sum(); trv = (tr.sum() - b[:, 0].sum()) / tot.sum(); out = 1 - tr.sum() / tot.sum(); wif = tot.sum() / CLK / (K_ * 1e-3) / SM
    cyc = E * 1e-3 * SM * CLK / N; print('| %s | %.5g | %.4f | %.2f | %.2f | %.2f | %.2f | %.1f | %.4f | %.2f | %.0f | %.0f | %.2f | %.2f |' % (ds, v['r'], K_, K_ / E, cb, trv, out, wif, tot.max() / CLK * 1e3, tot.max() / CLK * 1e3 / K_, cyc, 110 + 4.9 * q, cyc * cb / (4.9 * q), cyc * (trv + out) / 110))
print('\n## Arkade counts (replay = F207 CPU count x 1.10; recorder = intersection program calls per query / 2)\n')
print('| dataset | replay mean | recorder mean | recorder / replay | replay warp max sum / lanes | recorder warp max sum / lanes | per query median ratio |\n|---|---|---|---|---|---|---|')
ak = {}
for ds in ('gowalla', 'glove3d', 'manuscript'):
    o = old['arkade_' + ds]; p = load(ds + '.txt'); N = len(p); qi = np.random.default_rng(7).choice(N, 10000, replace=False); q = p[qi]; t = cKDTree(p)
    rep = 1.10 * t.query_ball_point(q, o['r'], p=np.inf, return_length=True, workers=-1).astype(float)
    hc = np.fromfile(sorted(glob.glob(R + 'ark_split_%s_fixed_*_1_cnt.bin' % ds))[-1], '<u4').astype(float) / 2; rs, _ = wmax_sum(rep); hs, _ = wmax_sum(hc); ak[ds] = (rep, hc)
    print('| %s | %.1f | %.1f | %.2f | %.2f | %.2f | %.2f |' % (ds, rep.mean(), hc.mean(), hc.mean() / rep.mean(), rs / (len(rep) // 32 * 32), hs / (len(hc) // 32 * 32), np.median(hc / np.maximum(rep, 1))))
print('\n## Arkade time split (probe runs, 2 runs each; launch = CUDA event with probes)\n')
print('| dataset | variant | launch ms | callback share of warp cycles | traversal window share | warps | SMs used (of 142) | mean warp life ms | slowest warp life ms | slowest / launch | model B ms (F207) | slowest warp / model B | in the slowest warp: warp calls / its max lane count | life cycles per call / model cycles per call | its max lane count / model B count | product of the three | callback cycles per call (all warps) |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
for ds in ('gowalla', 'glove3d', 'manuscript'):
    o = old['arkade_' + ds]
    for var in ('fixed', 'orig'):
        L = []; rep, hc = ak[ds]; _, wm = wmax_sum(hc); _, wmr = wmax_sum(rep); f = 0.09 if var == 'fixed' else 1.0; B = o['B'] * f
        for fn in sorted(glob.glob(R + 'ark_split_%s_%s_*_warp.bin' % (ds, var))):
            b, tot, tr = probes(fn); s_ = open(fn.replace('_warp.bin', '.txt')).read(); K_ = float(re.search(r'KERNEL queries \d+ ms (\S+)', s_)[1]); gs = re.search(r'sms_used (\d+)', s_)[1]
            life = (b[:, 13] - b[:, 12]) / 1e6; i = int(np.argmax(life)); Fc = b[i, 1] / wm[i]; Fl = life[i] * 1e-3 * CLK / b[i, 1] / (21000 * f); Fn = wm[i] / wmr.max()
            L.append((K_, b[:, 0].sum() / tot.sum(), (tr.sum() - b[:, 0].sum()) / tot.sum(), int(gs), life.mean(), life.max(), B, life.max() / B, Fc, Fl, Fn, Fc * Fl * Fn, b[:, 0].sum() / b[:, 1].sum()))
        a = np.array(L).mean(0)
        print('| %s | %s | %.3f | %.2f | %.2f | %d | %d | %.3f | %.3f | %.2f | %.3f | %.2f | %.2f | %.2f | %.2f | %.2f | %.0f |' % (ds, var, a[0], a[1], a[2], len(b), a[3], a[4], a[5], a[5] / a[0], a[6], a[7], a[8], a[9], a[10], a[11], a[12]))
