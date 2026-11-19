"""Scores part 3 of model/predictions_5090_batch7.md (commit 2362bd4): RTNN callback calibration on the 5090 (variant S of predictions_phase13: cb from real against stub). Run with /usr/bin/python3 (scipy). CALIBRATED on the same input: not blind."""
import csv, glob, io, re, statistics as st, sys, numpy as np
sys.path.insert(0, 'model/priorworks'); import model_v1_rtnn as m1
SM, CLK, N, K, CBREF, L = 170, 2.5e9, 100000, 50, 1.0110154, 575.
pts = np.array(open('/scratch/tgrogers-pc14-disk01/cashman3/thirdparty/div1_sm120/data/u100000.csv').read().replace(',', ' ').split(), dtype=np.float64).reshape(-1, 3).astype(np.float32).astype(np.float64)
from scipy.spatial import cKDTree
tree = cKDTree(pts)
def ker(kind, r, which):
    f = sorted(glob.glob('notes/pc14_workloads/calib_%s_r%s_*_[123].txt' % (kind, r)))
    return st.median(float(re.search(which, open(x).read())[1]) for x in f)
def ncu(pat):
    f = sorted(glob.glob('notes/pc14_workloads/' + pat))[-1]; t = open(f).read(); rows = list(csv.DictReader(io.StringIO(t[t.find('"ID"'):])))
    return [float(r['Metric Value'].replace(',', '')) for r in rows if r['Kernel Name'] == 'optixLaunch' and r['Metric Name'] == 'gpu__time_duration.sum'][0] / 1e6
res = {}
for r, ref_ker in (('0.05', ncu('ncu_rtnnC_r05_*.csv')), ('0.025', ncu('ncu_rtnnC_r025_*.csv'))):
    C, S = m1.replay_counts(pts, float(r), tree); Ce = np.where(S > K, K * (C + 1) / (S + 1), C); q = 1.10 * Ce + 1
    er, es = ker('real', r, r'KERNEL queries \d+ ms (\S+)'), ker('stub', r, r'KERNEL queries \d+ ms (\S+)')
    sr, ss = ker('real', r, r'time search compute: (\S+) ms'), ker('stub', r, r'time search compute: (\S+) ms')
    ks = ncu('ncu_stubC_r%s_*.csv' % r)
    cb = max(0.0, (er - es) * 1e-3 * SM * CLK / (N * 1.10 * C.mean()))   # event difference (the pre launch gap cancels)
    cbk = max(0.0, (ref_ker - ks) * 1e-3 * SM * CLK / (N * 1.10 * C.mean()))   # ncu kernel difference
    pr = lambda c: max(0.0, 4.9 + c - CBREF)
    T = lambda c: (110 + pr(c) * q.mean()) * N / (SM * CLK) * 1e3
    ch = q.max() * L / CLK * 1e3
    uncal = (110 + 4.9 * q.mean()) * N / (SM * CLK) * 1e3
    print('r %s: C_mean %.2f S_mean %.2f q_mean %.2f q_max %.0f | event real %.4f stub %.4f (stub/real %.2f) | ncu real %.4f stub %.4f (%.2f) | search compute real %.4f stub %.4f' % (r, C.mean(), S.mean(), q.mean(), q.max(), er, es, es / er, ref_ker, ks, ks / ref_ker, sr, ss))
    print('   cb (event diff) %.2f, cb (ncu diff) %.2f cycles per candidate; uncalibrated v1 %.4f ms; chain %.4f ms' % (cb, cbk, uncal, ch))
    for lab, c in (('cb event', cb), ('cb ncu', cbk)):
        p = max(T(c), ch); print('   calibrated (%s): predicted %.4f ms | kernel (ncu) %.4f ratio %.2f %s | event %.4f ratio %.2f | search compute %.4f ratio %.2f' % (lab, p, ref_ker, ref_ker / p, 'in' if .5 <= ref_ker / p <= 2 else 'OUT', er, er / p, sr, sr / p))
    res[r] = (ref_ker / max(T(cb), ch), uncal, ref_ker)
