"""Arkade Glove 3D: is the 2.56 recorder / replay ratio the dropped third column? (2026-11-17; predictions model/priorworks/predictions_arkade_callback.md part A)
Usage: /usr/bin/python3 model/priorworks/check_ark_dim.py"""
import numpy as np, json, glob
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
R = '/home/min/a/cashman3/eoss/notes/q36/raw/'; D = '/dev/shm/pw/data/'; old = json.load(open('/home/min/a/cashman3/eoss/model/priorworks/pred.json'))
for ds in ('glove3d', 'gowalla', 'manuscript'):
    o = old['arkade_' + ds]; N = o['N']; a = np.array(open(D + ds + '.txt').read().replace(',', ' ').split(), dtype=np.float64).reshape(-1, 3)
    p = a.astype(np.float32).astype(np.float64); q = p[np.random.default_rng(7).choice(N, 10000, replace=False)]
    hc = np.fromfile(sorted(glob.glob(R + 'ark_split_%s_fixed_*_1_cnt.bin' % ds))[-1], '<u4').astype(float) / 2
    p2 = p.copy(); p2[:, 2] = 0; q2 = q.copy(); q2[:, 2] = 0
    c3 = 1.1 * cKDTree(p).query_ball_point(q, o['r'], p=np.inf, return_length=True, workers=-1).astype(float)
    c2 = 1.1 * cKDTree(p2).query_ball_point(q2, o['r'], p=np.inf, return_length=True, workers=-1).astype(float)
    nz = (c2 > 0) & (hc > 0)
    print('%s: r %.6g; mean 3D x1.1 %.1f, 2D x1.1 %.1f, recorder %.1f; recorder/3D %.3f, recorder/2D %.3f; median per query recorder/2D %.3f; Spearman(2D, recorder) %.4f; Spearman(3D, recorder) %.4f' % (
        ds, o['r'], c3.mean(), c2.mean(), hc.mean(), hc.mean() / c3.mean(), hc.mean() / c2.mean(), np.median(hc[nz] / c2[nz]), spearmanr(c2, hc)[0], spearmanr(c3, hc)[0]))
