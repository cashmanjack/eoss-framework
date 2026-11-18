"""Phase 11 C: lane statistics of the real data rows against the misses of model v1 (2026-11-18, report only, model v1 stays frozen).
Usage: python3 -I model/priorworks/div_real.py RAWDIR TS      RAWDIR holds the Div_1 lane histogram dumps of notes/q41_run_div.sh (stamp TS).
Per row: mean lanes per callback (and over 32), mean lanes weighted by time, share of callbacks and of warp time at 8 or fewer lanes (definitions of ttu/bench/Div_1/lh_summary.py, F231).
The model's miss per row is measured over predicted of model v1 (results_model_v1.md, ratio v1); the Arkade original rows reuse the lane statistics of the fixed build (the probe is only in the fixed build).
Prints the table and the Spearman rank correlation of each lane measure with log(ratio) over the 14 rows and within programs."""
import sys, os, glob, importlib.util, numpy as np
from scipy.stats import spearmanr
spec = importlib.util.spec_from_file_location('lhs', os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../ttu/bench/Div_1/lh_summary.py')); lhs = importlib.util.module_from_spec(spec); spec.loader.exec_module(lhs)
raw, ts = sys.argv[1], sys.argv[2]
V1 = {'rtnn_bunny_r1': 1.89, 'rtnn_bunny_r2': 1.97, 'rtnn_dragon_r1': 2.03, 'rtnn_dragon_r2': 2.42, 'tk_road_k5': 0.71, 'tk_road_k659': 0.59, 'tk_porto_k5': 0.46, 'tk_uni_k5': 1.29,
      'ark_gowalla': (2.81, 1.75), 'ark_glove3d': (2.77, 3.24), 'ark_manuscript': (3.41, 3.20)}       # ratio v1 (fixed, original for Arkade)
rows = []; stats = {}
for row in V1:
    fs = sorted(glob.glob('%s/%s_%s_b*.bin' % (raw, row, ts)) + glob.glob('%s/%s_%s_r*.bin' % (raw, row, ts)) + glob.glob('%s/%s_%s.bin' % (raw, row, ts)))
    if not fs: print('missing', row); continue
    a = np.vstack([lhs.load(f) for f in fs]); o = lhs.summ(a)[0]; stats[row] = o
    print('%-16s files %2d warps %7d callbacks %.3e mean lanes %5.2f (eff %.3f) time weighted lanes %5.2f  callbacks <=8: %.3f  warp time <=8: %.3f  <=16: %.3f' % (row, len(fs), o['warps_with_callbacks'], o['callbacks'], o['mean_lanes_per_callback'], o['lane_eff_callbacks'], o['mean_lanes_time_weighted'], o['share_callbacks_le8'], o['warp_time_le8_lanes'], o['warp_time_le16_lanes']))
X = {m: [] for m in ('mean_lanes_per_callback', 'mean_lanes_time_weighted', 'share_callbacks_le8', 'warp_time_le8_lanes')}; Y = []; P = []
for row, v in V1.items():
    if row not in stats: continue
    for r in (v if isinstance(v, tuple) else (v,)):
        for m in X: X[m].append(stats[row][m])
        Y.append(np.log(r)); P.append(row.split('_')[0])
Y = np.array(Y); P = np.array(P); print('\nrows used: %d (Arkade counted twice: fixed and original, same lane statistics)' % len(Y))
for m in X:
    x = np.array(X[m]); rho = spearmanr(x, Y)
    s = '  within programs: ' + ', '.join('%s %.2f' % (p, spearmanr(x[P == p], Y[P == p])[0]) for p in ('rtnn', 'tk', 'ark') if (P == p).sum() > 2)
    print('%-26s Spearman with log(measured/predicted) over all rows: %+.2f (p %.2f)%s' % (m, rho[0], rho[1], s))
print('\nper program mean of the lane measure, and mean log miss:')
for p in ('rtnn', 'tk', 'ark'):
    k = P == p
    if k.any(): print('%-5s mean lanes %.2f  warp time <=8 %.3f  mean ratio v1 %.2f' % (p, np.mean(np.array(X['mean_lanes_per_callback'])[k]), np.mean(np.array(X['warp_time_le8_lanes'])[k]), np.exp(Y[k].mean())))
