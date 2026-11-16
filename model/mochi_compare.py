"""Mochi: compare the committed predictions (model/predictions_mochi.md, run A1 replay) with run B (recorder). CPU only.
    /usr/bin/python3 model/mochi_compare.py REC_DIR A1_PRED_DIR B_PRED_DIR   (pred dirs: /dev/shm/mochi/pred_dump_A1, pred_dump_B with fNN.rep files)"""
import sys, numpy as np
rec, pa, pb = sys.argv[1:4]
ns = np.fromfile(rec + '/launch_ns.bin', np.uint64).astype(np.float64).reshape(-1, 400)
print('| frame | measured calls / query | A1 replay (predicted) | B own replay | pred error A1 | error B own | measured lane eff | A1 lane eff | B own lane eff | span median ms | span min | span max | RT part lower bound ms |')
print('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
LB = [0.0339] * 5 + [0.0357, 0.0366, 0.0382, 0.0398, 0.0413]
res = []
for f in range(10):
    m = np.fromfile('%s/cand_f%02d.bin' % (rec, f), np.uint32).astype(np.float64); N = len(m)
    def eff(c): w = c[: N // 32 * 32].reshape(-1, 32); return w.sum() / (32 * w.max(1).sum())
    ra = np.fromfile('%s/f%02d.rep' % (pa, f), np.int32).reshape(-1, 8)[:, 2] * 1.006; rb = np.fromfile('%s/f%02d.rep' % (pb, f), np.int32).reshape(-1, 8)[:, 2] * 1.006
    s = ns[f] / 1e6; res.append(np.median(s))
    print('| %d | %.3f | %.3f | %.3f | %+.1f%% | %+.1f%% | %.3f | %.3f | %.3f | %.4f | %.4f | %.4f | %.4f |' % (f, m.mean(), ra.mean(), rb.mean(), 100 * (ra.mean() / m.mean() - 1), 100 * (rb.mean() / m.mean() - 1), eff(m), eff(ra), eff(rb), np.median(s), s.min(), s.max(), LB[f]))
print('\nspan ratio frame 9 / frame 0 (medians): %.2f; frames 0 to 4 medians: %s' % (res[9] / res[0], ' '.join('%.4f' % x for x in res[:5])))
