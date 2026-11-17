"""Model audit (2026-11-17, no GPU): the existing performance model, no refit, on the F197 configurations.
Run from the repo root: /usr/bin/python3 model/audit_lane_sharing.py
Models: (A) model/performance_model.py as fitted on Memory_4 (notes/performance_model.md), (B) the F123 'share' form (notes/q11/reuse_feature.json, kappa = 1 version).
Inputs per F197 launch: q = 1 (every ray one query), W = 32 (rate) or 1 (latency), nodes from the replay (model/results_unit3.md), S1 = k x lines (k = 7.46 sectors per line request, F95/queue 5; lines per ray = nodes: 11.7 nodes vs 11.0 lines at 1e5),
F = measured extra fills per ray (ncu, results_unit3.md), D = 0 (65 MB < 72 MB L2). Coherent lanes: S1 divided by 32 (identical requests merged), F = 0, sharing ratio 32. Incoherent: sharing ratio 1.1 (F123 value for random rays)."""
import sys, json
sys.path.insert(0, 'model'); sys.path.insert(0, 'ttu/bench/Memory_4')
import performance_model as PM, run_model as RM
p = PM.fit(RM.memory4_rows())
sh = json.load(open('notes/q11/reuse_feature.json'))['share_k1']['par']      # A, b1, C, g, kappa
A, b1, kappa = sh[0], sh[1], sh[4]
K = 7.46
SIZES = {1000: dict(nodes=5.95, F=2.5, c_coh=11.9, c_inc=58.3, L_coh=3841, L_inc=6576, f67=3550),
         10000: dict(nodes=9.3, F=15.9, c_coh=15.3, c_inc=68.8, L_coh=5622, L_inc=15524, f67=4960),
         100000: dict(nodes=11.7, F=41.0, c_coh=16.5, c_inc=97.9, L_coh=6412, L_inc=19165, f67=5870),
         1000000: dict(nodes=13.6, F=59.0, c_coh=19.7, c_inc=89.5, L_coh=7043, L_inc=17591, f67=6556)}


def modelA(nodes, F, W, coherent, g=1):
    S1 = K * nodes * (g / 32.0 if coherent else 1.0)       # g unused for A
    return 1.0 / PM.predict_sm(p, dict(q=1.0, S1=S1, F=F, D=0.0, W=W))


def modelB(nodes, F, W, ratio):
    S1 = K * nodes
    fixed = A * (p['a_q'] * 1 + p['b_q'] / W + p['c_qW'] * W / 16.0)
    return fixed + b1 * F * ratio ** (-kappa)            # miss term S1 x (1 - hit) = F (measured fills); lane sharing = ratio^-kappa


if __name__ == '__main__':
    print('Memory_4 parameters', {k: round(v, 3) for k, v in p.items()}); print('F123 share k1: A %.3f b1 %.3f C %.0f g %.2f kappa %g' % tuple(sh))
    print('\nRATE, W = 32 (outside the Memory_4 fit range 4 to 16), cycles per ray per SM')
    print('%9s | %6s %6s | A coh / inc / ratio | B coh / inc / ratio | measured coh / inc / ratio' % ('soup', 'nodes', 'F'))
    for n, d in SIZES.items():
        ac, ai = modelA(d['nodes'], 0.0, 32, True), modelA(d['nodes'], d['F'], 32, False)
        bc, bi = modelB(d['nodes'], 0.0, 32, 32.0), modelB(d['nodes'], d['F'], 32, 1.1)
        print('%9d | %6.2f %6.1f | %6.1f %6.1f %5.2f | %6.1f %6.1f %5.2f | %6.1f %6.1f %5.2f' % (n, d['nodes'], d['F'], ac, ai, ai / ac, bc, bi, bi / bc, d['c_coh'], d['c_inc'], d['c_inc'] / d['c_coh']))
    print('\nUPPER BOUND of the sharing term (B, every fetch a miss, ratio 1: miss = b1 x S1)')
    for n, d in SIZES.items():
        bc = modelB(d['nodes'], 0.0, 32, 32.0); ub = A * (p['b_q'] / 32 + p['c_qW'] * 2) + b1 * K * d['nodes']
        print('%9d | B coherent %.1f, incoherent upper bound %.1f, ratio at most %.2f (measured %.2f)' % (n, bc, ub, ub / bc, d['c_inc'] / d['c_coh']))
    print('\nONE WARP latency (cycles per warp = 32 x c at W = 1 for model A; B has no latency form)')
    for n, d in SIZES.items():
        lc, li = 32 * modelA(d['nodes'], 0.0, 1, True), 32 * modelA(d['nodes'], d['F'], 1, False)
        print('%9d | A coh %6.0f inc %6.0f | F67 line (identical lanes) %5d | measured coh %5d inc %5d' % (n, lc, li, d['f67'], d['L_coh'], d['L_inc']))
