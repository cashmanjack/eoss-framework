"""Extra hardware hits at short tmax (predictions: model/predictions_hwhits.md). usage: hwhits_analyze.py EXACT.bin HW_TS (reads notes/tmaxreplay/hw_<tmax>_HW_TS.res)"""
import glob, sys, numpy as np
ex, ts = sys.argv[1:3]; n = 262144
raw = open(ex, 'rb').read(); tf = np.frombuffer(raw[:8 * n], np.float64); pf = np.frombuffer(raw[8 * n:], np.int32)
rays = np.fromfile('notes/tmaxreplay/soup_20261116T225523.rays', np.float32).reshape(-1, 8)
for tm in ('0.02', '0.1', '0.2', '1'):
    o = np.fromfile('notes/tmaxreplay/hw_%s_%s.res' % (tm, ts), np.uint32).reshape(n, 8); t = float(tm)
    hh = o[:, 0] & 1 == 1; eh = tf <= t; ht = o[:, 1].view(np.float32).astype(np.float64)
    extra = hh & ~eh; miss = ~hh & eh
    print('\ntmax %s: hw hit %.4f  exact hit %.4f  | hw hit, exact miss: %d (%.4f)   hw miss, exact hit: %d (%.4f)   both agree %.4f' % (tm, hh.mean(), eh.mean(), extra.sum(), extra.mean(), miss.sum(), miss.mean(), (hh == eh).mean()))
    if extra.sum():
        r = tf[extra] / t; fin = np.isfinite(r)
        print('  extras: exact nearest hit exists beyond tmax in %d of %d (none at all: %d)' % (fin.sum(), extra.sum(), (~fin).sum()))
        q = np.percentile(r[fin], [0, 10, 50, 90, 100]) if fin.any() else []
        print('  t_first / tmax of extras (min, p10, median, p90, max):', np.round(q, 4))
        for lim in (1.0001, 1.001, 1.01, 1.1, 2.0): print('    within t_first <= %.4f x tmax: %.3f' % (lim, (r[fin] <= lim).sum() / extra.sum()), end='')
        print()
        print('  hardware reported t of extras / tmax (min, median, max):', np.round(np.percentile(ht[extra], [0, 50, 100]) / t, 4), ' reported prim == exact nearest prim: %.3f' % (o[extra, 2].astype(np.int64) == pf[extra]).mean())
        print('  hw t above tmax: %d of %d' % ((ht[extra] > t).sum(), extra.sum()))
    if miss.sum():
        r = tf[miss] / t; print('  misses: t_first / tmax (min, median, max):', np.round(np.percentile(r, [0, 50, 100]), 4))
    both = hh & eh; print('  both hit: hw t equals exact t within 1e-5 relative: %.4f, prim equal %.4f' % ((np.abs(ht[both] - tf[both]) <= 1e-5 * np.maximum(1, tf[both])).mean(), (o[both, 2].astype(np.int64) == pf[both]).mean()))
