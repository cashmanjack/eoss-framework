"""ncu instructions per warp per candidate, Sphere_1 on the 5090 (predictions_5090_batch9.md). python3 analyze_5090_batch9_ncu.py WORKDIR"""
import csv, glob, sys, numpy as np
W = sys.argv[1]; WARPS = 1048576 / 32
def val(f):
    d = {}
    for r in csv.reader(l for l in open(f) if l.startswith('"')):
        if r[0] == 'ID': continue
        d[r[12]] = float(r[14].replace(',', ''))
    return d
for pre in ('ncu', 'ncub'):
    print('==', pre, '(D1 axis 1.27 r)' if pre == 'ncu' else '(D1b near miss)')
    for k in ('sph', 'custom', 'tri0', 'tri1'):
        ns = (16, 64, 256); ins = []; sec = []
        for n in ns:
            d = val(glob.glob('%s/%s_%s_n%d_2*.csv' % (W, pre, k, n))[0]); ins.append(d['smsp__inst_executed.sum'] / WARPS); sec.append(d['lts__t_sectors.sum'] / WARPS)
        b, a = np.polyfit(ns, ins, 1); bs, as_ = np.polyfit(ns, sec, 1)
        print('%-7s inst/warp N=16,64,256: %s  per candidate %.1f (intercept %.0f)   L2 sectors/warp %s per candidate %.2f' % (k, ' '.join('%.0f' % x for x in ins), b, a, ' '.join('%.1f' % x for x in sec), bs))
