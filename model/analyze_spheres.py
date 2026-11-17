"""Sphere_1 analysis (predictions: model/predictions_spheres.md). /usr/bin/python3 model/analyze_spheres.py d1|d2 WORKDIR [NSM=128]"""
import glob, gzip, json, sys, numpy as np, statistics as st
_json_load = json.load
def _open(f): return gzip.open(f, 'rt') if f.endswith('.gz') else open(f)   # raw files may be gzipped (lossless, to save quota)
def _bin(f): return np.frombuffer(gzip.open(f).read() if f.endswith('.gz') else open(f, 'rb').read(), np.uint32).reshape(-1, 4)
which, W = sys.argv[1], sys.argv[2]; NSM = int(sys.argv[3]) if len(sys.argv) > 3 else 128
def load(pat): return {f: json.load(_open(f)) for f in sorted(glob.glob(W + '/' + pat + '_2*.json*'))}
def sat_cycles(j):
    s = j['sat']; ms = st.median(s['ms']); nw = s['ncalw']; c = np.array(s['cal'], dtype=np.float64).reshape(-1, 4)
    f = np.median((c[:, 1] - c[:, 0]) / (c[:, 3] - c[:, 2]))   # GHz = cycles per ns over one warp's lifetime (every warp)
    return ms, f, ms * 1e-3 * f * 1e9 * NSM / s['n']
def warp_cycles(j):
    return [st.median(w['cyc'][1:]) for w in j['warp']]
def fit(x, y):
    A = np.vstack([x, np.ones(len(x))]).T; (b, a), res, *_ = np.linalg.lstsq(A, y, rcond=None); yp = A @ [b, a]
    r2 = 1 - ((y - yp) ** 2).sum() / ((y - np.mean(y)) ** 2).sum(); return b, a, r2
if which in ('d1', 'd1b'):
    ns = [1, 4, 16, 64, 256]; arms = ['sph', 'custom', 'tri0', 'tri1']; one = {}; sat = {}; freq = {}
    for k in arms:
        for n in ns:
            j = json.load(_open(glob.glob('%s/%s_%s_n%d_2*.json*' % (W, which, k, n))[0])); one[k, n] = warp_cycles(j)[0]; ms, f, c = sat_cycles(j); sat[k, n] = c; freq[k, n] = f
    for n in (ns if which == 'd1' else []):
        a = _bin(glob.glob('%s/d1_count_n%d_2*.bin*' % (W, n))[0]); print('count arm N=%d: calls per ray min %d max %d (hits %d)' % (n, a[:, 3].min(), a[:, 3].max(), a[:, 0].sum()))
    for k in arms:
        for n in ns:
            a = _bin(glob.glob('%s/%s_%s_n%d_2*.bin*' % (W, which, k, n))[0])
            if a[:, 0].sum(): print('WARNING hits in', k, n, a[:, 0].sum())
    print('SM clock GHz (median over cells) %.3f' % np.median(list(freq.values())))
    print('\n one warp, cycles (identical lanes, median of 11 launches):'); print('arm      ' + ' '.join('%8d' % n for n in ns) + '    slope  intercept   R2')
    res = {}
    for k in arms:
        y = np.array([one[k, n] for n in ns]); b, a, r2 = fit(np.array(ns, float), y); res[('one', k)] = (b, a, r2)
        print('%-8s ' % k + ' '.join('%8.0f' % v for v in y) + '  %7.1f %9.0f %6.3f' % (b, a, r2))
    print('\n saturation, cycles per ray per SM:'); print('arm      ' + ' '.join('%8d' % n for n in ns) + '    slope  intercept   R2')
    for k in arms:
        y = np.array([sat[k, n] for n in ns]); b, a, r2 = fit(np.array(ns, float), y); res[('sat', k)] = (b, a, r2)
        print('%-8s ' % k + ' '.join('%8.2f' % v for v in y) + '  %7.2f %9.2f %6.3f' % (b, a, r2))
    print('\nratios of slopes: one warp C/S %.2f C/T0 %.2f C/T1 %.2f ; saturation C/S %.2f C/T0 %.2f C/T1 %.2f' % (
        res[('one', 'custom')][0] / res[('one', 'sph')][0], res[('one', 'custom')][0] / res[('one', 'tri0')][0], res[('one', 'custom')][0] / res[('one', 'tri1')][0],
        res[('sat', 'custom')][0] / res[('sat', 'sph')][0], res[('sat', 'custom')][0] / res[('sat', 'tri0')][0], res[('sat', 'custom')][0] / res[('sat', 'tri1')][0]))
    print('GAS bytes N=256: ' + ', '.join('%s %d' % (k, json.load(_open(glob.glob('%s/%s_%s_n256_2*.json*' % (W, which, k))[0]))['gas_bytes']) for k in arms))
if which == 'd2':
    arms = ['sph', 'custom', 'tri0', 'tri1']; rs = ['0.02', '0.04', '0.08']; B = lambda k, r: _bin(glob.glob('%s/d2_%s_r%s_2*.bin*' % (W, k, r))[0])
    print('r      mean IS calls (count arm)  min..max   hit fraction sph / custom / tri0 / tri1')
    for r in rs:
        c = B('count', r); print('%s  %.2f  %d..%d   ' % (r, c[:, 3].mean(), c[:, 3].min(), c[:, 3].max()) + ' / '.join('%.4f' % B(k, r)[:, 0].mean() for k in arms))
    print('\ncorrectness (vs built in sphere; first 1,048,576 rays): custom hit+sphere agree %, t within 1e-4 rel; tri arms: hit rate over rays the spheres hit, same sphere %, median |dt|/r')
    for r in rs:
        s = B('sph', r); c = B('custom', r); ok = (s[:, 0] == c[:, 0]) & ((s[:, 0] == 0) | (s[:, 2] == c[:, 2])); ts = s[:, 1].view(np.float32); tc = c[:, 1].view(np.float32); both = (s[:, 0] == 1) & (c[:, 0] == 1)
        print('r %s custom: agree %.5f, t rel within 1e-4 on %.5f of both-hit' % (r, ok.mean(), (np.abs(ts[both] - tc[both]) <= 1e-4 * np.abs(ts[both])).mean()))
        for k in ('tri0', 'tri1'):
            t = B(k, r); h = s[:, 0] == 1; tt = t[:, 1].view(np.float32); same = h & (t[:, 0] == 1) & (t[:, 2] == s[:, 2])
            print('   %s: hit rate on sphere hits %.4f, same sphere %.4f, median |dt|/r %.4f, extra hits where spheres miss %d' % (k, (t[h, 0] == 1).mean(), same[h].mean(), np.median(np.abs(tt[same] - ts[same])) / float(r), int(((s[:, 0] == 0) & (t[:, 0] == 1)).sum())))
    print('\nsaturation: cycles per ray per SM (SM clock GHz in brackets), GAS MB'); print('r      ' + '  '.join('%-14s' % k for k in arms))
    for r in rs:
        row = []
        for k in arms:
            j = json.load(_open(glob.glob('%s/d2_%s_r%s_2*.json*' % (W, k, r))[0])); ms, f, c = sat_cycles(j); row.append('%7.1f (%.2f) %5.1fMB' % (c, f, j['gas_bytes'] / 2 ** 20))
        print(r + '  ' + '  '.join(row))
