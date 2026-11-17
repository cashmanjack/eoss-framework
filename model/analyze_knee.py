"""Knee map results vs model/predictions_knee.md. /usr/bin/python3 model/analyze_knee.py notes/unit3g/k_summary_<TS>.json notes/unit3g/knee_inputs.json notes/unit3g/knee_pred.json"""
import json, sys
import numpy as np
sys.argv, args = [sys.argv[0], sys.argv[2]], sys.argv[1:]
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    sys.argv = [sys.argv[0], args[1]]
    import importlib.util
    spec = importlib.util.spec_from_file_location('pk', 'model/predict_knee.py')
    # reuse helpers without rerunning the calibration output file: execute with the output json guarded
    src = open('model/predict_knee.py').read().split("json.dump(")[0]; ns = {}; exec(compile(src, 'pk', 'exec'), ns)
knee, D_at, STAT, G, INP = ns['knee'], ns['D_at'], ns['STAT'], ns['G'], ns['INP']
res = json.load(open(args[0])); pred = json.load(open(args[2]))['pred']; M = {(r['size'], r['rays'], r['w']): r['c'] for r in res}
SIZES = (1000, 10000, 100000, 1000000); WS = (4, 8, 16, 32)
meas = {}
print('measured c(g), g = 1 2 4 8 16 32, and g_knee (None = no knee: top / floor < 1.5)')
for n in SIZES:
    for W in WS:
        c = [M[(n, 'cls%d' % g, W)] for g in (1, 2, 4, 8, 16, 32)]; meas[(n, W)] = knee(c)
        print('%8d W%-2d | %s | top/floor %.1f | knee %s' % (n, W, ' '.join('%6.1f' % x for x in c), c[0] / c[-1], 'none' if meas[(n, W)] is None else '%.2f' % meas[(n, W)]))
print('\nscore: within a factor 2 of the measured knee (or both none), 15 configurations (not soup:100,000 W=32)')
for k in STAT:
    hit = 0; rows = []
    for n in SIZES:
        for i, W in enumerate(WS):
            if (n, W) == (100000, 32): continue
            p = pred['%s|%d' % (k, n)][i]; p = None if p in (0.0, 99.0) else p; m = meas[(n, W)]
            ok = (p is None and m is None) or (p is not None and m is not None and 0.5 <= p / m <= 2); hit += ok; rows.append(ok)
    m32 = [meas[(n, 32)] for n in SIZES if meas[(n, 32)]]; p32 = [pred['%s|%d' % (k, n)][3] for n in SIZES if pred['%s|%d' % (k, n)][3] not in (0.0, 99.0)]
    trend_ok = not (max(m32) / min(m32) > 2 and (not p32 or max(p32) / min(p32) < 1.5))
    print('%-14s %2d of 15 within a factor 2; size trend rule %s; rule 1+2 %s' % (k, hit, 'ok' if trend_ok else 'FAILS', 'PASSES' if hit >= 12 and trend_ok else 'fails'))
print('\nwhat the knee tracks (post hoc): log2 D at the measured knee over all configurations with a knee')
for k, (s, x) in STAT.items():
    v = [np.log2(D_at(n, s, meas[(n, W)], W, x)) for n in SIZES for W in WS if meas[(n, W)]]
    print('%-14s n=%d mean log2 D %.2f spread (sd) %.2f' % (k, len(v), np.mean(v), np.std(v)))
