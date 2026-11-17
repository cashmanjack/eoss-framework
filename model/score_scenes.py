"""Score the camera and bounce ray predictions (model/predictions_scenes.md) against the measured launches. CPU only.
    /usr/bin/python3 model/score_scenes.py          reads notes/hs2/f123_scenes_*.json (the one file) and the newest notes/hs2/measure_TAG_*.json per set; prints the tables used in model/results_scenes.md"""
import glob, json, numpy as np
ORD = ['shuffled', 'morton', 'oracle d1', 'oracle d2', 'oracle d3', 'oracle d4', 'oracle full']
P = json.load(open(glob.glob('notes/hs2/f123_scenes_*.json')[0])); tot = {'f123': [], 'm4': [], 'base': []}
print('| set | quantity | ' + ' | '.join(ORD) + ' | rms log error |'); print('|---|---|' + '---|' * (len(ORD) + 1))
for tag, p in P.items():
    f = sorted(glob.glob('notes/hs2/measure_%s_2*.json' % tag))[-1]; m = {r['order']: r for r in json.load(open(f))['results']}
    wrong = sum(r['wrong'] for r in m.values()); ms = {k: m[k]['ms'] for k in m}
    meas = [ms['natural'] / ms[k] for k in ORD]; a = [p['f123'][k] for k in ORD]; b = [p['memory4'][k] for k in ORD]
    e = lambda x: float(np.sqrt(np.mean([np.log(u / v) ** 2 for u, v in zip(x, meas)])))
    ef, em, eb = e(a), e(b), e([1.0] * len(ORD)); tot['f123'] += [np.log(u / v) for u, v in zip(a, meas)]; tot['m4'] += [np.log(u / v) for u, v in zip(b, meas)]; tot['base'] += [-np.log(v) for v in meas]
    print('| %s | measured | ' % tag + ' | '.join('%.3f' % x for x in meas) + ' | (wrong %d, natural %.3f ms) |' % (wrong, ms['natural']))
    print('| | F123 | ' + ' | '.join('%.3f' % x for x in a) + ' | %.3f |' % ef)
    print('| | Memory_4 | ' + ' | '.join('%.3f' % x for x in b) + ' | %.3f |' % em)
    print('| | predict 1.0 | ' + ' | '.join('1.000' for _ in ORD) + ' | %.3f |' % eb)
    print('| | F123 log error | ' + ' | '.join('%+.3f' % np.log(u / v) for u, v in zip(a, meas)) + ' | |')
print('pooled rms log error over %d points: F123 %.3f  Memory_4 %.3f  predict 1.0 %.3f' % (len(tot['f123']), *[float(np.sqrt(np.mean(np.square(tot[k])))) for k in ('f123', 'm4', 'base')]))
for kind in ('camera', 'bounce'):
    idx = [i for i, t in enumerate(P) for _ in ORD if kind in t]; sel = np.array([kind in t for t in P for _ in ORD])
    print('pooled %s (%d points): F123 %.3f  Memory_4 %.3f  predict 1.0 %.3f' % (kind, sel.sum(), *[float(np.sqrt(np.mean(np.square(np.array(tot[k])[sel])))) for k in ('f123', 'm4', 'base')]))
