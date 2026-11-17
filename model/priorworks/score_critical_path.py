"""Score predictions_critical_path (commit d181c1f) against the F207 raw runs (notes/q33/raw, private). Same measured quantities as compare_priorworks.py (medians).
Usage: /usr/bin/python3 model/priorworks/score_critical_path.py > model/priorworks/results_critical_path.md"""
import re, glob, json, numpy as np
R = '/home/min/a/cashman3/eoss/notes/q33/raw/'; M = '/home/min/a/cashman3/eoss/model/priorworks/'
cp = json.load(open(M + 'pred_cp.json')); old = json.load(open(M + 'pred.json'))
med = lambda x: float(np.median(x)) if len(x) else float('nan'); lg = lambda m, p: abs(np.log(m / p))
print('# Prior works: critical path bound scored (2026-11-17, triton GPU 1)\n\nRatio = measured / predicted, closes when 0.5 to 2. "F207" = throughput form alone. Predictions committed in d181c1f before this script was run.\n')
err = {'rtnn': ([], []), 'trueknn': ([], []), 'arkade': ([], [])}
print('## RTNN\n\n| dataset | r | measured | F207 | ratio | critical path | ratio |\n|---|---|---|---|---|---|---|')
for k, v in cp.items():
    if not k.startswith('rtnn'): continue
    _, ds, r = k.split('_'); m = med([float(re.search(r'time search compute: (\S+)', open(f).read())[1]) for f in glob.glob(R + 'rtnn_%s_%s_*.txt' % (ds, r))])
    print('| %s | %.5g | %.3f | %.3f | %.2f | %.3f | %.2f |' % (ds, old[k]['r'], m, v['thr'], m / v['thr'], v['ms'], m / v['ms'])); err['rtnn'][0].append(lg(m, v['thr'])); err['rtnn'][1].append(lg(m, v['ms']))
print('\n## TrueKNN (fixed build, launches = sum of per round GT spans)\n\n| dataset | k | measured | F207 | ratio | critical path | ratio | rounds where chain binds |\n|---|---|---|---|---|---|---|---|')
for k, v in cp.items():
    if not k.startswith('trueknn'): continue
    _, ds, kk = k.split('_'); L = []
    for f in glob.glob(R + 'tk_%s_%s_fixed_*.txt' % (ds, kk)):
        g = re.findall(r'GT span_ms (\S+)', open(f).read())
        if g: L.append(sum(map(float, g)))
    m = med(L); print('| %s | %s | %.2f | %.2f | %.2f | %.2f | %.2f | %d |' % (ds, kk[1:], m, v['thr'], m / v['thr'], v['ms'], m / v['ms'], v['bind'])); err['trueknn'][0].append(lg(m, v['thr'])); err['trueknn'][1].append(lg(m, v['ms']))
print('\n## Arkade (event time of the launch; rows equal F207 by construction)\n\n| dataset | variant | measured | prediction | ratio |\n|---|---|---|---|---|')
for k, v in cp.items():
    if not k.startswith('arkade'): continue
    ds = k.split('_')[1]
    for var, key in (('fixed', 'fixed'), ('orig', 'pinned')):
        m = med([float(re.search(r'KERNEL queries \d+ ms (\S+)', open(f).read())[1]) for f in glob.glob(R + 'ark_%s_%s_*.txt' % (ds, var))])
        print('| %s | %s | %.3f | %.3f | %.2f |' % (ds, var, m, v[key], m / v[key])); err['arkade'][0].append(lg(m, v[key])); err['arkade'][1].append(lg(m, v[key]))
print('\nMedian absolute log error (F207 / critical path): ' + ', '.join('%s %.2f / %.2f' % (a, med(b[0]), med(b[1])) for a, b in err.items()) + '.')
