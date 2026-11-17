"""Compare the committed predictions (model/priorworks/pred.json, commit a926b7a / 66baba7) with the raw runs in notes/q33/raw (private, timestamped, never overwritten).
Measured quantities: RTNN 'time search compute' (sum of launches); TrueKNN sum of the global timer span of every round (launch only) and the sum of the program's per round wall time ('Round N Time');
Arkade GT span (first warp start to last warp end) and the CUDA event time of the launch (KERNEL). Median of the runs. Output: markdown on stdout.
    /usr/bin/python3 model/priorworks/compare_priorworks.py > model/priorworks/results_priorworks.md"""
import re, glob, json, os, numpy as np
R = '/home/min/a/cashman3/eoss/notes/q33/raw/'; P = json.load(open('/home/min/a/cashman3/eoss/model/priorworks/pred.json'))
def med(x): return float(np.median(x)) if len(x) else float('nan')
def rat(m, p): return m / p if p and m == m else float('nan')
print('# Prior works on their datasets: results against the committed predictions (2026-11-17, triton GPU 1)\n')
print('Ratio = measured / predicted (1 = exact; success rule set before: within a factor 2 = ratio 0.5 to 2). Medians of the runs, ms.\n')
print('## RTNN (one version: the output buffer is already device memory, the distance is already float)\n\n| dataset | r | predicted | measured (runs) | ratio |\n|---|---|---|---|---|')
for k, v in P.items():
    if not k.startswith('rtnn'): continue
    _, ds, r = k.split('_'); t = [float(re.search(r'time search compute: (\S+)', open(f).read())[1]) for f in sorted(glob.glob(R + 'rtnn_%s_%s_*.txt' % (ds, r)))]
    print('| %s | %.5g | %.3f | %.3f (%d) | %.2f |' % (ds, v['r'], v['ms'], med(t), len(t), rat(med(t), v['ms'])))
print('\n## TrueKNN (original = managed memory frame buffer, fixed = device buffer; launches only = sum of per round global timer spans; wall = sum of the program\'s per round times)\n\n| dataset | k | predicted rounds | measured rounds | predicted launches | fixed launches | ratio | original launches | original / fixed | fixed wall | original wall |\n|---|---|---|---|---|---|---|---|---|---|---|')
for k, v in P.items():
    if not k.startswith('trueknn'): continue
    _, ds, kk = k.split('_'); res = {}
    for var in ('fixed', 'orig'):
        L = []; W = []; nr = []
        for f in sorted(glob.glob(R + 'tk_%s_%s_%s_*.txt' % (ds, kk, var))):
            s = open(f).read(); g = re.findall(r'GT span_ms (\S+)', s)
            if not g: continue
            L.append(sum(map(float, g))); W.append(1e3 * sum(map(float, re.findall(r'Round: \d+ Time: (\S+) seconds', s)))); nr.append(len(g))
        res[var] = (med(L), med(W), med(nr), len(L))
    f, o = res['fixed'], res['orig']
    print('| %s | %d | %d | %s | %.2f | %.2f (%d) | %.2f | %.2f (%d) | %.1f | %.1f | %.1f |' % (ds, v['k'], v['rounds'], '%.0f' % f[2] if f[3] else 'none', v['ms'], f[0], f[3], rat(f[0], v['ms']), o[0], o[3], rat(o[0], f[0]), f[1], o[1]))
print('\n## Arkade (KN 50, NORM 2, 10,000 queries; fixed = device buffer + float square, original = pinned host buffer + double pow; CUDA event time of the launch (the GT lines are not printed with the probes off))\n\n| dataset | r | predicted fixed | measured fixed GT | ratio | predicted original pinned | measured original GT | ratio | original / fixed (GT) | event time original / fixed |\n|---|---|---|---|---|---|---|---|---|---|')
for k, v in P.items():
    if not k.startswith('arkade'): continue
    ds = k.split('_')[1]; res = {}
    for var in ('fixed', 'orig'):
        g = []; e = []
        for f in sorted(glob.glob(R + 'ark_%s_%s_*.txt' % (ds, var))):
            s = open(f).read(); b = re.search(r'KERNEL queries \d+ ms (\S+)', s)
            if b: g.append(float(b[1])); e.append(float(b[1]))
        res[var] = (med(g), med(e), len(g))
    f, o = res['fixed'], res['orig']
    print('| %s | %.5g | %.3f | %.3f (%d) | %.2f | %.3f | %.3f (%d) | %.2f | %.1f | %.1f |' % (ds, v['r'], v['fixed'], f[0], f[2], rat(f[0], v['fixed']), v['pinned'], o[0], o[2], rat(o[0], v['pinned']), rat(o[0], f[0]), rat(o[1], f[1])))
