"""Compare model/predictions_trueknn2.md (second attempt) with the held out runs (notes/q20/tk/N_*.txt). Prints a markdown table; error = log(predicted / measured). Nothing refitted."""
import re, math, statistics as st
pred = {}; cur = None
for l in open('model/predictions_trueknn2.md'):
    m = re.match(r'## (\d+) points', l)
    if m: cur = int(m[1]); continue
    m = re.match(r'\| (\d) \| (\S+) \| (\d+) \| (\d+) \| (\S+) \| (\S+) \|$', l)
    if cur and m: pred.setdefault(cur, {})[int(m[1])] = (int(m[3]), float(m[4]), float(m[6]))
for n in (200000, 400000):
    runs = []
    for r in (1, 2, 3):
        d = {}
        for l in open('notes/q20/tk/%d_%d.txt' % (n, r)):
            m = re.match(r'RND (\d+) radius \S+ GT span_ms (\S+) CBK warps \d+ total_warp_cycles (\d+) f0_cycles (\d+) f0_calls (\d+)', l)
            if m: d[int(m[1])] = [float(m[2]), float(m[4]), float(m[5])]
            m = re.match(r'RND (\d+) points left (\d+)', l)
            if m: d[('l', int(m[1]))] = int(m[2])
        runs.append(d)
    med = {k: (st.median(r[k] for r in runs) if not isinstance(runs[0][k], list) else [st.median(r[k][i] for r in runs) for i in range(3)]) for k in runs[0]}
    print('## %d points: measured (median of 3) against predicted\n' % n)
    print('| round | active pred / meas | hits pred / warp calls meas | cycles per call (meas) | launch ms pred | launch ms meas | log error |\n|---|---|---|---|---|---|---|')
    tp = tm = 0; nr = max(k for k in med if not isinstance(k, tuple))
    for rd in range(1, nr + 1):
        a = med[('l', rd - 1)] if rd > 1 else n; m = med[rd]; p = pred[n].get(rd)
        if p: tp += p[2]
        tm += m[0]
        print('| %d | %s / %d | %s / %.0f | %.0f | %s | %.4f | %s |' % (rd, p[0] if p else '-', a, '%.0f' % p[1] if p else '-', m[2], m[1] / m[2], '%.4f' % p[2] if p else '-', m[0], '%.2f' % math.log(p[2] / m[0]) if p else '-'))
    print('| total | | | | %.4f | %.4f | %.2f |\n' % (tp, tm, math.log(tp / tm)))
    print('rounds predicted %d measured %d\n' % (len(pred[n]), nr))
