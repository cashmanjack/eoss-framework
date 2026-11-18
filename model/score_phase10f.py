"""Phase 10f item 3 scoring (predictions model/predictions_phase10f.md, committed 0d36050 before the runs). python3 model/score_phase10f.py > /dev/stdout ; reads notes/p10f/raw (private raw), medians over the repetitions, clocks from the nvidia-smi samples."""
import glob, re, statistics as st, numpy as np
R = '/home/min/a/cashman3/eoss/notes/p10f/raw/'
def files(n): return sorted(f for f in glob.glob(R + n + '_*.txt') if not f.endswith('.tk.txt'))
def med(x): return float(st.median(x))
def clk(n): return med([int(m) for f in glob.glob(R + n + '_*.clk') for m in re.findall(r', (\d+) MHz', open(f).read())])
rows = []
for n, p in (('rtnn_r05', 0.191), ('rtnn_r025', 0.058)):
    m = med([float(re.search(r'time search compute: (\S+)', open(f).read())[1]) for f in files(n)]); rows.append((n + ' search compute', m, p, clk(n)))
for n, P in (('knn_orig', [0.733, 1.086, 3.884, 12.22]), ('knn_reg', [0.183, 0.250, 0.767, 3.26])):
    L = [[float(x) for x in re.findall(r'LAUNCH round \d+ active \d+ ms (\S+)', open(f).read())][:4] for f in files(n)]; M = [med([l[i] for l in L]) for i in range(4)]
    for i in range(4): rows.append(('%s round %d' % (n, i + 1), M[i], P[i], clk(n)))
    rows.append((n + ' rounds 1 to 4 total', sum(M), sum(P), clk(n)))
    globals()[n] = sum(M)
rows.append(('register heap speedup (orig / reg, rounds 1 to 4)', knn_orig / knn_reg, 4.02, clk('knn_reg')))
rows.append(('arkade fixed 100k queries, GT span (committed basis F151 0.18 x 1.109)', med([float(re.search(r'GT span_ms (\S+)', open(f).read())[1]) for f in files('ark_100kq')]), 0.20, clk('ark_100kq')))
tk = [sum(float(x) for x in re.findall(r'^RND \d+ radius \S+ GT span_ms (\S+)', open(f).read(), re.M)) for f in files('tk400k')]; rows.append(('trueknn 400k total of round spans', med(tk), 12.95, clk('tk400k')))
print('| row | measured | predicted | measured / predicted | log error | inside factor 2 | SM clock MHz (median) |\n|---|---|---|---|---|---|---|')
le = []
for n, m, p, c in rows:
    r = m / p; le.append(abs(np.log(r))); print('| %s | %.4g | %.4g | %.2f | %+.2f | %s | %d |' % (n, m, p, r, np.log(r), 'yes' if 0.5 <= r <= 2 else 'NO', c))
print('\nmedian absolute log error over %d rows: %.2f; inside factor 2: %d of %d' % (len(le), med(le), sum(1 for x in le if x <= np.log(2) + 1e-9), len(le)))
ark10 = med([float(re.search(r'KERNEL queries \d+ ms (\S+)', open(f).read())[1]) for f in files('ark_10kq')]); ark_ev = med([float(re.search(r'KERNEL queries \d+ ms (\S+)', open(f).read())[1]) for f in files('ark_100kq')])
print('extra, unscored: arkade 10,000 queries CUDA event ms %.4f; 100,000 queries CUDA event ms %.4f; triton F207 ref fixed (pinned output) GT span 0.5714 ms x 1.109 = 0.634 (4090 measured %.4g)' % (ark10, ark_ev, rows[-2][1]))
print('trueknn per round spans (ms, median): ' + ' '.join('%.4g' % med([float(re.findall(r'^RND \d+ radius \S+ GT span_ms (\S+)', open(f).read(), re.M)[k]) for f in files('tk400k')]) for k in range(5)))
