"""Scores model/predictions_5090_batch5.md part B (commit a6923fd, before any run). python3 model/score_5090_workloads.py TS   (TS = timestamp of the 3 repetition run; raw in notes/pc14_workloads, private). Medians over repetitions; round 5 of kNN not predicted."""
import glob, re, sys, statistics as st, numpy as np
R = 'notes/pc14_workloads/'; TS = sys.argv[1]
def files(n): return sorted(f for f in glob.glob(R + n + '_' + TS + '_*.txt') if not f.endswith('.tk.txt'))
med = lambda x: float(st.median(x))
def clk(n): return med([int(m) for f in glob.glob(R + n + '_' + TS + '_*.clk') for m in re.findall(r', (\d+) MHz', open(f).read())])
rows = []
for n, p in (('rtnn_r05', 0.144), ('rtnn_r025', 0.044)):
    rows.append((n + ' search compute', med([float(re.search(r'time search compute: (\S+)', open(f).read())[1]) for f in files(n)]), p, None, clk(n)))
K = {'knn_orig': ([0.233, 0.346, 1.236, 3.889], [0.333, 0.476, 1.616, 5.012]), 'knn_reg': ([0.058, 0.079, 0.244, 1.038], [0.087, 0.111, 0.297, 1.193])}
for n, (P, Q) in K.items():
    L = [[float(x) for x in re.findall(r'LAUNCH round \d+ active \d+ ms (\S+)', open(f).read())] for f in files(n)]; M = [med([l[i] for l in L]) for i in range(4)]
    for i in range(4): rows.append(('%s round %d' % (n, i + 1), M[i], P[i], Q[i], clk(n)))
    rows.append((n + ' rounds 1 to 4 total', sum(M), sum(P), sum(Q), clk(n))); globals()[n] = sum(M)
    print(n, 'round 5 (active 66, not predicted) ms', med([l[4] for l in L]))
rows.append(('register heap speedup', knn_orig / knn_reg, 4.02, 4.41, clk('knn_reg')))
rows.append(('arkade fixed 100k queries GT span', med([float(re.search(r'GT span_ms (\S+)', open(f).read())[1]) for f in files('ark_100kq')]), 0.553, None, clk('ark_100kq')))
tk = [sum(float(x) for x in re.findall(r'^RND \d+ radius \S+ GT span_ms (\S+)', open(f).read(), re.M)) for f in files('tk400k')]; rows.append(('trueknn 400k total of round spans', med(tk), 10.02, None, clk('tk400k')))
print('| row | measured | v1 predicted | ratio | inside 2 | v1+F257 | ratio | inside 2 | SM MHz |\n|---|---|---|---|---|---|---|---|---|')
le = []
for n, m, p, q, c in rows:
    r = m / p; le.append(abs(np.log(r))); s = '| %s | %.4g | %.4g | %.2f | %s |' % (n, m, p, r, 'yes' if 0.5 <= r <= 2 else 'NO')
    s += ' %.4g | %.2f | %s |' % (q, m / q, 'yes' if 0.5 <= m / q <= 2 else 'NO') if q else ' | | |'
    print(s + ' %d |' % c)
print('\nv1 median abs log error over %d rows: %.2f; inside factor 2: %d' % (len(le), med(le), sum(x <= np.log(2) + 1e-9 for x in le)))
print('trueknn per round spans: ' + ' '.join('%.4g' % med([float(re.findall(r'^RND \d+ radius \S+ GT span_ms (\S+)', open(f).read(), re.M)[k]) for f in files('tk400k')]) for k in range(5)))
print('arkade 10k CUDA event ms %.4f, 100k %.4f' % (med([float(re.search(r'KERNEL queries \d+ ms (\S+)', open(f).read())[1]) for f in files('ark_10kq')]), med([float(re.search(r'KERNEL queries \d+ ms (\S+)', open(f).read())[1]) for f in files('ark_100kq')])))
