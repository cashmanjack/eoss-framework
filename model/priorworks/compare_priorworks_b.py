"""Variant (a) and (b) against the F207 measurements (CPU only). Forms: model/priorworks/FORM_callback_priorworks.md (committed a97a3e1) and predict_priorworks_warp.py (6bb41d3).
Raw: notes/q33/raw (F207 runs), notes/q34/raw (callback calibration runs). Usage: python3 model/priorworks/compare_priorworks_b.py > model/priorworks/results_priorworks_diag.md"""
import re, glob, json, numpy as np
R33 = '/home/min/a/cashman3/eoss/notes/q33/raw/'; R34 = '/home/min/a/cashman3/eoss/notes/q34/raw/'; M = '/home/min/a/cashman3/eoss/model/priorworks/'
old = json.load(open(M + 'pred.json')); wp = json.load(open(M + 'pred_warp.json')); SM, CLK = 142, 2.5e9
def med(x): return float(np.median(x)) if len(x) else float('nan')
def lg(m, p): return abs(np.log(m / p))
def rtt(pat): return med([float(re.search(r'time search compute: (\S+)', open(f).read())[1]) for f in sorted(glob.glob(pat))])
out = []
print('# Prior works, variants (a) and (b) against the F207 measurements (2026-11-17, triton GPU 1, no refit)\n\nRatio = measured / predicted; a row closes when 0.5 to 2. "F207" = mean count form, "(a)" = per warp slowest lane, "(b)" = measured callback cost swapped in (forms committed before the runs).\n')
print('## RTNN\n\nStub = same traversal, intersection program returns at once. cb = callback cycles per candidate per SM = (full - stub) x 142 x 2.5e9 / (1.10 x cube candidates in total); (b) cq = 4.9 + cb - cb_ref.\n')
# reference: 100,000 uniform 3D points, r 0.0232: cube candidates per query by CPU
from scipy.spatial import cKDTree
u = np.loadtxt('/dev/shm/pw/data/uni3_ref.csv', delimiter=','); Cref = cKDTree(u).query_ball_point(u, 0.0232, p=np.inf, return_length=True, workers=-1).astype(float)
fr, sr = rtt(R34 + 'rtnn_uni3_ref_ref_full_*'), rtt(R34 + 'rtnn_uni3_ref_ref_stub_*'); cbref = (fr - sr) * 1e-3 * SM * CLK / (1.10 * Cref.sum())
print('Reference (uniform 100,000, r 0.0232, %.1f candidates per query): full %.4f ms, stub %.4f ms, cb_ref %.2f cycles per candidate.\n' % (Cref.mean(), fr, sr, cbref))
print('| dataset | r | measured ms | stub ms | cb | F207 pred | ratio | (a) pred | ratio | (b) pred | ratio | (b) with (a) counts | ratio |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|')
rt_err = {'F207': [], 'a': [], 'b': [], 'ab': []}
for k, v in old.items():
    if not k.startswith('rtnn'): continue
    _, ds, r = k.split('_'); m = rtt(R33 + 'rtnn_%s_%s_*' % (ds, r)); full = rtt(R34 + 'rtnn_%s_%s_full_*' % (ds, r)); stub = rtt(R34 + 'rtnn_%s_%s_stub_*' % (ds, r)); N = v['N']
    cb = (full - stub) * 1e-3 * SM * CLK / (1.10 * v['C'] * N) if False else (full - stub) * 1e-3 * SM * CLK / (N * 1.10 * v['C_eff_hw'] if 'C_eff_hw' in v else N * 1.10 * v['C'])
    qm = 1.10 * (50 * (v['C'] + 1) / (v['S'] + 1) if v['S'] > 50 else v['C']) + 1; qw = wp[k]['warp_q'] * 1.0
    ms = lambda q, cq: (110 + cq * q) * N / (SM * CLK) * 1e3
    pb = ms(qm, 4.9 + cb - cbref); pab = ms(qw, 4.9 + cb - cbref); ok = full > stub
    print('| %s | %.5g | %.3f | %.3f%s | %.2f | %.3f | %.2f | %.3f | %.2f | %.3f | %.2f | %.3f | %.2f |' % (ds, v['r'], m, stub, '' if ok else ' (stub slower: invalid)', cb, v['ms'], m / v['ms'], wp[k]['ms'], m / wp[k]['ms'], pb, m / pb, pab, m / pab))
    rt_err['F207'].append(lg(m, v['ms'])); rt_err['a'].append(lg(m, wp[k]['ms'])); rt_err['b'].append(lg(m, pb)); rt_err['ab'].append(lg(m, pab))
print('\nMedian abs log error: ' + ', '.join('%s %.2f' % (a, med(b)) for a, b in rt_err.items()) + '. Callback share of the measured time (full - stub over full): ' + ', '.join('%s %.0f%%' % (k.split('_', 1)[1], 100 * (rtt(R34 + 'rtnn_%s_%s_full_*' % tuple(k.split('_')[1:])) - rtt(R34 + 'rtnn_%s_%s_stub_*' % tuple(k.split('_')[1:]))) / rtt(R34 + 'rtnn_%s_%s_full_*' % tuple(k.split('_')[1:]))) for k in old if k.startswith('rtnn')) + '.\n')
print('## TrueKNN (fixed build, device buffer; the probes are always on; (b) uses the measured callback warp cycles of every round, F160 constants 1.485 and 12.8 in flight)\n')
print('| dataset | k | measured launches ms | F207 pred | ratio | (a) pred | ratio | (b) pred | ratio | callback share of warp cycles |\n|---|---|---|---|---|---|---|---|---|---|')
tk_err = {'F207': [], 'a': [], 'b': []}
for k, v in old.items():
    if not k.startswith('trueknn'): continue
    _, ds, kk = k.split('_'); T = []; B = []; sh = []
    for f in sorted(glob.glob(R33 + 'tk_%s_%s_fixed_*.txt' % (ds, kk))):
        rows = re.findall(r'GT span_ms (\S+) CBK warps \d+ total_warp_cycles (\S+) f0_cycles (\S+)', open(f).read())
        if not rows: continue
        T.append(sum(float(a) for a, b, c in rows)); B.append(sum(float(c) * 1.485 / (SM * 12.8) / CLK * 1e3 for a, b, c in rows)); sh.append(sum(float(c) for a, b, c in rows) / sum(float(b) for a, b, c in rows))
    m = med(T); pb = med(B); pa = wp[k]['ms']
    print('| %s | %d | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f |' % (ds, v['k'], m, v['ms'], m / v['ms'], pa, m / pa, pb, m / pb, med(sh)))
    tk_err['F207'].append(lg(m, v['ms'])); tk_err['a'].append(lg(m, pa)); tk_err['b'].append(lg(m, pb))
print('\nMedian abs log error: ' + ', '.join('%s %.2f' % (a, med(b)) for a, b in tk_err.items()) + '.\n')
print('## Arkade (probe builds; (b): T = T_ref x F_d / F_ref, F = callback warp cycles; ref = 100,000 uniform 2D points and queries, r 0.005, same variant)\n')
print('| dataset | variant | measured ms (probe run) | measured ms (F207, no probes) | F207 pred | ratio | (a) pred | ratio | (b) pred | ratio | callback share of warp cycles |\n|---|---|---|---|---|---|---|---|---|---|---|')
def ark(ds, var):
    f = sorted(glob.glob(R34 + 'ark_%s_%s_*.txt' % (ds, var)))[0]; s = open(f).read(); t = float(re.search(r'KERNEL queries \d+ ms (\S+)', s)[1]); c = re.search(r'total_warp_cycles (\S+) trace_warp_cycles (\S+) f0_cycles (\S+)', s); return t, float(c[3]), float(c[3]) / float(c[1])
ak_err = {'F207': [], 'a': [], 'b': []}
for k, v in old.items():
    if not k.startswith('arkade'): continue
    ds = k.split('_')[1]
    for var in ('fixed', 'orig'):
        t, F, share = ark(ds, var); tr, Fr, _ = ark('ref', var); pb = tr * F / Fr
        nop = [float(re.search(r'KERNEL queries \d+ ms (\S+)', open(f).read())[1]) for f in sorted(glob.glob(R33 + 'ark_%s_%s_*.txt' % (ds, var)))]
        p0 = v['fixed'] if var == 'fixed' else v['pinned']; pa = wp[k]['fixed'] if var == 'fixed' else wp[k]['pinned']
        print('| %s | %s | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f |' % (ds, var, t, med(nop), p0, t / p0, pa, t / pa, pb, t / pb, share))
        if var == 'fixed': ak_err['F207'].append(lg(t, p0)); ak_err['a'].append(lg(t, pa)); ak_err['b'].append(lg(t, pb))
print('\nMedian abs log error (fixed rows): ' + ', '.join('%s %.2f' % (a, med(b)) for a, b in ak_err.items()) + '.')
