"""Stack scene (one triangle candidates), 5090, OptiX only. python3 analyze_5090_batch9_stack.py WORKDIR NSM"""
import glob, json, sys, numpy as np, statistics as st
W = sys.argv[1]; NSM = int(sys.argv[2]); ns = [1, 4, 16, 64, 256, 1024, 4096]
one = {}; sat = {}; gas = {}; clk = {}
for n in ns:
    f = glob.glob('%s/stack_n%d_2*.json' % (W, n))[0]; j = json.load(open(f))
    one[n] = st.median(j['warp'][0]['cyc'][1:]); s = j['sat']; ms = st.median(s['ms']); c = np.array(s['cal'], float).reshape(-1, 4)
    clk[n] = np.median((c[:, 1] - c[:, 0]) / (c[:, 3] - c[:, 2])); sat[n] = ms * 1e-3 * clk[n] * 1e9 * NSM / s['n']; gas[n] = j['gas_bytes']
    a = np.fromfile(j['verify'], np.uint32).reshape(-1, 4)
    if a[:, 0].sum(): print('WARNING hits at N', n, a[:, 0].sum())
print('N        ' + ' '.join('%9d' % n for n in ns)); print('one warp ' + ' '.join('%9.0f' % one[n] for n in ns)); print('sat cyc/ray/SM ' + ' '.join('%8.2f' % sat[n] for n in ns)); print('clock GHz ' + ' '.join('%.3f' % clk[n] for n in ns)); print('gas bytes ' + ' '.join('%d' % gas[n] for n in ns))
def fit(xs, ys):
    A = np.vstack([xs, np.ones(len(xs))]).T; (b, a), *_ = np.linalg.lstsq(A, ys, rcond=None); yp = A @ [b, a]; return b, a, 1 - ((ys - yp) ** 2).sum() / ((ys - np.mean(ys)) ** 2).sum()
for lab, lo in (('all N', 1), ('N >= 64', 64), ('N >= 256', 256)):
    x = np.array([n for n in ns if n >= lo], float)
    for nm, d in (('one warp', one), ('saturation', sat)):
        b, a, r2 = fit(x, np.array([d[int(n)] for n in x])); print('%-9s %-10s slope %.3f intercept %.1f R2 %.4f' % (lab, nm, b, a, r2))
print('per step slopes sat:', ' '.join('%.3f' % ((sat[ns[k + 1]] - sat[ns[k]]) / (ns[k + 1] - ns[k])) for k in range(len(ns) - 1)))
print('per step slopes one warp:', ' '.join('%.2f' % ((one[ns[k + 1]] - one[ns[k]]) / (ns[k + 1] - ns[k])) for k in range(len(ns) - 1)))
