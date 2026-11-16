"""TrueKNN with the per round callback cost GIVEN (measured in the same runs): CALLBACK COST GIVEN, NOT A PREDICTION.
Model of F156 with its pooled c replaced by the measured per round c_r (callback cycles per warp call of that round, median of the runs at that size):
  T_r = calls_r x c_r x g / (142 x W x clk),  calls_r = rho x exact warp max box count (CPU, as F156),  rho, g, W from the rebuilt 100,000 point set A (5 runs, pooled over rounds).
Second variant: calls_r also taken from the measurement (isolates the call count error from the rest).
Data: notes/q21/tk_new/{A,200000,400000}_<stamp>_<i>.txt (private, timestamped raw), points regenerated as u<n>.csv from numpy default_rng(seed 1, 2, 3) as in model/q21.
    /usr/bin/python3 model/q22/trueknn_given.py STAMP > model/results_trueknn_given.md"""
import re, sys, glob, math, statistics as st, numpy as np
sys.path.insert(0, 'model/q21'); from trueknn_rule2 import feats
SM = 142; CLK = 2.5e9; stamp = sys.argv[1]
def load(tag):
    runs = []
    for f in sorted(glob.glob('notes/q21/tk_new/%s_%s_*.txt' % (tag, stamp))):
        d = {}
        for l in open(f):
            m = re.match(r'RND (\d+) radius (\S+) GT span_ms (\S+) CBK warps (\d+) total_warp_cycles (\d+) f0_cycles (\d+) f0_calls (\d+)', l)
            if m: d[int(m[1])] = dict(span=float(m[3]), tot=float(m[5]), cyc=float(m[6]), calls=float(m[7]), radius=float(m[2]))
            m = re.match(r'RND (\d+) points left (\d+)', l)
            if m: d[('left', int(m[1]))] = int(m[2])
        runs.append(d)
    return runs
def med(runs, r, key): return st.median([d[r][key] for d in runs if r in d])
A = load('A'); fA = feats(100000, 1); nA = len(fA)
calls_m = sum(med(A, r, 'calls') for r in range(1, nA + 1)); rho = calls_m / sum(f['calls'] for f in fA)
cyc = sum(med(A, r, 'cyc') for r in range(1, nA + 1)); g = sum(med(A, r, 'tot') for r in range(1, nA + 1)) / cyc
W = sum(med(A, r, 'tot') for r in range(1, nA + 1)) / (SM * CLK * sum(med(A, r, 'span') for r in range(1, nA + 1)) * 1e-3)
print('# TrueKNN, callback cost GIVEN (measured per round in the same runs): NOT A PREDICTION\n')
print('2026-11-16, triton GPU 1, rebuilt program (Arkade OWL tree port of s01-trueknn, /dev/shm/tk, third party code not committed). Raw runs stamp %s (private, notes/q21/tk_new). Points: uniform random in the unit cube, regenerated (numpy seeds 1, 2, 3), so they are NOT the F156 point files; k = 16, start radius 0.004, radius doubling, refit between rounds, one launch per round, 5 runs of A and 3 runs of each held size.\n' % stamp)
print('Constants from A (rebuilt, 5 runs, medians, pooled over rounds): rho (hardware warp calls over exact warp max box count) %.3f, g (warp cycles over callback cycles) %.3f, W (warps in flight per SM) %.2f. Per round callback cost c_r is GIVEN: callback cycles / warp calls of that round, median of 3 runs at that size. Model: T_r = rho x exact calls_r x c_r x g / (142 W clk). Second column group: calls_r taken from the measurement.\n' % (rho, g, W))
print('A per round, rebuilt vs the F156 build (medians of 5): span ms rebuilt %s' % ' '.join('%.3f' % med(A, r, 'span') for r in range(1, nA + 1)))
old = {}
for fn in glob.glob('notes/q20/tk/A_*.txt'):
    for l in open(fn):
        m = re.match(r'RND (\d+) radius (\S+) GT span_ms (\S+)', l)
        if m: old.setdefault(int(m[1]), []).append(float(m[3]))
print('F156 build: %s; callback cycles per call rebuilt %s' % (' '.join('%.3f' % st.median(old[r]) for r in sorted(old)), ' '.join('%.0f' % (med(A, r, 'cyc') / med(A, r, 'calls')) for r in range(1, nA + 1))))
print('Per run spread of the round spans in A (min to max over 5 runs): %s\n' % ' '.join('%.3f-%.3f' % (min(d[r]['span'] for d in A), max(d[r]['span'] for d in A)) for r in range(1, nA + 1)))
def table(n, seed, nm):
    R = load(str(n)); f = feats(n, seed); nr = max(max(k for k in d if not isinstance(k, tuple)) for d in R)
    print('## %d points (%d runs)\n' % (n, len(R)))
    print('| round | radius | active pred | active meas | calls pred | calls meas | c given (cycles per call) | T pred (calls pred) ms | T pred (calls meas) ms | T meas ms (min to max) | log err | log err (calls meas) |\n|---|---|---|---|---|---|---|---|---|---|---|---|')
    tp = tpm = tm = 0; e1 = []; e2 = []
    for r in range(1, max(nr, len(f)) + 1):
        pr = f[r - 1] if r <= len(f) else None
        have = [d for d in R if r in d]
        if not have:
            print('| %d | | %s | 0 | %s | | | | | not measured | | |' % (r, pr['act'] if pr else '', '%.0f' % (pr['calls'] * rho) if pr else '')); 
            if pr: tp += 0
            continue
        c = st.median([d[r]['cyc'] / d[r]['calls'] for d in have]); cm = med(R, r, 'calls'); spans = [d[r]['span'] for d in have]; m = st.median(spans)
        act_m = med(R, ('left', r - 1), 0) if False else (st.median([d[('left', r - 1)] for d in have]) if r > 1 else n)
        if pr:
            callp = pr['calls'] * rho; T = callp * c * g / (SM * W * CLK) * 1e3; Tm = cm * c * g / (SM * W * CLK) * 1e3
            tp += T; tpm += Tm; tm += m; e1.append(math.log(T / m)); e2.append(math.log(Tm / m))
            print('| %d | %g | %d | %d | %.0f | %.0f | %.0f | %.4f | %.4f | %.4f (%.4f to %.4f) | %+.2f | %+.2f |' % (r, pr['r'], pr['act'], act_m, callp, cm, c, T, Tm, m, min(spans), max(spans), math.log(T / m), math.log(Tm / m)))
        else:
            tm += m; print('| %d | %g | not predicted | %d | | %.0f | %.0f | | | %.4f | | |' % (r, med(R, r, 'radius'), act_m, cm, c, m))
    print('| total | | | | | | | %.4f | %.4f | %.4f | %+.2f | %+.2f |' % (tp, tpm, tm, math.log(tp / tm), math.log(tpm / tm)))
    print('\nImplied per round (medians of the runs), pooled A values in brackets: W = total warp cycles / (142 clk span) %s [%.1f]; g = total warp cycles / callback cycles %s [%.2f].' % (' '.join('%.1f' % (med(R, r, 'tot') / (SM * CLK * med(R, r, 'span') * 1e-3)) for r in range(1, nr + 1)), W, ' '.join('%.2f' % (med(R, r, 'tot') / med(R, r, 'cyc')) for r in range(1, nr + 1)), g))
    print('\nrounds predicted %d, measured %d; rms per round log error %.2f (calls predicted), %.2f (calls measured). F156 total log error for this size: %s; F158: %s.\n' % (len(f), nr, math.sqrt(sum(x * x for x in e1) / len(e1)), math.sqrt(sum(x * x for x in e2) / len(e2)), {200000: '-0.64', 400000: '-0.16'}[n], {200000: '+0.44', 400000: '+0.82'}[n]))
table(200000, 2, 'B'); table(400000, 3, 'C')
print('Reading: the cost per call is given, so this checks only the rest of the model (calls, g, W). It does not predict the cost per call. The active set and call counts come from the CPU box rule as in F156. The small totals with predicted calls are partly a cancellation: the call count is under predicted by 1.3 to 1.4 times in the big rounds, and with the measured call count the same model is 1.3 to 1.5 times too high (+0.28, +0.41). The last round (1 or 142 active queries) is a latency bound chain that a throughput model cannot give (0.55 to 0.60 ms measured). The rebuilt A differs from the F156 build in rounds 1 to 3 (see above), so these numbers are not a like for like comparison with F156 or F158.')
