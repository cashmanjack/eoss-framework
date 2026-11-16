"""Queue 15: hand off term (form: model/q15/FORM_handoff.md, committed before this ran). CPU only.
Run: /usr/bin/python3 -I model/q15/handoff.py   writes model/results_handoff_term.md"""
import os, sys, re, io, glob, contextlib, statistics as st
import numpy as np
from scipy.optimize import least_squares
H = os.path.expanduser('~/eoss'); sys.path.insert(0, H + '/model/q14'); sys.path.insert(0, H + '/model'); os.chdir(H)
with contextlib.redirect_stdout(io.StringIO()): import cb_latency as CL
R, GHZ, NSM, rms, byname = CL.R, CL.GHZ, CL.NSM, CL.rms, CL.byname
HC = 875.0
L = []
def P(*a):
    t = ' '.join(str(x) for x in a); print(t); L.append(t)
# candidate (lane calls) and lane efficiency per workload: the replay values, written before the measurement where they exist
ARK = {'ark_a1e5_0.005': (10.91, 0.581), 'ark_a1e5_0.0112': (54.02, 0.759), 'ark_a1e6_0.0016': (10.62, 0.507)}
def knn_ce(rd):
    t = open(H + '/notes/q14/logs/knn_off_1.log').read(); m = re.search(r'ROUND %d active \d+ radius \S+ KERNEL_MS \S+ cand_per_query (\S+) lane_efficiency (\S+)' % rd, t); return float(m.group(1)), float(m.group(2))
for x in R:
    w, n = x['wl'], x['name']
    if w == 'rtnn': c = tuple([n.split('_')[1], int(n.split('_')[2][1:]), n.split('_')[3]]); d = CL.RR.data[c]; ce = (d['cand'], d['eff'])
    elif w == 'knn': ce = knn_ce(int(n[-1]))
    elif w == 'ark': ce = ARK[n]
    else: continue
    x['cand'], x['eff'] = ce; x['lanes'] = x['N'] * ce[0]; x['Np'] = x['lanes'] / (32 * ce[1]); x['Nm'] = x['N_ld']
    ns = min(NSM, x['warps']); W = x['W']
    x['T_h_meas'] = HC * x['Nm'] / (ns * W); x['T_h_pred'] = HC * x['Np'] / (ns * W); x['T_h_lane'] = HC * x['lanes'] / (ns * W)
    f = x['T_floor']; x['T_fh'] = f + x['T_h_meas']; x['T_fhp'] = f + x['T_h_pred']; x['T_fhl'] = f + x['T_h_lane']; x['T_fhmax'] = max(f, x['T_h_meas'])
P('## Hand off term per launch (H = 875 cycles, not fitted)')
P('| workload | warp calls: replay prediction | probe count | lane candidates | W | hand off term ms (per call, measured N) | per lane | floor ms (F131) | callbacks measured ms | hand off share of callback busy time |'); P('|---|---|---|---|---|---|---|---|---|---|')
for x in R:
    if 'Nm' not in x or x['name'] == 'knn_round5': continue
    meas = x['share'][0] * x['ms_on']; g = lambda T: T / GHZ * 1e3
    P('| %s | %.0f | %.0f | %.0f | %.1f | %.3f | %.3f | %.3f | %.3f | %.0f%% |' % (x['name'], x['Np'], x['Nm'], x['lanes'], x['W'], g(x['T_h_meas']), g(x['T_h_lane']), g(x['T_floor']), meas, 100 * g(x['T_h_meas']) / meas))
# ---------------- RTNN launch time, same splits as F131 / q14
keys = [('T_floor', 'before: floor only (F131 form)'), ('T_fh', 'after: floor + hand off, per warp call, probe N'), ('T_fhp', 'after: floor + hand off, per warp call, replay predicted N'), ('T_fhl', 'after: floor + hand off, per lane candidate'), ('T_fhmax', 'sensitivity: max(floor, hand off)')]
res = {}; c0s = {}
P('\n## Launch time, RTNN (12 configurations; c0 fitted on the other radii of the same size, held out; as in q14)')
for key, nm in keys:
    ab, ratio = [], []
    for s, tr, te in CL.splits:
        trc = [(s, r, o) for r in tr for o in ('sorted', 'unsorted')]; tec = [(s, te, o) for o in ('sorted', 'unsorted')]
        f = least_squares(lambda p: CL.rt_err(trc, p, key), [100], bounds=(0, np.inf)).x; ab.append(rms(CL.rt_err(tec, f, key)))
        pr = lambda c: f[0] + CL.rt_cb(c, key) + CL.RR.data[c]['miss']; ratio.append(abs(np.log(pr((s, te, 'unsorted')) / pr((s, te, 'sorted')) / (CL.rt_meas((s, te, 'unsorted')) / CL.rt_meas((s, te, 'sorted'))))))
    tf = []
    for a, b in (('u1e5', 'u1e6'), ('u1e6', 'u1e5')):
        trc = [c for c in CL.RT if c[0] == a]; tec = [c for c in CL.RT if c[0] == b]; f = least_squares(lambda p: CL.rt_err(trc, p, key), [100], bounds=(0, np.inf)).x; tf.append(rms(CL.rt_err(tec, f, key)))
    fall = least_squares(lambda p: CL.rt_err(CL.RT, p, key), [100], bounds=(0, np.inf)).x; res[key] = (np.mean(ab), np.mean(ratio), np.mean(tf), fall[0], rms(CL.rt_err(CL.RT, fall, key))); c0s[key] = fall[0]
P('| callback term | radius splits held out rms log error | sorted/unsorted ratio error | size transfers | c0 (all 12) | in sample |'); P('|---|---|---|---|---|---|')
for key, nm in keys: r = res[key]; P('| %s | %.2f | %.2f | %.2f | %.0f | %.2f |' % (nm, *r))
# ---------------- kNN rounds 1 to 4
P('\n## Launch time, growing radius kNN rounds 1 to 4 (c0 from the 12 RTNN fit of the same row; miss term as in q14)')
act = [200000, 200000, 200000, 197660, 83]; qpr = [1.134, 1.919, 8.145, 56.026]; bef = [0.187, 0.203, 0.284, 0.734]
P('| round | measured ms | ' + ' | '.join(k[0] for k in keys) + ' |'); P('|---|---|' + '---|' * len(keys))
E = {k[0]: [] for k in keys}
for i in range(4):
    x = byname['knn_round%d' % (i + 1)]; N = x['N']; miss = max(bef[i] * 1e-3 * 1.5e11 / act[i] - (110.0 + 4.9 * (qpr[i] + 1)), 0)
    m = x['ms_off']; row = []
    for key, _ in keys:
        p = (c0s[key] + x[key] * NSM / N + miss) * N / (NSM * GHZ) * 1e3; row.append(p); E[key].append(np.log(p / m))
    P('| %d | %.3f | ' % (i + 1, m) + ' | '.join('%.3f (%+.2f)' % (p, np.log(p / m)) for p in row) + ' |')
P('rounds 1 to 4 rms log error: ' + ', '.join('%s %.2f' % (k[1].split(':')[0] + ' ' + k[0], rms(E[k[0]])) for k in keys))
# ---------------- Arkade
P('\n## Launch time, Arkade (callback term alone; time = floor + hand off)')
P('| config | measured ms (probes off) | before: floor | err | + hand off (probe N) | err | + hand off (replay N) | err | + per lane | err | hand off alone |'); P('|---|---|---|---|---|---|---|---|---|---|---|')
g = lambda T: T / GHZ * 1e3; ea = {k[0]: [] for k in keys}
for x in R:
    if x['wl'] != 'ark': continue
    m = x['ms_off']; v = {k: g(x[k]) for k, _ in keys}
    for k in v: ea[k].append(np.log(v[k] / m))
    P('| %s | %.2f | %.3f | %+.2f | %.3f | %+.2f | %.3f | %+.2f | %.3f | %+.2f | %.3f |' % (x['name'], m, v['T_floor'], np.log(v['T_floor'] / m), v['T_fh'], np.log(v['T_fh'] / m), v['T_fhp'], np.log(v['T_fhp'] / m), v['T_fhl'], np.log(v['T_fhl'] / m), g(x['T_h_meas'])))
P('Arkade rms log error: ' + ', '.join('%s %.2f' % (k[0], rms(ea[k[0]])) for k in keys))
# ---------------- Arkade with the empty intersection program (100,000 points, r 0.005; probes on, 3 runs)
fs = sorted(glob.glob(H + '/notes/q14/logs/arkempty_a1e5_0.005_[0-9].log')); ms_e = st.median([float(re.search(r'KERNEL queries \d+ ms ([\d.]+)', open(f).read()).group(1)) for f in fs])
cal = [int(re.search(r'f0_calls (\d+)', open(f).read()).group(1)) for f in fs][0]; cyc = [int(re.search(r'f0_cycles (\d+)', open(f).read()).group(1)) for f in fs][0]
xa = byname['ark_a1e5_0.005']; th = g(HC * cal / (NSM * xa['W']))
P('\n## Arkade with an empty intersection program (100,000 points, r 0.005; probes on, median of 3 runs)')
P('measured launch %.1f ms; intersection program calls %d (full program %d); callback cycles per call %.0f; callback share of warp time %.0f%%' % (ms_e, cal, int(xa['N_ld']), cyc / cal, 100 * 0))
P('hand off term (H 875, W %.1f from the full program ncu run): %.3f ms; the program itself has no instructions to count (floor ~0): predicted %.3f ms against %.1f measured, log error %+.2f' % (xa['W'], th, th, ms_e, np.log(th / ms_e)))
P('time per call implied by the measured launch: %.0f SM cycles of round trip per warp call at W %.1f (against H 875): factor %.0f' % (ms_e * 1e-3 * GHZ * NSM * xa['W'] / cal, xa['W'], ms_e * 1e-3 * GHZ * NSM * xa['W'] / cal / HC))
open(H + '/model/results_handoff_term.md', 'w').write('# Hand off term for custom primitives (queue 15), generated by model/q15/handoff.py; form fixed in model/q15/FORM_handoff.md before running\n\n' + '\n'.join(L) + '\n')
