"""Queue 18: MVA with traversal unit and hand off stations (form: model/q18/FORM_mva2.md, committed before this ran). CPU only.
Run: /usr/bin/python3 -I model/q18/mva2.py predict    writes model/predictions_mva.md (inputs and predictions only; the measured loaded latency and launch times are NOT read)
     /usr/bin/python3 -I model/q18/mva2.py compare    writes model/results_mva2.md   (predictions against measurements, no refit)"""
import os, sys, re, io, csv, contextlib, statistics as st
import numpy as np
H = os.path.expanduser('~/eoss'); sys.path.insert(0, H + '/model/q14'); sys.path.insert(0, H + '/model'); os.chdir(H)
with contextlib.redirect_stdout(io.StringIO()): import cb_latency as CL
import callback_cost as CB
MODE = sys.argv[1]; GHZ = 2.5e9; XTU = 0.14
LG = H + '/notes/q15/logs/'; NC = H + '/notes/q15/ncu/'; Q17 = H + '/notes/q17/'
L = []
def P(*a):
    t = ' '.join(str(x) for x in a); print(t); L.append(t)
# ---------- solver: Bard Schweitzer MVA, two symmetric chains (SMs of one TPC), continuous populations
def solve(W, stations, Z, nchains=2):
    """stations: list of (demand, shared) ; shared = visited by both chains, else the chain's own. Returns (X per chain, residence per station, queue lengths)"""
    K = len(stations); Q = np.full(K, W / (K + 1.0))   # per chain queue lengths (symmetric)
    for _ in range(100000):
        R = np.zeros(K)
        for k, (D, shared) in enumerate(stations):
            others = (nchains * Q[k] if shared else Q[k])      # sum over chains of Q_ck at the station seen by chain c
            R[k] = D * (1.0 + others - Q[k] / W)
        X = W / (R.sum() + Z); Qn = X * R
        if np.max(np.abs(Qn - Q)) < 1e-10: Q = Qn; break
        Q = 0.5 * Q + 0.5 * Qn
    return X, R, Q
# ---------- hand off capacity from the non target launches (F133 data: RTNN and RT-BarnesHut)
rho = [x['N_ld'] / (min(CL.NSM, x['warps']) * x['ms_off'] * 1e-3 * GHZ) for x in CL.R if x['wl'] in ('rtnn', 'bh') and x['ms_off'] == x['ms_off']]
RHO_MAX = max(rho); D_H = 1.0 / RHO_MAX
# ---------- inputs per target launch
def clog(path, key):
    return open(path).read()
def knn_inputs(v, r):
    ld = [open(LG + 'knnfix_%s_loaded_%d.log' % (v, i)).read() for i in (1, 2, 3)]; sm = [open(LG + 'knnfix_%s_small_%d.log' % (v, i)).read() for i in (1, 2, 3)]; off = [open(LG + 'knnfix_%s_off_%d.log' % (v, i)).read() for i in (1, 2, 3)]
    pc = lambda t, r_: re.search(r'CBK round %d warps \d+ total_warp_cycles (\d+) trace_warp_cycles (\d+) f0_cycles (\d+) f0_calls (\d+)' % r_, t)
    calls = st.median(int(pc(t, r).group(4)) for t in ld)
    e = st.median(float(re.search(r'ROUND %d active \d+ radius \S+ KERNEL_MS \S+ cand_per_query \S+ lane_efficiency (\S+)' % r, t).group(1)) for t in ld)
    Lu = st.median(int(pc(t, r).group(3)) / int(pc(t, r).group(4)) for t in sm); C = st.median(int(pc(t, r).group(2)) / int(pc(t, r).group(4)) for t in sm)
    N = [200000, 200000, 200000, 197660][r - 1]; warps = N / 32
    for row in csv.reader(open(NC + 'occ_%s_r%d.csv' % (v, r - 1))):
        if len(row) > 14 and row[12] == 'sm__warps_active.avg.per_cycle_active': W = float(row[14])
    fl = CB.term(NC + 'knn_%s_r%d' % (v, r - 1), warps, 'max')['cycles']; f = fl * CL.NSM / calls
    meas = None
    if MODE == 'compare':
        meas = dict(L=st.median(int(pc(t, r).group(3)) / int(pc(t, r).group(4)) for t in ld),
                    ms=st.median(float(re.search(r'ROUND %d active \d+ radius \S+ KERNEL_MS (\S+)' % r, t).group(1)) for t in off))
    return dict(name='knn %s round %d' % (v, r), kind='knn_' + v, calls=calls, e=e, W=W, f=f, Lu=Lu, C=C, nsm=CL.NSM, meas=meas)
ark = [x for x in CL.R if x['name'] == 'ark_a1e5_0.005'][0]
ark_sm = [r for r in ark['sm'] if r.get('total')]; ark_Lu = float(np.median([r['f0c'] / r['f0n'] for r in ark_sm])); ark_C = float(np.median([r['trace'] / r['f0n'] for r in ark_sm])); ark_f = ark['T_floor'] * CL.NSM / ark['N_ld']
def ark_inputs(host, nsm):
    t = open(Q17 + 'arkade_%s_full_device.txt' % host).read()
    cb = [(int(c), int(d)) for c, d in re.findall(r'f0_cycles (\d+) f0_calls (\d+)', t)]; gt = [float(x) for x in re.findall(r'GT span_ms ([\d.]+)', t)]
    meas = None
    if MODE == 'compare':   # measured values are only read by the compare step; triton's per call probe value is the 84,000 cycles of F143 / F144 (its file has no CBK line)
        meas = dict(L=st.median(a / b for a, b in cb) if cb else 84000.0, ms=st.median(gt))
    return dict(name='arkade device buffer, %s (%d SMs)' % (host, nsm), kind='ark', calls=105089, e=0.581, W=ark['W'], f=ark_f, Lu=ark_Lu, C=ark_C, nsm=nsm, meas=meas)
targets = [knn_inputs(v, r) for v in ('reg', 'orig') for r in (1, 2, 3, 4)] + [ark_inputs('io', 60), ark_inputs('triton', 142)]
def predict(x, full=True):
    D_tu = 32.0 * x['e'] / XTU; f = x['f']
    if full:
        stations = [(f, False), (D_tu, True), (D_H, False)]; Z = max(x['C'] - f - D_tu - D_H, 0.0)
    else:
        stations = [(f, False)]; Z = max(x['C'] - f, 0.0)
    X, R, Q = solve(x['W'], stations, Z)
    Lp = R[0] + max(x['Lu'] - f, 0.0)
    T = x['calls'] / (min(x['nsm'], 1e9) * X) / GHZ * 1e3
    util = dict(issue=X * f, tu=2 * X * D_tu, handoff=X * D_H) if full else dict(issue=X * f)
    return dict(L=Lp, T=T, X=X, util=util, Z=Z, D_tu=D_tu)
pr = [(x, predict(x, True), predict(x, False)) for x in targets]
if MODE == 'predict':
    P('# MVA with traversal unit and hand off stations: predictions (written before any comparison; form in model/q18/FORM_mva2.md, committed first)')
    P('Constants: X_tu = %.2f lanes per cycle per TPC (F105), hand off capacity 1 / %.5f = %.0f cycles per warp call per SM (highest call rate of the non target launches, an upper bound, not a saturation test), clock 2.5 GHz, issue multiplier 1.' % (XTU, RHO_MAX, D_H))
    P('Inputs per launch: calls = warp calls of the callback program, e = lane efficiency, W = warps in flight per SM (ncu), f = floor cycles per call (F131), L_unl and C = unloaded callback latency and trace window cycles per call (one warp per SM launch). Output: loaded latency per call L (cycles) and launch time T (ms).\n')
    P('| launch | calls | e | W | f | L unloaded | C (trace per call) | D_tu | issue only: L | issue only: T ms | full network: L | full network: T ms | utilisation issue / TU / hand off |'); P('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for x, p, q in pr:
        P('| %s | %d | %.3f | %.1f | %.0f | %.0f | %.0f | %.0f | %.0f | %.3f | %.0f | %.3f | %.2f / %.2f / %.2f |' % (x['name'], x['calls'], x['e'], x['W'], x['f'], x['Lu'], x['C'], p['D_tu'], q['L'], q['T'], p['L'], p['T'], p['util']['issue'], p['util']['tu'], p['util']['handoff']))
    open(H + '/model/predictions_mva.md', 'w').write('\n'.join(L) + '\n')
else:
    rms = lambda v: float(np.sqrt(np.mean(np.square(v))))
    P('# MVA with traversal unit and hand off stations against measurement (no refit; predictions committed first in model/predictions_mva.md)')
    P('Error = log(predicted / measured). L = loaded callback latency per warp call (probe), T = launch time (kNN: probes off, CUDA events; Arkade: global timer span).\n')
    P('| launch | L measured | L issue only | L full | err issue only | err full | T measured ms | T issue only | T full | err issue only | err full |'); P('|---|---|---|---|---|---|---|---|---|---|---|')
    E = {}
    for x, p, q in pr:
        m = x['meas']; e1 = [np.log(q['L'] / m['L']), np.log(p['L'] / m['L']), np.log(q['T'] / m['ms']), np.log(p['T'] / m['ms'])]; E.setdefault(x['kind'], []).append(e1)
        P('| %s | %.0f | %.0f | %.0f | %+.2f | %+.2f | %.3f | %.3f | %.3f | %+.2f | %+.2f |' % (x['name'], m['L'], q['L'], p['L'], e1[0], e1[1], m['ms'], q['T'], p['T'], e1[2], e1[3]))
    P('\nrms log error by kind: | kind | n | L issue only | L full | T issue only | T full |'); P('|---|---|---|---|---|---|')
    for k, v in E.items():
        a = np.array(v); P('| %s | %d | %.2f | %.2f | %.2f | %.2f |' % (k, len(v), rms(a[:, 0]), rms(a[:, 1]), rms(a[:, 2]), rms(a[:, 3])))
    # order check: speedup orig / reg
    P('\nSpeedup of the register heap over the original (launch time ratio), rounds 1 to 4:')
    P('| round | measured | issue only | full |'); P('|---|---|---|---|')
    for r in range(1, 5):
        o = [t for t in pr if t[0]['name'] == 'knn orig round %d' % r][0]; g = [t for t in pr if t[0]['name'] == 'knn reg round %d' % r][0]
        P('| %d | %.2f | %.2f | %.2f |' % (r, o[0]['meas']['ms'] / g[0]['meas']['ms'], o[2]['T'] / g[2]['T'], o[1]['T'] / g[1]['T']))
    open(H + '/model/results_mva2.md', 'w').write('\n'.join(L) + '\n')
