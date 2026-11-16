"""Queue 19: MVA with issue, L1, L2, DRAM stations (form and kill rule: model/q19/FORM_mva3.md, committed before this ran). CPU only.
Run: /usr/bin/python3 -I model/q19/mva3.py predict   -> model/predictions_mva3.md (inputs and predictions; measured loaded latency and launch times are NOT read)
     /usr/bin/python3 -I model/q19/mva3.py compare   -> model/results_mva3.md   (against measurement, no refit, kill rule applied)"""
import os, sys, re, io, csv, contextlib, statistics as st
import numpy as np
H = os.path.expanduser('~/eoss'); sys.path.insert(0, H + '/model/q14'); sys.path.insert(0, H + '/model'); os.chdir(H)
with contextlib.redirect_stdout(io.StringIO()): import cb_latency as CL
import callback_cost as CB, callback_counts as CC
MODE = sys.argv[1]; GHZ = 2.5e9; XTU = 0.14; SHARE = 1.1; D_DRAM = 60 / (504e9 / 32 / 2.5e9)
LG = H + '/notes/q15/logs/'; NC = H + '/notes/q15/ncu/'
L = []
def P(*a):
    t = ' '.join(str(x) for x in a); print(t); L.append(t)
def solve(W, stations, Z, nchains=2):   # same solver as model/q18/mva2.py
    K = len(stations); Q = np.full(K, W / (K + 1.0))
    for _ in range(100000):
        R = np.zeros(K)
        for k, (D, shared) in enumerate(stations):
            others = (nchains * Q[k] if shared else Q[k]); R[k] = D * (1.0 + others - Q[k] / W)
        X = W / (R.sum() + Z); Qn = X * R
        if np.max(np.abs(Qn - Q)) < 1e-10: Q = Qn; break
        Q = 0.5 * Q + 0.5 * Qn
    return X, R, Q
rho = [x['N_ld'] / (min(CL.NSM, x['warps']) * x['ms_off'] * 1e-3 * GHZ) for x in CL.R if x['wl'] in ('rtnn', 'bh') and x['ms_off'] == x['ms_off']]
D_H = 1.0 / max(rho)
# replay table, first (rand order) table of model/predictions_knn_4070ti.md: nodes per query and share of fetches beyond 480 lines per round
tab = open(H + '/model/predictions_knn_4070ti.md').read().split('# kNN')[1]
REPLAY = {int(m[1]): (float(m[2]), float(m[3])) for m in re.finditer(r'^\| (\d) \| \d+ \| [\d.]+ \| ([\d.]+) \| [\d.]+ \| [\d.]+ \| [\d.]+ \| ([\d.]+) \|', tab, re.M)}
def knn_inputs(v, r):
    ld = [open(LG + 'knnfix_%s_loaded_%d.log' % (v, i)).read() for i in (1, 2, 3)]; sm = [open(LG + 'knnfix_%s_small_%d.log' % (v, i)).read() for i in (1, 2, 3)]; off = [open(LG + 'knnfix_%s_off_%d.log' % (v, i)).read() for i in (1, 2, 3)]
    pc = lambda t, r_: re.search(r'CBK round %d warps \d+ total_warp_cycles (\d+) trace_warp_cycles (\d+) f0_cycles (\d+) f0_calls (\d+)' % r_, t)
    calls = st.median(int(pc(t, r).group(4)) for t in ld)
    e = st.median(float(re.search(r'ROUND %d active \d+ radius \S+ KERNEL_MS \S+ cand_per_query \S+ lane_efficiency (\S+)' % r, t).group(1)) for t in ld)
    Lu = st.median(int(pc(t, r).group(3)) / int(pc(t, r).group(4)) for t in sm); C = st.median(int(pc(t, r).group(2)) / int(pc(t, r).group(4)) for t in sm)
    N = [200000, 200000, 200000, 197660][r - 1]; warps = N / 32
    for row in csv.reader(open(NC + 'occ_%s_r%d.csv' % (v, r - 1))):
        if len(row) > 14 and row[12] == 'sm__warps_active.avg.per_cycle_active': W = float(row[14])
    d = NC + 'knn_%s_r%d' % (v, r - 1); fn, h, _ = CB.load(d); rw = CC.raw(d + '/raw.csv')
    iss = l1 = l2 = 0.0
    for x in fn:
        if x['calls'] <= 1.5 * warps: continue
        iss += x['calls'] * (x['ins'] - x.get('dp', 0)) * CB.C_I + x['calls'] * x.get('dp', 0) * CB.C_DP
        l1 += x['calls'] * (x['tags'] + x['lld'] + x['lst']) * CB.C_TAG
        l2 += x['calls'] * (1 - h) * x['l2g'] * CB.C_X
    f = CB.term(d, warps, 'max')['cycles'] * CL.NSM / calls     # F149's single station demand
    nodes, miss = REPLAY[r]; nf = nodes * N / SHARE / calls
    lts = float(rw['lts__t_sectors.sum'].replace(',', '')); lh = float(rw['lts__t_sector_hit_rate.pct'].replace(',', '')) / 100
    D = dict(i=iss / calls, l1=l1 / calls + nf * CB.C_TAG, l2=l2 / calls + nf * 4 * miss * CB.C_X, d=lts * (1 - lh) / calls * D_DRAM)
    meas = None
    if MODE == 'compare':
        meas = dict(L=st.median(int(pc(t, r).group(3)) / int(pc(t, r).group(4)) for t in ld), ms=st.median(float(re.search(r'ROUND %d active \d+ radius \S+ KERNEL_MS (\S+)' % r, t).group(1)) for t in off))
    return dict(name='knn %s round %d' % (v, r), calls=calls, e=e, W=W, f=f, Lu=Lu, C=C, nf=nf, D=D, h=h, lh=lh, meas=meas)
targets = [knn_inputs(v, r) for v in ('reg', 'orig') for r in (1, 2, 3, 4)]
def predict(x, split=True):
    D = x['D']; D_tu = 32.0 * x['e'] / XTU; sm = D['i'] + D['l1'] + D['l2'] + D['d']
    if split:
        st_ = [(D['i'], False), (D['l1'], False), (D['l2'], False), (D['d'], False), (D_tu, True), (D_H, False)]; Z = max(x['C'] - sm - D_tu - D_H, 0.0); nsm = 4
    else:
        st_ = [(x['f'], False), (D_tu, True), (D_H, False)]; Z = max(x['C'] - x['f'] - D_tu - D_H, 0.0); sm = x['f']; nsm = 1
    X, R, Q = solve(x['W'], st_, Z)
    Lp = R[:nsm].sum() + max(x['Lu'] - sm, 0.0); T = x['calls'] / (CL.NSM * X) / GHZ * 1e3
    return dict(L=Lp, T=T, util=[X * s[0] * (2 if s[1] else 1) for s in st_])
pr = [(x, predict(x, True), predict(x, False)) for x in targets]
if MODE == 'predict':
    P('# MVA with issue, L1, L2, DRAM stations: predictions (written before any comparison; form and kill rule in model/q19/FORM_mva3.md, committed first)')
    P('Constants: c_i 0.25, c_tag 1.00, c_x 1.90 (F131), lane sharing 1.1 (F123), DRAM 9.52 cycles per sector per SM (504 GB/s over 60 SMs), X_tu 0.14, hand off 1 / %.5f = %.0f cycles per call, clock 2.5 GHz.' % (1 / D_H, D_H))
    P('Per call demands in SM cycles (D_i issue, D_l1, D_l2, D_d DRAM); n_f = RT node fetches per call; h = L1 hit rate of global loads, L2h = L2 sector hit rate (ncu, same launch). F149 = the single station f of F149 with TU and hand off.\n')
    P('| launch | calls | e | W | n_f | h | L2h | D_i | D_l1 | D_l2 | D_d | f (F149) | C | L unl | split: L | split: T ms | F149: L | F149: T ms |'); P('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for x, p, q in pr:
        D = x['D']; P('| %s | %d | %.3f | %.1f | %.1f | %.3f | %.3f | %.0f | %.0f | %.0f | %.0f | %.0f | %.0f | %.0f | %.0f | %.3f | %.0f | %.3f |' % (x['name'], x['calls'], x['e'], x['W'], x['nf'], x['h'], x['lh'], D['i'], D['l1'], D['l2'], D['d'], x['f'], x['C'], x['Lu'], p['L'], p['T'], q['L'], q['T']))
    open(H + '/model/predictions_mva3.md', 'w').write('\n'.join(L) + '\n')
else:
    rms = lambda v: float(np.sqrt(np.mean(np.square(v))))
    P('# MVA with issue, L1, L2, DRAM stations against measurement (no refit; predictions committed first in model/predictions_mva3.md)')
    P('Error = log(predicted / measured). L = loaded callback latency per warp call (probe), T = launch time (probes off, CUDA events).\n')
    P('| launch | L measured | split L | err | F149 L | err | T measured ms | split T | err | F149 T | err |'); P('|---|---|---|---|---|---|---|---|---|---|---|')
    E = []
    for x, p, q in pr:
        m = x['meas']; e = [np.log(p['L'] / m['L']), np.log(q['L'] / m['L']), np.log(p['T'] / m['ms']), np.log(q['T'] / m['ms'])]; E.append(e)
        P('| %s | %.0f | %.0f | %+.2f | %.0f | %+.2f | %.3f | %.3f | %+.2f | %.3f | %+.2f |' % (x['name'], m['L'], p['L'], e[0], q['L'], e[1], m['ms'], p['T'], e[2], q['T'], e[3]))
    a = np.array(E)
    P('\nrms log error over the 8 kNN launches: split T %.3f (kill rule: dropped if above 0.3), F149 form T %.3f; split L %.3f, F149 form L %.3f.' % (rms(a[:, 2]), rms(a[:, 3]), rms(a[:, 0]), rms(a[:, 1])))
    P('KILL RULE: %s' % ('FIRED, MVA dropped' if rms(a[:, 2]) > 0.3 else 'not fired'))
    P('\nSpeedup of the register heap over the original (launch time ratio), rounds 1 to 4: | round | measured | split | F149 form |'); P('|---|---|---|---|')
    for r in range(1, 5):
        o = [t for t in pr if t[0]['name'] == 'knn orig round %d' % r][0]; g = [t for t in pr if t[0]['name'] == 'knn reg round %d' % r][0]
        P('| %d | %.2f | %.2f | %.2f |' % (r, o[0]['meas']['ms'] / g[0]['meas']['ms'], o[1]['T'] / g[1]['T'], o[2]['T'] / g[2]['T']))
    open(H + '/model/results_mva3.md', 'w').write('\n'.join(L) + '\n')
