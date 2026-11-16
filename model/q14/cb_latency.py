"""Queue 14: callback latency probes -> time breakdown table and latency aware callback term (form: model/q14/FORM.md, fixed before this was run). CPU only.
Run: /usr/bin/python3 -I model/q14/cb_latency.py    (reads notes/q14/logs, notes/q14/ncu, notes/q13/ncu, notes/q12/rtnn_refit.txt)  writes model/results_latency_term.md"""
import os, re, sys, csv, statistics as st
import numpy as np
from scipy.optimize import least_squares
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..'))
import parse_logs as PL
import callback_cost as CB
ND = os.path.expanduser('~/eoss/notes/q13/ncu'); NQ = os.path.expanduser('~/eoss/notes/q14/ncu'); GHZ = 2.5e9; NSM = 60
L = []
def P(*a):
    t = ' '.join(str(x) for x in a); print(t); L.append(t)
rms = lambda x: float(np.sqrt(np.mean(np.square(x))))
def ncu_w(name):
    p = '%s/%s.csv' % (NQ, name)
    if not os.path.exists(p): return float('nan')
    for r in csv.reader(open(p)):
        if len(r) > 14 and r[12] == 'sm__warps_active.avg.per_cycle_active': return float(r[14].replace(',', ''))
    return float('nan')
def cb_out(d, wl):   # (cycles, calls) of the outermost callback program
    return (d['f1c'], d['f1n']) if wl == 'bh' else (d['f0c'], d['f0n'])
def shares(d, wl):
    tot = d['total']; cbc = cb_out(d, wl)[0]
    if wl == 'bh': trav = d['f0c']; oth = tot - trav - cbc
    else: trav = d['trace'] - cbc; oth = tot - d['trace']
    return cbc / tot, trav / tot, oth / tot
# ---------------- configs
RT = [('u1e5', r, o) for r in (2, 4, 6) for o in ('sorted', 'unsorted')] + [('u1e6', r, o) for r in (1, 2, 3) for o in ('sorted', 'unsorted')]
rname = lambda c: 'rtnn_%s_r%d_%s' % c
rows = []   # dict per launch: wl, name, N (queries), ms_off, ms_on, W, loaded dict, small dict
def med_runs(prefix, mode, key=None):
    rs = PL.runs(prefix, mode)
    if prefix == 'knn': return rs
    return rs
def collect():
    out = []
    for c in RT:
        n = rname(c); ld = PL.runs(n, 'loaded'); sm = PL.runs(n, 'small'); of = PL.runs(n, 'off'); Nq = 1e5 if c[0] == 'u1e5' else 1e6
        out.append(dict(wl='rtnn', name=n, N=Nq, ms_off=PL.med([r.get('ms') for r in of]), ms_on=PL.med([r.get('ms') for r in ld]), W=ncu_w(n), ld=ld, sm=sm, ncu='%s/%s' % (ND, n), warps=Nq / 32))
    ld = PL.runs('knn', 'loaded'); sm = PL.runs('knn', 'small'); of = PL.runs('knn', 'off')
    for rd in range(1, 6):
        gs = lambda rs, k: PL.med([r['rounds'].get(rd, {}).get(k) for r in rs])
        N = gs(ld, 'active')
        mk = lambda rs: [r['rounds'].get(rd, {}) for r in rs]
        out.append(dict(wl='knn', name='knn_round%d' % rd, N=N, ms_off=gs(of, 'ms'), ms_on=gs(ld, 'ms'), W=ncu_w('knn_r%d' % (rd - 1)), ld=mk(ld), sm=mk(sm), ncu='%s/knn_r%d' % (ND, rd - 1), warps=N / 32))
    for t, N in (('g100k', 1e5), ('g1M', 1e6)):
        n = 'bh_' + t; ld = PL.runs(n, 'loaded'); sm = PL.runs(n, 'small'); of = PL.runs(n, 'off')
        out.append(dict(wl='bh', name=n, N=N, ms_off=PL.med([r.get('ms') for r in of]), ms_on=PL.med([r.get('ms') for r in ld]), W=ncu_w(n), ld=ld, sm=sm, ncu='%s/bh_%s' % (ND, t), warps=N / 32))
    for f, rad, N in (('a1e5', '0.005', 1e5), ('a1e5', '0.0112', 1e5), ('a1e6', '0.0016', 1e5)):
        n = 'ark_%s_%s' % (f, rad); ld = PL.runs(n, 'loaded'); sm = PL.runs(n, 'small'); of = PL.runs(n, 'off')
        out.append(dict(wl='ark', name=n, N=N, ms_off=PL.med([r.get('ms') for r in of]), ms_on=PL.med([r.get('ms') for r in ld]), W=ncu_w(n), ld=ld, sm=sm, ncu='%s/ark_a1e5_r5' % ND, warps=N / 32))
    return out
R = collect()
def lat(runs, wl):
    v = []
    for r in runs:
        if r.get('f0n') is None and wl != 'bh': continue
        c, n = cb_out(r, wl)
        if n: v.append((c, n))
    if not v: return float('nan'), float('nan')
    return st.median([c / n for c, n in v]), st.median([n for c, n in v])
for x in R:
    x['L_ld'], x['N_ld'] = lat(x['ld'], x['wl']); x['L_unl'], x['N_unl'] = lat(x['sm'], x['wl'])
    ld = [r for r in x['ld'] if r.get('total')]
    x['share'] = tuple(np.median([shares(r, x['wl'])[k] for r in ld]) for k in range(3)) if ld else (float('nan'),) * 3
    x['totcyc'] = np.median([r['total'] for r in ld]) if ld else float('nan')
    x['ovh'] = x['ms_on'] / x['ms_off'] - 1 if x['ms_off'] == x['ms_off'] else float('nan')
# ---------------- Part A: probe overhead, time breakdown
P('## Time breakdown of summed warp time (probes on; loaded launch)')
P('| workload | launch ms (probes off) | probes on | probe overhead | W (ncu warps / SM) | implied W = warp cycles / (60 x on-time) | callbacks | traversal | other |'); P('|---|---|---|---|---|---|---|---|---|')
for x in R:
    imp = x['totcyc'] / (NSM * x['ms_on'] * 1e-3 * GHZ) if x['ms_on'] == x['ms_on'] else float('nan')
    P('| %s | %.3f | %.3f | %+.0f%% | %.1f | %.1f | %.0f%% | %.0f%% | %.0f%% |' % (x['name'], x['ms_off'], x['ms_on'], 100 * x['ovh'], x['W'], imp, 100 * x['share'][0], 100 * x['share'][1], 100 * x['share'][2]))
P('callbacks = outermost callback program (RTNN, kNN, Arkade: intersection program, which contains the any hit of kNN; BH: optixInvoke = closest hit or miss); traversal = trace window minus callbacks (BH: optixTraverse windows); other = raygen outside the trace window (BH: also the loop glue and the first touch prologue). Share of summed warp cycles, not of wall time: with the same warps in flight they are equal.')
# BH phases
for x in R:
    if x['wl'] == 'bh':
        ph = [r['phase'] for r in x['ld'] if r.get('phase')]
        if ph:
            p = np.median(np.array(ph), axis=0); d = [r for r in x['ld'] if r.get('total')][0]; tot = np.median([r['total'] for r in x['ld'] if r.get('total')])
            P('BH %s: prologue (raygen start to loop) %.1f%%, loop %.1f%% (traverse %.1f%%, invoke %.1f%%, glue %.1f%%), epilogue %.2f%% of warp time' % (x['name'], 100 * p[0] / tot, 100 * p[1] / tot, 100 * np.median([r['f0c'] for r in x['ld'] if r.get('total')]) / tot, 100 * np.median([r['f1c'] for r in x['ld'] if r.get('total')]) / tot, 100 * (p[1] - np.median([r['f0c'] + r['f1c'] for r in x['ld'] if r.get('total')])) / tot, 100 * p[2] / tot))
# ---------------- Part B: per call latencies
P('\n## Latency per warp call of the outermost callback program (SM cycles)')
P('| workload | calls (loaded) | L loaded | L unloaded (1 warp per SM) | loaded / unloaded | warp calls per warp |'); P('|---|---|---|---|---|---|')
for x in R: P('| %s | %.0f | %.0f | %.0f | %.2f | %.1f |' % (x['name'], x['N_ld'], x['L_ld'], x['L_unl'], x['L_ld'] / x['L_unl'], x['N_ld'] / x['warps']))
sub = {}
for x in R:
    ld = [r for r in x['ld'] if r.get('f2n') or r.get('f1n')]
    if x['wl'] == 'knn':
        a = [r for r in x['ld'] if r.get('f1n')]
        if a: sub[x['name']] = 'any hit inside the intersection program: %.0f cycles per call loaded (%.0f%% of the intersection program time)' % (np.median([r['f1c'] / r['f1n'] for r in a]), 100 * np.median([r['f1c'] / r['f0c'] for r in a]))
    if x['wl'] == 'bh' and ld:
        sub[x['name']] = 'closest hit %.0f cycles per call, miss %.0f cycles per call, optixTraverse %.0f cycles per call (loaded)' % (np.median([r['f2c'] / r['f2n'] for r in ld]), np.median([r['f3c'] / r['f3n'] for r in ld]), np.median([r['f0c'] / r['f0n'] for r in ld]))
for k, v in sub.items(): P('%s: %s' % (k, v))
# ---------------- Part C: callback time (SM cycles per SM)
def floor_cycles(x):
    if x['wl'] == 'ark' and x['name'] != 'ark_a1e5_0.005':   # counts of the first config per call, times this config's calls
        base = CB.term(x['ncu'], 1e5 / 32, 'max'); calls0 = sum(p[1] for p in base['parts']); per = base['cycles'] / calls0
        return per * x['N_ld']
    return CB.term(x['ncu'], x['warps'], 'max')['cycles']
for x in R:
    x['T_floor'] = floor_cycles(x); nsm = min(NSM, x['warps']); x['T_lat_ld'] = x['L_ld'] * x['N_ld'] / (nsm * x['W']); x['T_lat_unl'] = x['L_unl'] * x['N_ld'] / (nsm * x['W'])   # N from the loaded run (same launch size)
    x['T_cb_unl'] = max(x['T_floor'], x['T_lat_unl']); x['T_cb_ld'] = max(x['T_floor'], x['T_lat_ld'])
P('\n## Callback busy time per SM (ms at 2.5 GHz): measured, floor (F131), latency terms')
P('| workload | callback busy time measured (probes on, = share x launch) | floor F131 | latency unloaded L | max(floor, unloaded) | error floor | error max(floor, unloaded) | latency loaded L | error loaded |'); P('|---|---|---|---|---|---|---|---|---|')
e = {'floor': {}, 'unl': {}, 'ld': {}}
for x in R:
    meas = x['share'][0] * x['ms_on']; f = x['T_floor'] / GHZ * 1e3; u = x['T_cb_unl'] / GHZ * 1e3; l = x['T_lat_ld'] / GHZ * 1e3
    ef, eu, el = np.log(f / meas), np.log(u / meas), np.log(l / meas); e['floor'][x['name']] = ef; e['unl'][x['name']] = eu; e['ld'][x['name']] = el
    P('| %s | %.3f | %.3f | %.3f | %.3f | %+.2f | %+.2f | %.3f | %+.2f |' % (x['name'], meas, f, x['T_lat_unl'] / GHZ * 1e3, u, ef, eu, l, el))
for wl in ('rtnn', 'knn', 'bh', 'ark'):
    ns = [x['name'] for x in R if x['wl'] == wl and e['floor'][x['name']] == e['floor'][x['name']] and x['name'] != 'knn_round5']
    P('%s callback busy time, rms log error: floor %.2f, max(floor, unloaded latency) %.2f, loaded latency %.2f' % (wl, rms([e['floor'][n] for n in ns]), rms([e['unl'][n] for n in ns]), rms([e['ld'][n] for n in ns])))
# ---------------- Part D: launch time, F131 structure with the callback term replaced
class RR:
    data = {}
    for l in open('notes/q12/rtnn_refit.txt').read().split('\n')[1:13]:
        p = [t.strip() for t in l.replace('/', '|').split('|')]; n = p[0].split('_'); data[(n[0], int(n[1][1:]), n[2])] = dict(cyc=float(p[1]), q=float(p[2]), cand=float(p[3]), eff=float(p[4]), miss=float(p[5]))
byname = {x['name']: x for x in R}
def rt_cb(c, key): x = byname[rname(c)]; return x[key] * NSM / x['N']   # cycles per query per SM
def rt_meas(c): x = byname[rname(c)]; return x['ms_off'] * 1e-3 * GHZ * NSM / x['N']
def rt_err(cfgs, p, key): return [np.log((p[0] + rt_cb(c, key) + RR.data[c]['miss']) / rt_meas(c)) for c in cfgs]
splits = [(s, tuple(r for r in rs if r != te), te) for s, rs in (('u1e5', (2, 4, 6)), ('u1e6', (1, 2, 3))) for te in rs]
P('\n## Launch time, RTNN (12 configurations; c0 fitted on the other radii of the same size as in Finding 131, held out; measured = probes off launch time of this queue)')
res = {}
for key in ('T_floor', 'T_cb_unl', 'T_cb_ld'):
    ab, ratio = [], []
    for s, tr, te in splits:
        trc = [(s, r, o) for r in tr for o in ('sorted', 'unsorted')]; tec = [(s, te, o) for o in ('sorted', 'unsorted')]
        f = least_squares(lambda p: rt_err(trc, p, key), [100], bounds=(0, np.inf)).x; ab.append(rms(rt_err(tec, f, key)))
        pr = lambda c: f[0] + rt_cb(c, key) + RR.data[c]['miss']; ratio.append(abs(np.log(pr((s, te, 'unsorted')) / pr((s, te, 'sorted')) / (rt_meas((s, te, 'unsorted')) / rt_meas((s, te, 'sorted'))))))
    tf = []
    for a, b in (('u1e5', 'u1e6'), ('u1e6', 'u1e5')):
        trc = [c for c in RT if c[0] == a]; tec = [c for c in RT if c[0] == b]; f = least_squares(lambda p: rt_err(trc, p, key), [100], bounds=(0, np.inf)).x; tf.append(rms(rt_err(tec, f, key)))
    fall = least_squares(lambda p: rt_err(RT, p, key), [100], bounds=(0, np.inf)).x; res[key] = (np.mean(ab), np.mean(ratio), np.mean(tf), fall[0], rms(rt_err(RT, fall, key)))
P('| callback term | radius splits held out rms log error | sorted/unsorted ratio error | size transfers | c0 (all 12) | in sample |'); P('|---|---|---|---|---|---|')
for key, nm in (('T_floor', 'before: floor (F131 form)'), ('T_cb_unl', 'after: max(floor, unloaded latency x calls / W)'), ('T_cb_ld', 'after: max(floor, loaded latency x calls / W) (consistency)')):
    r = res[key]; P('| %s | %.2f | %.2f | %.2f | %.0f | %.2f |' % (nm, r[0], r[1], r[2], r[3], r[4]))
c0 = res['T_floor'][3]; c0u = res['T_cb_unl'][3]; c0l = res['T_cb_ld'][3]
P('\n## Launch time, growing radius kNN rounds (c0 from the 12 RTNN fit of the same row; miss term as in callback_eval.py)')
act = [200000, 200000, 200000, 197660, 83]; qpr = [1.134, 1.919, 8.145, 56.026]; bef = [0.187, 0.203, 0.284, 0.734]
P('| round | measured ms (probes off) | before: floor | after: unloaded latency | after: loaded latency | err before | err unloaded | err loaded |'); P('|---|---|---|---|---|---|---|---|')
E = {'b': [], 'u': [], 'l': []}
for i in range(5):
    x = byname['knn_round%d' % (i + 1)]; N = x['N']; miss = max(bef[i] * 1e-3 * 1.5e11 / act[i] - (110.0 + 4.9 * (qpr[i] + 1)), 0) if i < 4 else 0.0
    f = lambda c0_, T: (c0_ + T * NSM / N + miss) * N / (NSM * GHZ) * 1e3
    pb, pu, pl = f(c0, x['T_floor']), f(c0u, x['T_cb_unl'] if x['T_cb_unl'] == x['T_cb_unl'] else x['T_floor']), f(c0l, x['T_cb_ld'])
    m = x['ms_off']; P('| %d | %.3f | %.3f | %.3f | %.3f | %+.2f | %+.2f | %+.2f |' % (i + 1, m, pb, pu, pl, np.log(pb / m), np.log(pu / m), np.log(pl / m)))
    if i < 4: E['b'].append(np.log(pb / m)); E['u'].append(np.log(pu / m)); E['l'].append(np.log(pl / m))
P('rounds 1 to 4 rms log error: before %.2f, unloaded latency %.2f, loaded latency %.2f' % (rms(E['b']), rms(E['u']), rms(E['l'])))
P('\n## Launch time, RT-BarnesHut (traversal c 20 per traversal per SM as in Finding 131; larger of traversal and callback plus a quarter of the smaller)')
P('| bodies | measured ms (probes off) | before: floor | after: unloaded | after: loaded | err before | err unloaded | err loaded | startup (small launch, 1,920 bodies) ms |'); P('|---|---|---|---|---|---|---|---|---|')
for tag, trav in (('g100k', 1.58e7), ('g1M', 2.23e8)):
    x = byname['bh_' + tag]; t = trav * 20 / (NSM * GHZ) * 1e3; cm = lambda T: T / GHZ * 1e3; comb = lambda a, b: max(a, b) + 0.25 * min(a, b)
    pb, pu, pl = comb(t, cm(x['T_floor'])), comb(t, cm(x['T_cb_unl'])), comb(t, cm(x['T_cb_ld'])); m = x['ms_off']
    su = PL.med([r.get('ms') for r in x['sm']])
    P('| %d | %.2f | %.2f | %.2f | %.2f | %+.2f | %+.2f | %+.2f | %.2f |' % (x['N'], m, pb, pu, pl, np.log(pb / m), np.log(pu / m), np.log(pl / m), su))
P('\n## Launch time, Arkade (L2 build, 2D synthetic data; callback term alone, no replay miss term)')
P('| config | measured ms (probes off) | before: floor | after: unloaded | after: loaded | err before | err unloaded | err loaded |'); P('|---|---|---|---|---|---|---|---|')
for x in R:
    if x['wl'] != 'ark': continue
    m = x['ms_off']; cm = lambda T: T / GHZ * 1e3; P('| %s | %.2f | %.2f | %.2f | %.2f | %+.2f | %+.2f | %+.2f |' % (x['name'], m, cm(x['T_floor']), cm(x['T_cb_unl']), cm(x['T_cb_ld']), np.log(cm(x['T_floor']) / m), np.log(cm(x['T_cb_unl']) / m), np.log(cm(x['T_cb_ld']) / m)))
open(os.path.expanduser('~/eoss/model/results_latency_term.md'), 'w').write('# Callback latency probes: time breakdown and latency aware callback term (queue 14), generated by model/q14/cb_latency.py\n\n' + '\n'.join(L) + '\n')
# ---------------- Part E (post hoc, exploratory): is the queueing delay of a callback proportional to the floor cost per call?  L_ld = L_unl + (W - 1) k f,  f = floor SM cycles per warp call
P('\n## Post hoc: loaded latency = unloaded latency + (W - 1) x k x floor cost per call (k fitted on RTNN only, then applied to the other workloads)')
P('| workload | floor cost per call f (cycles) | L loaded | L unloaded | k = (L ld - L unl) / ((W - 1) f) |'); P('|---|---|---|---|---|')
for x in R:
    if x['name'] == 'knn_round5' or x['L_unl'] != x['L_unl']: continue
    x['f'] = x['T_floor'] * NSM / x['N_ld']; x['k'] = (x['L_ld'] - x['L_unl']) / ((x['W'] - 1) * x['f'])
    P('| %s | %.0f | %.0f | %.0f | %.2f |' % (x['name'], x['f'], x['L_ld'], x['L_unl'], x['k']))
kr = float(np.median([x['k'] for x in R if x['wl'] == 'rtnn' and 'k' in x])); P('median k over the 12 RTNN launches: %.2f' % kr)
P('| workload | L loaded | predicted with k RTNN | log error |'); P('|---|---|---|---|')
ee = {}
for x in R:
    if 'k' not in x: continue
    pl = x['L_unl'] + (x['W'] - 1) * kr * x['f']; ee.setdefault(x['wl'], []).append(np.log(pl / x['L_ld']))
    P('| %s | %.0f | %.0f | %+.2f |' % (x['name'], x['L_ld'], pl, np.log(pl / x['L_ld'])))
for k, v in ee.items(): P('%s rms log error of the loaded latency: %.2f' % (k, rms(v)))
open(os.path.expanduser('~/eoss/model/results_latency_term.md'), 'w').write('# Callback latency probes: time breakdown and latency aware callback term (queue 14), generated by model/q14/cb_latency.py\n\n' + '\n'.join(L) + '\n')
