"""One model for both ray kinds with a reuse feature (queue 11, 2026-11-15). Run from the repo root with /usr/bin/python3.
Model: cost of an order c = fixed(q) + miss, fixed(q) = the Ada fitted a_q q + b_q q/W + c_qW q W/16 times a free scale A (OptiX harness on the 4070 Ti runs at another level than the Ada fit),
miss = S1 x [ beta_L1 x (1 - h(C, g)) + beta_L2 x (1 - h(C2, g)) ] + gamma x D/32, where h(C, g) = mean over the node fetches of the CPU replay of 1 / (1 + (d / C)^g) with d the LRU reuse distance (lines of 128 B,
cold fetches count as misses): g large is the hard 'distance under the capacity' share, small g a gradual fall. S1 = L1 sectors requested per ray (the number of node fetches times a constant).
The fit is on log speedups over the natural order as in refit_optix.py, same training sets (soup 300,000 random rays and bounce rays on the 100,000 soup), nothing else.
    model/reuse_feature.py [--c 480]  -> prints the table and writes notes/q11/reuse_feature.json"""
import json, os, sys, numpy as np
from scipy.optimize import least_squares
sys.path.insert(0, 'model'); sys.path.insert(0, 'ttu/bench/Memory_4')
import performance_model as PM, run_model as RM
p0 = PM.fit(RM.memory4_rows())
KEY = {'natural': 'natural', 'shuffled': 'shuffled', 'morton-o k5': 'morton-o_k5', 'oracle d1': 'oracle_d1', 'oracle d2': 'oracle_d2', 'oracle d3': 'oracle_d3', 'oracle d4': 'oracle_d4', 'oracle full': 'oracle_full'}


def hist(path):
    raw = None
    for l in open(path):
        if l.startswith('sm nodes raw'): raw = int(l.split()[3])
        if l.startswith('sm nodes dedupe'):
            p = l.split(); return int(p[3]), int(p[4]), int(p[5]), np.array(p[6:], np.float64), raw   # total, cold, overflow, hist by distance, raw (lane level) fetches


def load(tag, kind, nsm):
    if kind == 'random':
        f = json.load(open('notes/q10/check/predict_%s.json' % tag))['feats']; m = {r['order']: r['ms'] for r in json.load(open('notes/q10/check/measure_%s.json' % tag))['results']}
        hd = lambda k: hist('notes/q10/check/reuse_%s_%s.txt' % (tag, KEY[k]))
    else:
        f = json.load(open('notes/io/predict/predict_bounce_%d.json' % tag))['feats']; m = {r['order']: r['ms'] for r in json.load(open('notes/io/bounce/bounce_oracle_s%d.json' % tag))['results']}
        hd = lambda k: hist('notes/io/predict/reuse_%d_%s.txt' % (tag, KEY[k]))
    orders = [k for k in f if k in m and k in KEY]
    return dict(name=(tag if kind == 'random' else 'bounce%d' % tag), kind=kind, feats={k: f[k] for k in orders}, ms={k: m[k] for k in orders}, hist={k: hd(k) for k in orders})


SETS = [load('s3e5', 'random', 0), load('s15e5', 'random', 0), load('bistro', 'random', 0), load(100000, 'bounce', 0), load(3000000, 'bounce', 0)]
TRAIN = ['s3e5', 'bounce100000']
D_OF = {s['name']: s for s in SETS}


def share(h, C, g):
    tot, cold, over, hh, _ = h; d = np.arange(len(hh)) + 0.5
    hit = (hh / (1 + (d / C) ** g)).sum()
    return hit / tot          # cold and overflow fetches are misses (they are in tot, not in hh)


def cost(par, s, k, mode):
    f = s['feats'][k]; A = par[0]
    fixed = A * (p0['a_q'] * f['q'] + p0['b_q'] * f['q'] / f['W'] + p0['c_qW'] * f['q'] * f['W'] / 16.0)
    if mode == 'F':      # F120: sector fills F from the hard threshold at 480 lines
        return fixed + par[1] * f['F']
    if mode == 'F+D':
        return fixed + par[1] * f['F'] + par[2] * f['D'] / 32.0
    h = s['hist'][k]
    if mode == 'hard':   # the share of fetches with reuse distance under C lines (C fixed 480 = 60 KB), cost = S1 x (1 - share)
        return fixed + par[1] * f['S1'] * (1 - share(h, 480, 60.0))
    if mode == 'soft':   # gradual: C and the sharpness fitted
        return fixed + par[1] * f['S1'] * (1 - share(h, par[2], par[3]))
    if mode == 'soft+D':
        return fixed + par[1] * f['S1'] * (1 - share(h, par[2], par[3])) + par[4] * f['D'] / 32.0
    if mode == 'share':   # as soft, miss cost per fetch scaled by (lane fetches per warp fetch)^-kappa: fetches that several lanes share overlap their latency
        return fixed + par[1] * f['S1'] * (1 - share(h, par[2], par[3])) * (h[4] / h[0]) ** (-par[4])
    if mode == 'share_fix':   # fixed part also scales with sharing: raw lane fetches enter the L1 pipe
        return fixed + par[1] * f['S1'] * (1 - share(h, par[2], par[3])) + par[4] * f['S1'] * (h[4] / h[0])
    raise ValueError(mode)


def rows(par, name, mode):
    s = D_OF[name]; out = []
    for k in s['feats']:
        if k == 'natural': continue
        out.append((k, cost(par, s, k, mode) and cost(par, s, 'natural', mode) / cost(par, s, k, mode), s['ms']['natural'] / s['ms'][k]))
    return out


rms = lambda r: float(np.sqrt(np.mean([np.log(a / b) ** 2 for _, a, b in r])))
START = {'F': [1.0, 0.5], 'F+D': [1.0, 0.5, 0.01], 'hard': [1.0, 0.5], 'soft': [1.0, 0.5, 480., 2.], 'soft+D': [1.0, 0.5, 480., 2., 0.01], 'share': [1.0, 0.5, 480., 2., 0.5], 'share_fix': [1.0, 0.5, 480., 2., 0.01]}
LOW = {'F': [0, 0], 'F+D': [0, 0, 0], 'hard': [0, 0], 'soft': [0, 0, 30., 0.3], 'soft+D': [0, 0, 30., 0.3, 0], 'share': [0, 0, 30., 0.3, -3], 'share_fix': [0, 0, 30., 0.3, 0]}
HIGH = {'F': [50, 50], 'F+D': [50, 50, 50], 'hard': [50, 50], 'soft': [50, 50, 20000., 30.], 'soft+D': [50, 50, 20000., 30., 50], 'share': [50, 50, 20000., 30., 6], 'share_fix': [50, 50, 20000., 30., 50]}


def fit(mode, train):
    res = lambda par: [np.log(a / b) for n in train for _, a, b in rows(par, n, mode)]
    best = None
    for A0 in (0.3, 1.0, 3.0):
        for b0 in (0.05, 0.5):
            x0 = list(START[mode]); x0[0], x0[1] = A0, b0
            r = least_squares(res, x0, bounds=(LOW[mode], HIGH[mode]))
            if best is None or r.cost < best.cost: best = r
    return best.x


if __name__ == '__main__':
    out = {}
    for mode in ('F', 'F+D', 'hard', 'soft', 'soft+D', 'share', 'share_fix'):
        par = fit(mode, TRAIN); e = {n: rows(par, n, mode) for n in D_OF}
        test = [n for n in D_OF if n not in TRAIN]
        print('%-7s par %s' % (mode, np.round(par, 3)))
        for n in D_OF: print('   %-14s %s rms log err %.3f' % (n, 'train' if n in TRAIN else 'test ', rms(e[n])))
        te = [x for n in test for x in e[n]]; print('   held out (random s15e5 + Bistro + bounce 3M, %d points) %.3f ; train %.3f' % (len(te), rms(te), rms([x for n in TRAIN for x in e[n]])))
        out[mode] = dict(par=list(par), rms={n: rms(e[n]) for n in e}, held=rms(te), rows={n: [(k, a, b) for k, a, b in e[n]] for n in e})
    os.makedirs('notes/q11', exist_ok=True); json.dump(out, open('notes/q11/reuse_feature.json', 'w'), indent=1)
    # robustness (added after the first run): leave one scene out over all five sets, and kappa fixed to 1
    print('leave one set out (fit on the other four), model "share":')
    loso = {}
    for n in D_OF:
        par = fit('share', [m for m in D_OF if m != n]); loso[n] = rms(rows(par, n, 'share')); print('   %-14s %.3f   par %s' % (n, loso[n], np.round(par, 2)))
    print('   rms over all %d held out points: %.3f' % (sum(len(rows([1, 1, 480, 2, 1], n, 'share')) for n in D_OF), float(np.sqrt(np.mean([v ** 2 * len(rows([1, 1, 480, 2, 1], n, 'share')) for n, v in loso.items()] ) * len(D_OF) / sum(len(rows([1, 1, 480, 2, 1], n, 'share')) for n in D_OF)))))
    out['loso'] = loso
    mode_k1 = 'share'
    res1 = lambda q: [np.log(a / b) for n in TRAIN for _, a, b in rows(list(q[:4]) + [1.0], n, mode_k1)]
    r1 = least_squares(res1, [1.0, 0.5, 480., 2.], bounds=(LOW['share'][:4], HIGH['share'][:4])); pk = list(r1.x) + [1.0]
    e1 = {n: rows(pk, n, mode_k1) for n in D_OF}; te1 = [x for n in D_OF if n not in TRAIN for x in e1[n]]
    print('kappa = 1 fixed: par %s, held out %.3f, train %.3f' % (np.round(pk, 3), rms(te1), rms([x for n in TRAIN for x in e1[n]])))
    out['share_k1'] = dict(par=pk, held=rms(te1), rms={n: rms(e1[n]) for n in e1}, rows={n: e1[n] for n in e1})
    json.dump(out, open('notes/q11/reuse_feature.json', 'w'), indent=1)
