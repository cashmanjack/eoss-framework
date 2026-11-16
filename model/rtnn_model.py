"""RTNN: the one model (reuse feature, lane sharing; model/reuse_feature.py share model with kappa 1, fitted on soup 300,000 random rays and bounce rays on the 100,000 soup only) applied to RTNN's custom primitive
BVH, sorted against unsorted queries and points (RTX 4070 Ti, radius search K 50, 100,000 points radius 4 and 1,000,000 points radius 2). Inputs from the CPU replay (model/replay/replay_aabb.cpp, reuse_aabb.cpp, rtnn_prep.py)
and the BITTU style recorder (patched RTNN, RTNN_REC); measured times are the CUDA event times of the search launch (5 runs, median). CPU only.
    /usr/bin/python3 model/rtnn_model.py"""
import json, os, sys, statistics as st, re, numpy as np
sys.path.insert(0, 'model'); sys.path.insert(0, 'ttu/bench/Memory_4')
import reuse_feature as RF, performance_model as PM
PAR = json.load(open('notes/q11/reuse_feature.json'))['share_k1']['par']
D = '/dev/shm/rt/dump/'; OUT = 'notes/q11/rtnn'
CFG = {'u1e5_r4_sorted': 1e5, 'u1e5_r4_unsorted': 1e5, 'u1e6_r2_sorted': 1e6, 'u1e6_r2_unsorted': 1e6}
a_r = RF.hist('notes/q5/reuse/reuse_1e5_random.txt')[0] / (142 * 2 * 39 * 32); KK = 82.0 / a_r
L = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); L.append(s)
def rhist(path, variant):
    raw = None
    for l in open(path):
        p = l.split()
        if p[0] == 'sm' and p[1] == variant and p[2] == 'raw': raw = int(p[3])
        if p[0] == 'sm' and p[1] == variant and p[2] == 'dedupe': return int(p[3]), int(p[4]), int(p[5]), np.array(p[6:], np.float64), raw
meas = {n: st.median(float(re.search(r'KERNEL queries \d+ ms ([\d.]+)', open('%s/%s_plain%d.log' % (OUT, n, i)).read()).group(1)) for i in range(1, 6)) for n in CFG}
rec = {n: dict(zip(('cand', 'eff', 'wmax'), (lambda m: (float(m.group(1)), float(m.group(2)), float(m.group(3))))(re.search(r'candidates_per_query ([\d.]+) lane_efficiency ([\d.]+) mean_warp_ratio [\d.]+ mean_warp_max ([\d.]+)', open('%s/%s_rec.log' % (OUT, n)).read())))) for n in CFG}
rep = {n: np.fromfile(D + n + '/rep.bin', np.int32).reshape(-1, 8) for n in CFG}
def feats(n, variant, q):
    h = rhist('%s/rtnn_replay/reuse_%s.txt' % ('notes/q11', n), variant); win = 1 if CFG[n] == 1e5 else 2; rays = win * 39 * 60 * 32
    return dict(q=q, S1=KK * h[0] / rays, F=KK * (h[1] + h[2] + h[3][480:].sum()) / rays, D=0.0, W=13), h
P('RTNN on the 4070 Ti: replay (CPU) and recorder (hardware) per query, then the model.')
P('%-18s | replay: nodes leaf slots prims in slots exact cand | recorder: cand lane eff | event ms (median of 5)' % 'config')
for n in CFG:
    r = rep[n]; P('%-18s | %6.2f %6.2f %6.2f %6.2f | %6.2f %.3f | %.4f' % (n, r[:, 0].mean(), r[:, 1].mean(), r[:, 2].mean(), r[:, 7].mean(), rec[n]['cand'], rec[n]['eff'], meas[n]))
P('hardware candidates over replay prims in the entered leaf slots: ' + ', '.join('%s %.3f' % (n, rec[n]['cand'] / rep[n][:, 2].mean()) for n in CFG) + '; over exact box candidates: ' + ', '.join('%s %.3f' % (n, rec[n]['cand'] / rep[n][:, 7].mean()) for n in CFG))
P('\nReuse (one SM stream, 60 KB LRU = 480 lines), variant nodes / nodes+leaf slots:')
for variant in ('nodes', 'nodes+leaf'):
    for n in CFG:
        f, h = feats(n, variant, 1.0); P('  %-10s %-18s lines/ray(dedupe) %.1f  miss share beyond 480 lines %.3f  cold share %.3f  lane sharing raw/dedupe %.2f  S1 %.1f F %.1f' % (variant, n, h[0] / (f['S1'] / KK * ((1 if CFG[n] == 1e5 else 2) * 39 * 60 * 32) * 1.0 / ((1 if CFG[n] == 1e5 else 2) * 39 * 60 * 32)) if False else h[0] / ((1 if CFG[n] == 1e5 else 2) * 39 * 60 * 32), (h[1] + h[2] + h[3][480:].sum()) / h[0], h[1] / h[0], h[4] / h[0], f['S1'], f['F']))
P('\nModel prediction of the unsorted over sorted time (the model: fixed part with q queries per ray, miss part from S1, the reuse share and lane sharing; parameters fitted on triangle launches only):')
meas_ratio = {s: meas['%s_unsorted' % s] / meas['%s_sorted' % s] for s in ('u1e5_r4', 'u1e6_r2')}
for variant in ('nodes', 'nodes+leaf'):
    for qname, qf in (('q = candidates + 1', lambda n: rec[n]['cand'] + 1), ('q = leaf slots with a candidate + 1', lambda n: rep[n][:, 4].mean()), ('q = 1', lambda n: 1.0)):
        out = []
        for s in ('u1e5_r4', 'u1e6_r2'):
            c = {}
            for t in ('sorted', 'unsorted'):
                n = '%s_%s' % (s, t); f, h = feats(n, variant, qf(n)); fixed = PAR[0] * (RF.p0['a_q'] * f['q'] + RF.p0['b_q'] * f['q'] / f['W'] + RF.p0['c_qW'] * f['q'] * f['W'] / 16.0)
                miss = PAR[1] * f['S1'] * (1 - RF.share((h[0], h[1], h[2], h[3], h[4]), PAR[2], PAR[3])) * (h[4] / h[0]) ** (-PAR[4]); c[t] = (fixed, miss)
            out.append('%s predicted %.2f (fixed %.0f miss sorted %.0f unsorted %.0f cycles per ray) measured %.2f' % (s, sum(c['unsorted']) / sum(c['sorted']), c['sorted'][0], c['sorted'][1], c['unsorted'][1], meas_ratio[s]))
        P('  %-10s %-36s | %s' % (variant, qname, ' | '.join(out)))
P('\nAbsolute scale check (cycles per query per SM from the event time, 2.5 GHz, 60 SMs): ' + ', '.join('%s %.0f' % (n, meas[n] * 1e-3 * 2.5e9 * 60 / CFG[n]) for n in CFG))
P('Measured unsorted / sorted launch time: ' + ', '.join('%s %.2f' % (s, v) for s, v in meas_ratio.items()) + '; older process timer numbers (Finding 116): 22.9 and 3.98.')
open('notes/q11/rtnn_model.txt', 'w').write('\n'.join(L) + '\n')
