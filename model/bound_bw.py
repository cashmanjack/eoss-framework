"""Bandwidth bound: time >= bytes per launch / measured DRAM bandwidth (2026-11-17, tethys). CPU only, reads existing files.
    /usr/bin/python3 model/bound_bw.py bytes                         bytes per launch of every cell (committed with the predictions, before any bandwidth number)
    /usr/bin/python3 model/bound_bw.py score BW_GBPS [--ncu]         scoring; BW_GBPS = measured copy bandwidth of the 4090 (model/bound_bw_bandwidth.cu)
Bytes per launch = node traffic + ray reads + result writes.
  scenes (F205, F208, Rungholt orders): rays x 32 B read + rays x 32 B written (F213: 268 MB and 268 to 270 MB at 8,388,608 rays)
     + node traffic = D x rays, D = the replay DRAM feature in bytes per ray (F209, notes/hs/predict_*.json, notes/hs2/predict_*.json).
  prior works (priorworks/pred.json): node traffic NOT included (no replay for these builds), so the bytes are a lower bound:
     RTNN N queries x (12 B point + 50 x 4 B indices; camera.cu writes frame_buffer[q*K+i] as unsigned int), TrueKNN sum over rounds of active x (12 + k x 8 B)
     (8 B = index + float distance as in the Arkade Neigh record; ASSUMED, the TrueKNN source is not on this machine), Arkade 10,000 x (12 + 50 x 8 B)
     (Neigh = ind + dist, 8 B, arkade.diff). Device buffer rows only (the pinned host and managed memory rows move data over PCIe / paging, not DRAM bound).
Model combination: model time of order i = measured natural time / F123 speedup of order i (the F123 form predicts a speedup only; natural time is the anchor);
  max form: speedup = max(t_nat, b_nat) / max(t_nat / F123_i, b_i), b = bytes / BW.
  Bound alone: speedup = b_nat / b_i (and absolute time b_i against the measured time)."""
import glob, json, re, sys
import numpy as np
NR = 8388608; RAYB = 64   # 32 B read + 32 B written
SC205 = ['sponza', 'conference', 'rungholt']
SC208 = ['conference_camera', 'conference_bounce', 'sponza_camera', 'sponza_bounce', 'rungholt_camera', 'rungholt_bounce']
NCU = {  # F213 table (model/results_rungholt_ncu.md): DRAM read MB, write MB, ncu ms
    'natural': (985, 270, 1.488), 'oracle d1': (401, 269, 1.147), 'oracle d2': (401, 268, 1.005), 'oracle d3': (402, 268, 0.934),
    'oracle d4': (402, 268, 0.890), 'oracle full': (402, 268, 0.884), 'morton-o k5': (510, 269, 1.081)}

def cells205(s):
    P = json.load(open('notes/hs/predict_%s.json' % s))['feats']; M = {r['order']: r['ms'] for r in json.load(open('notes/hs/measure_%s.json' % s))['results']}
    F = json.load(open('notes/hs/f123_predictions_sponza_conference_rungholt.json'))[s]['f123']
    return [dict(order=o, bytes=NR * RAYB + P[o]['D'] * NR, ms=M[o], f123=(1.0 if o == 'natural' else F[o])) for o in P], 'natural'

def cells208(tag):
    P = json.load(open('notes/hs2/predict_%s.json' % tag))['feats']; f = sorted(glob.glob('notes/hs2/measure_%s_2*.json' % tag))[-1]
    M = {r['order']: r['ms'] for r in json.load(open(f))['results']}; F = json.load(open(glob.glob('notes/hs2/f123_scenes_*.json')[0]))[tag]['f123']
    return [dict(order=o, bytes=NR * RAYB + P[o]['D'] * NR, ms=M[o], f123=(1.0 if o == 'natural' else F[o])) for o in P], 'natural'

def combine(cells, BW):
    nat = [c for c in cells if c['order'] == 'natural'][0]; tn = nat['ms'] * 1e-3; bn = nat['bytes'] / BW; out = []
    for c in cells:
        if c['order'] == 'natural': continue
        b = c['bytes'] / BW; tm = nat['ms'] * 1e-3 / c['f123']; t = c['ms'] * 1e-3
        meas = tn / t; f = tn / tm; bound = bn / b; mx = max(tn, bn) / max(tm, b)
        out.append(dict(order=c['order'], meas=meas, f123=f, bound=bound, mx=mx, bind=b > tm, b_over_t=b / t, bnat_over_tnat=bn / tn, b_ms=b * 1e3, t_ms=c['ms']))
    return out

def rms(xs): return float(np.sqrt(np.mean(np.square(xs))))

def scenes(BW, fmt=True):
    groups = [('F205 ' + s, cells205(s)[0]) for s in SC205] + [('F208 ' + t, cells208(t)[0]) for t in SC208]
    print('| set | n | F123 rms | bound alone rms | max(F123, bound) rms | cells where bound binds | bound / measured time: natural, min, max over orders |'); print('|---|---|---|---|---|---|---|')
    pool = {'F205': [[], [], []], 'F208': [[], [], []]}; nb = {'F205': 0, 'F208': 0}; nn = {'F205': 0, 'F208': 0}; detail = []
    for name, cells in groups:
        r = combine(cells, BW); L = lambda k: [np.log(x[k] / x['meas']) for x in r]
        for i, k in enumerate(('f123', 'bound', 'mx')): pool[name[:4]][i] += L(k)
        nb[name[:4]] += sum(x['bind'] for x in r); nn[name[:4]] += len(r)
        bo = [x['b_over_t'] for x in r]
        print('| %s | %d | %.3f | %.3f | %.3f | %d | %.2f, %.2f, %.2f |' % (name, len(r), rms(L('f123')), rms(L('bound')), rms(L('mx')), sum(x['bind'] for x in r), r[0]['bnat_over_tnat'], min(bo), max(bo)))
        detail.append((name, r))
    print()
    for k in pool: print('pooled %s (%d cells): F123 %.3f  bound alone %.3f  max form %.3f  bound binds in %d of %d' % (k, nn[k], *[rms(v) for v in pool[k]], nb[k], nn[k]))
    return detail

def rungholt_ncu(BW):
    P = json.load(open('notes/hs/predict_rungholt.json'))['feats']; M = {r['order']: r['ms'] for r in json.load(open('notes/hs/measure_rungholt.json'))['results']}
    F = json.load(open('notes/hs/f123_predictions_sponza_conference_rungholt.json'))['rungholt']['f123']
    print('\nDiagnostic (labeled, in sample: the bytes are the F213 ncu bytes, known after the fact, not part of the pass rules): Rungholt random, ncu DRAM bytes (read + write) in place of the replay bytes')
    print('| order | replay bytes MB | ncu bytes MB | replay bound ms | ncu bound ms | measured ms | ncu bound / measured |'); print('|---|---|---|---|---|---|---|')
    cells = []
    for o, (rd, wr, ms) in NCU.items():
        rb = (NR * RAYB + P[o]['D'] * NR) / 1e6
        print('| %s | %.0f | %d | %.3f | %.3f | %.3f | %.2f |' % (o, rb, rd + wr, rb * 1e6 / BW * 1e3, (rd + wr) * 1e6 / BW * 1e3, M[o], (rd + wr) * 1e6 / BW * 1e3 / M[o]))
        cells.append(dict(order=o, bytes=(rd + wr) * 1e6, ms=M[o], f123=(1.0 if o == 'natural' else F[o])))
    r = combine(cells, BW); L = lambda k: [np.log(x[k] / x['meas']) for x in r]
    print('rms log speedup error, 6 orders (Morton included): F123 %.3f  bound alone %.3f  max form %.3f  (bound binds in %d of %d)' % (rms(L('f123')), rms(L('bound')), rms(L('mx')), sum(x['bind'] for x in r), len(r)))

def parse_results():
    rows = []; txt = open('model/priorworks/results_priorworks.md').read(); P = json.load(open('model/priorworks/pred.json'))
    sec = None; seen = {}
    for ln in txt.splitlines():
        if ln.startswith('## RTNN'): sec = 'rtnn'
        elif ln.startswith('## TrueKNN'): sec = 'tk'
        elif ln.startswith('## Arkade'): sec = 'ark'
        c = [x.strip() for x in ln.strip().strip('|').split('|')]
        if sec == 'rtnn' and re.match(r'^(bunny|dragon)$', c[0]):
            seen[c[0]] = seen.get(c[0], 0) + 1; key = 'rtnn_%s_r%d' % (c[0], seen[c[0]])   # table rows come in the order r1, r2 per dataset
            p = P[key]; rows.append(dict(name=key, model=p['ms'], meas=float(c[3].split()[0]), bytes=p['N'] * (12 + 50 * 4)))
        elif sec == 'tk' and c[0] in ('road', 'porto1m', 'uni1m'):
            k = int(c[1]); key = 'trueknn_%s_k%d' % (c[0], k); p = P[key]
            rows.append(dict(name=key, model=p['ms'], meas=float(c[5].split()[0]), bytes=sum(a * (12 + 8 * k) for a in p['acts'])))
        elif sec == 'ark' and c[0] in ('gowalla', 'glove3d', 'manuscript'):
            key = 'arkade_' + c[0]; p = P[key]; rows.append(dict(name=key, model=p['fixed'], meas=float(c[3].split()[0]), bytes=10000 * (12 + 50 * 8)))
    return rows

def priorworks(BW):
    rows = parse_results()
    print('\nPrior works rows (device buffer / fixed versions; triton RTX 6000 Ada; bandwidth %.0f GB/s = 960 GB/s peak x the copy fraction measured on the 4090; bytes are a LOWER bound: no node traffic)' % (BW / 1e9))
    print('| row | bytes MB | bound ms | model ms | measured ms | bound / model | bound / measured | binds (bound > model) | max(model, bound) / measured |'); print('|---|---|---|---|---|---|---|---|---|')
    nb = 0; e0 = []; e1 = []
    for r in rows:
        b = r['bytes'] / BW * 1e3; m = max(r['model'], b); nb += b > r['model']; e0.append(np.log(r['model'] / r['meas'])); e1.append(np.log(m / r['meas']))
        print('| %s | %.1f | %.4f | %.3f | %.3f | %.3f | %.4f | %s | %.3f |' % (r['name'], r['bytes'] / 1e6, b, r['model'], r['meas'], b / r['model'], b / r['meas'], 'YES' if b > r['model'] else 'no', m / r['meas']))
    print('rows %d, bound binds in %d; median |log error| model %.3f, max form %.3f; rms %.3f / %.3f' % (len(rows), nb, np.median(np.abs(e0)), np.median(np.abs(e1)), rms(e0), rms(e1)))

if __name__ == '__main__':
    if sys.argv[1] == 'bytes':
        print('| set | order | bytes MB (ray 64 B per ray + replay node traffic) |\n|---|---|---|')
        for s in SC205:
            for c in cells205(s)[0]: print('| F205 %s | %s | %.0f |' % (s, c['order'], c['bytes'] / 1e6))
        for t in SC208:
            for c in cells208(t)[0]: print('| F208 %s | %s | %.0f |' % (t, c['order'], c['bytes'] / 1e6))
        print('\n| prior works row | bytes MB (lower bound, no node traffic) |\n|---|---|')
        for r in parse_results(): print('| %s | %.1f |' % (r['name'], r['bytes'] / 1e6))
    else:
        BW = float(sys.argv[2]) * 1e9; scenes(BW); rungholt_ncu(BW)
        frac = float(sys.argv[2]) / 1008.0; priorworks(960e9 * frac)
