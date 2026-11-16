"""Queue 17: mean value analysis of the loaded callback latency (form: model/q17/FORM_mva.md, committed before this ran). CPU only.
Run: /usr/bin/python3 -I model/q17/mva.py   writes model/results_mva.md"""
import os, sys, io, contextlib
import numpy as np
H = os.path.expanduser('~/eoss'); sys.path.insert(0, H + '/model/q14'); sys.path.insert(0, H + '/model'); os.chdir(H)
with contextlib.redirect_stdout(io.StringIO()): import cb_latency as CL
L = []
def P(*a):
    t = ' '.join(str(x) for x in a); print(t); L.append(t)
def mva(D, Z, n):
    """Exact MVA, one queue D and delay Z; non integer n by linear interpolation. Returns (residence at the queue, throughput)."""
    Q = 0.0; res = {}
    for k in range(1, int(np.ceil(n)) + 1):
        R = D * (1 + Q); X = k / (R + Z); Q = X * R; res[k] = (R, X)
    lo = int(np.floor(n)); fr = n - lo
    if lo < 1: return res[1]
    if fr == 0: return res[lo]
    return tuple((1 - fr) * a + fr * b for a, b in zip(res[lo], res[lo + 1]))
rows = []
for x in CL.R:
    if x['name'] == 'knn_round5' or not (x['L_unl'] == x['L_unl']): continue
    nsm = min(CL.NSM, x['warps']); calls = x['N_ld']; f = x['T_floor'] * nsm / calls
    sm = [r for r in x['sm'] if r.get('total')]
    C = float(np.median([r['total'] / CL.cb_out(r, x['wl'])[1] for r in sm]))      # unloaded warp cycles per callback call
    Lu, Ll, W = x['L_unl'], x['L_ld'], x['W']
    D1 = f; Z1 = max(C - f, 0.0); R1, _ = mva(D1, Z1, W); v1 = R1 + max(Lu - f, 0.0)
    Dm = max(Lu - f, 0.0) + f; Zm = max(C - Lu, 0.0)     # V2: the whole callback (issue plus memory) is one FIFO server
    R2, _ = mva(Dm, Zm, W); v2 = R2
    k = (Ll - Lu) / ((W - 1) * f) if f > 0 else float('nan'); post = Lu + (W - 1) * 0.37 * f
    lo, hi = 0.01, 5000.0                      # post hoc diagnostic (a fit, not a prediction): the demand multiplier m (D = m f) at which V1 reproduces the measured loaded latency
    for _ in range(80):
        mid = np.sqrt(lo * hi); Rm, _ = mva(mid * f, Z1, W); (lo, hi) = (mid, hi) if Rm + max(Lu - f, 0.0) < Ll else (lo, mid)
    R3, _ = mva(2.0 * f, Z1, W); v3 = R3 + max(Lu - f, 0.0)   # post hoc: one multiplier 2 for every workload (chosen after the m column was seen)
    rows.append(dict(v3=v3, m=np.sqrt(lo * hi), wl=x['wl'], name=x['name'], W=W, f=f, C=C, Lu=Lu, Ll=Ll, v1=v1, v2=v2, post=post, k=k, k1=(v1 - Lu) / ((W - 1) * f), k2=(v2 - Lu) / ((W - 1) * f)))
P('# Mean value analysis of the loaded callback latency (Queue 17, 4070 Ti data of F133; form fixed first in model/q17/FORM_mva.md)')
P('Cycles per warp call. f = floor (issue) demand, C = unloaded warp cycles per callback call, W = warps in flight per SM (ncu). V1: issue queue, everything else a delay. V2: the whole callback is one server. k = (L loaded - L unloaded) / ((W - 1) f), the implied queueing weight (V1 and V2 columns give the k the network implies).\n')
P('| workload | W | f | C unloaded | L unloaded | L loaded (measured) | V1 | V2 | post hoc rule (k 0.37) | k measured | k V1 | k V2 | err unloaded | err V1 | err V2 | err rule | m needed (post hoc) |'); P('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
e = lambda p, m: np.log(p / m)
for r in rows:
    P('| %s | %.1f | %.0f | %.0f | %.0f | %.0f | %.0f | %.0f | %.0f | %.2f | %.2f | %.2f | %+.2f | %+.2f | %+.2f | %+.2f | %.1f |' % (r['name'], r['W'], r['f'], r['C'], r['Lu'], r['Ll'], r['v1'], r['v2'], r['post'], r['k'], r['k1'], r['k2'], e(r['Lu'], r['Ll']), e(r['v1'], r['Ll']), e(r['v2'], r['Ll']), e(r['post'], r['Ll']), r['m']))
rms = lambda v: float(np.sqrt(np.mean(np.square(v))))
P('\nrms log error by kind (predicted over measured loaded latency):'); P('| kind | n | unloaded | V1 | V2 | post hoc rule (k 0.37) | V1 with m = 2 (post hoc, one constant) |'); P('|---|---|---|---|---|---|---|')
for wl in ('rtnn', 'knn', 'bh', 'ark'):
    rr = [r for r in rows if r['wl'] == wl]
    P('| %s | %d | %.2f | %.2f | %.2f | %.2f | %.2f |' % (wl, len(rr), rms([e(r['Lu'], r['Ll']) for r in rr]), rms([e(r['v1'], r['Ll']) for r in rr]), rms([e(r['v2'], r['Ll']) for r in rr]), rms([e(r['post'], r['Ll']) for r in rr]), rms([e(r['v3'], r['Ll']) for r in rr])))
rr = rows; P('| all | %d | %.2f | %.2f | %.2f | %.2f | %.2f |' % (len(rr), rms([e(r['Lu'], r['Ll']) for r in rr]), rms([e(r['v1'], r['Ll']) for r in rr]), rms([e(r['v2'], r['Ll']) for r in rr]), rms([e(r['post'], r['Ll']) for r in rr]), rms([e(r['v3'], r['Ll']) for r in rr])))
rr = [r for r in rows if r['wl'] != 'ark']; P('| all but Arkade | %d | %.2f | %.2f | %.2f | %.2f | %.2f |' % (len(rr), rms([e(r['Lu'], r['Ll']) for r in rr]), rms([e(r['v1'], r['Ll']) for r in rr]), rms([e(r['v2'], r['Ll']) for r in rr]), rms([e(r['post'], r['Ll']) for r in rr]), rms([e(r['v3'], r['Ll']) for r in rr])))
open(H + '/model/results_mva.md', 'w').write('\n'.join(L) + '\n')
