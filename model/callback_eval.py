"""Evaluation of the SM callback term (queue 13). CPU only: reads the ncu counts of notes/q13 data (/dev/shm/cb copied to notes/q13/ncu) and earlier measured launch times.
Error before and after, log ratio predicted / measured. Run: /usr/bin/python3 model/callback_eval.py
before = the model as committed (RTNN: refit fixed part c0 + cq q, Finding 127; kNN: Finding 130 predictions; BH: Finding 129 bracket point c = 20 plus the guessed shading work)
after  = fixed part c0 (one parameter, fitted on RTNN) + measured callback cycles (ncu counts x microbenchmarked costs, no fit) + the same miss term; BH: traversal c = 20 and the callback term, larger plus a quarter of the smaller."""
import io, os, re, sys, json, contextlib, statistics as st
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0, 'model'); sys.path.insert(0, 'ttu/bench/Memory_4')
class RR:   # the numbers of notes/q12/rtnn_refit.txt (replay outputs in /dev/shm were deleted); rounded to the printed precision
    CF = [(s, r, o) for s, rs in (('u1e5', (2, 4, 6)), ('u1e6', (1, 2, 3))) for r in rs for o in ('sorted', 'unsorted')]
    data = {}
    for l in open('notes/q12/rtnn_refit.txt').read().split('\n')[1:13]:
        p = [x.strip() for x in l.replace('/', '|').split('|')]; n = p[0].split('_'); data[(n[0], int(n[1][1:]), n[2])] = dict(cyc=float(p[1]), q=float(p[2]), cand=float(p[3]), eff=float(p[4]), miss={'nodes': float(p[5]), 'nodes+leaf': float(p[6])})
    @staticmethod
    def name(c): return '%s_r%d_%s' % c
    @staticmethod
    def pred(c, p, variant, kind): d = RR.data[c]; return p[0] + p[1] * d['q'] + d['miss'][variant]
    @staticmethod
    def err(cfgs, p, variant, kind): return [np.log(RR.pred(c, p, variant, kind) / RR.data[c]['cyc']) for c in cfgs]
    @staticmethod
    def rerr(rads, s, p, variant, kind): return [np.log(RR.pred((s, r, 'unsorted'), p, variant, kind) / RR.pred((s, r, 'sorted'), p, variant, kind) / (RR.data[(s, r, 'unsorted')]['cyc'] / RR.data[(s, r, 'sorted')]['cyc'])) for r in rads]
import callback_cost as CB
ND = os.environ.get('CBDIR', '/dev/shm/cb'); L = []
def P(*a):
    t = ' '.join(str(x) for x in a); print(t); L.append(t)
rms = lambda x: float(np.sqrt(np.mean(np.square(x))))
FORM = os.environ.get('CBFORM', 'max')
# ---------------- RTNN
cb = {}
for c in RR.CF:
    n = RR.name(c); N = 1e5 if c[0] == 'u1e5' else 1e6; r = CB.term('%s/rtnn_%s' % (ND, n), N / 32, FORM); r2 = CB.term('%s/rtnn_%s' % (ND, n), N / 32, 'sum')
    cb[c] = r['cycles'] * 60 / N; cb[c + ('sum',)] = r2['cycles'] * 60 / N; RR.data[c]['h'] = r['h']
P('## RTNN: callback cycles per query per SM (measured counts x costs) against the measured total')
P('| config | measured cycles / query / SM | callback (max form) | callback (sum form) | share | L1 hit rate | warp ins per call (first callback function) | tags per call (same) |')
P('|---|---|---|---|---|---|---|---|')
for c in RR.CF:
    d = RR.data[c]; f = CB.load('%s/rtnn_%s' % (ND, RR.name(c)))[0]; g = [x for x in f if x['calls'] > 1.5 * (1e5 if c[0] == 'u1e5' else 1e6) / 32][0]
    P('| %s | %.0f | %.1f | %.1f | %.2f | %.2f | %.1f | %.1f |' % (RR.name(c), d['cyc'], cb[c], cb[c + ('sum',)], cb[c] / d['cyc'], d['h'], g['ins'], g['tags']))
def pred_after(c, p, variant='nodes', form='max'):
    d = RR.data[c]; return p[0] + cb[c if form == 'max' else c + ('sum',)] + d['miss'][variant]
def err_after(cfgs, p, form): return [np.log(pred_after(c, p, 'nodes', form) / RR.data[c]['cyc']) for c in cfgs]
def ratio_after(c, p, form): return pred_after((c[0], c[1], 'unsorted'), p, 'nodes', form) / pred_after((c[0], c[1], 'sorted'), p, 'nodes', form)
splits = [(s, tuple(r for r in rs if r != te), te) for s, rs in (('u1e5', (2, 4, 6)), ('u1e6', (1, 2, 3))) for te in rs]
rows = []
for s, tr, te in splits:
    trc = [(s, r, o) for r in tr for o in ('sorted', 'unsorted')]; tec = [(s, te, o) for o in ('sorted', 'unsorted')]
    fb = least_squares(lambda p: RR.err(trc, p, 'nodes', 'fixed'), [100, 10], bounds=(0, np.inf)).x
    fa = {fm: least_squares(lambda p: err_after(trc, p, fm), [100], bounds=(0, np.inf)).x for fm in ('max', 'sum')}
    rr = lambda f, k: abs(np.log(f(((s, te, 's')), k) / (RR.data[(s, te, 'unsorted')]['cyc'] / RR.data[(s, te, 'sorted')]['cyc'])))
    rows.append((s, te, rms(RR.err(tec, fb, 'nodes', 'fixed')), rms(err_after(tec, fa['max'], 'max')), rms(err_after(tec, fa['sum'], 'sum')),
                 abs(RR.rerr([te], s, fb, 'nodes', 'fixed')[0]), abs(np.log(ratio_after((s, te, 's'), fa['max'], 'max') / (RR.data[(s, te, 'unsorted')]['cyc'] / RR.data[(s, te, 'sorted')]['cyc']))), fb, fa['max'][0]))
for a, b in (('u1e5', 'u1e6'), ('u1e6', 'u1e5')):
    trc = [c for c in RR.CF if c[0] == a]; tec = [c for c in RR.CF if c[0] == b]
    fb = least_squares(lambda p: RR.err(trc, p, 'nodes', 'fixed'), [100, 10], bounds=(0, np.inf)).x
    fa = {fm: least_squares(lambda p: err_after(trc, p, fm), [100], bounds=(0, np.inf)).x for fm in ('max', 'sum')}
    rs_ = (2, 4, 6) if b == 'u1e5' else (1, 2, 3)
    rows.append((a + '->' + b, 'all', rms(RR.err(tec, fb, 'nodes', 'fixed')), rms(err_after(tec, fa['max'], 'max')), rms(err_after(tec, fa['sum'], 'sum')),
                 rms([abs(RR.rerr([r], b, fb, 'nodes', 'fixed')[0]) for r in rs_]), rms([abs(np.log(ratio_after((b, r, 's'), fa['max'], 'max') / (RR.data[(b, r, 'unsorted')]['cyc'] / RR.data[(b, r, 'sorted')]['cyc']))) for r in rs_]), fb, fa['max'][0]))
P('\n## RTNN held out rms log error of the absolute time (before = refit c0 + cq q, 2 parameters; after = c0 + measured callback, 1 parameter; same splits as Finding 127)')
P('| train -> test | before (F127) | after (max form) | after (sum form) | ratio error before | ratio error after | c0 before, cq | c0 after |'); P('|---|---|---|---|---|---|---|---|')
for r in rows: P('| %s %s | %.2f | %.2f | %.2f | %.2f | %.2f | %.0f, %.1f | %.0f |' % (r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7][0], r[7][1], r[8]))
sp = [r for r in rows if r[1] != 'all']; tf = [r for r in rows if r[1] == 'all']
P('mean over the radius splits: before %.2f after %.2f (sum form %.2f); size transfers: before %.2f after %.2f (sum form %.2f)' % (np.mean([r[2] for r in sp]), np.mean([r[3] for r in sp]), np.mean([r[4] for r in sp]), np.mean([r[2] for r in tf]), np.mean([r[3] for r in tf]), np.mean([r[4] for r in tf])))
allc = list(RR.CF); f0 = least_squares(lambda p: err_after(allc, p, 'max'), [100], bounds=(0, np.inf)).x; fall = least_squares(lambda p: RR.err(allc, p, 'nodes', 'fixed'), [100, 10], bounds=(0, np.inf)).x
P('fit on all 12: c0 after %.0f (before: c0 %.0f, cq %.1f); in sample rms log error before %.2f after %.2f' % (f0[0], fall[0], fall[1], rms(RR.err(allc, fall, 'nodes', 'fixed')), rms(err_after(allc, f0, 'max'))))

# ---- post hoc variants (chosen after seeing the above; labelled as such): (P1) no miss term: c0 + callback (sum form); (P2) c0 + a x callback (sum form), a fitted
def predv(c, p, v):
    d = RR.data[c]
    if v == 'P1': return p[0] + cb[c + ('sum',)]
    return p[0] + p[1] * cb[c + ('sum',)]
def errv(cfgs, p, v): return [np.log(predv(c, p, v) / RR.data[c]['cyc']) for c in cfgs]
def ratv(s, r, p, v): return np.log(predv((s, r, 'unsorted'), p, v) / predv((s, r, 'sorted'), p, v) / (RR.data[(s, r, 'unsorted')]['cyc'] / RR.data[(s, r, 'sorted')]['cyc']))
POST = {}
for v, x0 in (('P1', [100]), ('P2', [100, 1.0])):
    ab, rb = [], []
    for s, tr, te in splits:
        trc = [(s, r, o) for r in tr for o in ('sorted', 'unsorted')]; tec = [(s, te, o) for o in ('sorted', 'unsorted')]
        f = least_squares(lambda p: errv(trc, p, v), x0, bounds=(0, np.inf)).x; ab.append(rms(errv(tec, f, v))); rb.append(abs(ratv(s, te, f, v)))
    tb, tr_ = [], []
    for a, b in (('u1e5', 'u1e6'), ('u1e6', 'u1e5')):
        trc = [c for c in RR.CF if c[0] == a]; tec = [c for c in RR.CF if c[0] == b]; rs_ = (2, 4, 6) if b == 'u1e5' else (1, 2, 3)
        f = least_squares(lambda p: errv(trc, p, v), x0, bounds=(0, np.inf)).x; tb.append(rms(errv(tec, f, v))); tr_.append(rms([ratv(b, r, f, v) for r in rs_]))
    fall_ = least_squares(lambda p: errv(allc, p, v), x0, bounds=(0, np.inf)).x; POST[v] = fall_
    P('post hoc %s: radius splits mean error %.2f (ratio error %.2f), size transfers %.2f (ratio error %.2f); fit on all 12: %s, in sample %.2f' % (v, np.mean(ab), np.mean(rb), np.mean(tb), np.mean(tr_), np.round(fall_, 2), rms(errv(allc, fall_, v))))
x = np.array([RR.data[c]['cyc'] for c in RR.CF]); 
for nm, z in (('callback max form', np.array([cb[c] for c in RR.CF])), ('callback sum form', np.array([cb[c + ('sum',)] for c in RR.CF])), ('replay miss term', np.array([RR.data[c]['miss']['nodes'] for c in RR.CF])), ('replay primitives in slots q', np.array([RR.data[c]['q'] for c in RR.CF]))):
    A = np.vstack([np.ones_like(z), z]).T; b = np.linalg.lstsq(A, x, rcond=None)[0]; P('correlation of the 12 measured cycles per query with the %s: %.3f (line %.0f + %.2f x)' % (nm, np.corrcoef(z, x)[0, 1], b[0], b[1]))
# ---------------- kNN
P('\n## Growing radius kNN (200,000 points, K 16): per round launch time')
meas = {}
for rd in range(1, 6): meas[rd] = st.median(float(re.search(r'ROUND %d active \d+ radius \S+ KERNEL_MS ([\d.]+)' % rd, open('notes/q12/knn/rand_%d.log' % i).read()).group(1)) for i in (1, 2, 3))
act = [200000, 200000, 200000, 197660, 83]; qpr = [1.134, 1.919, 8.145, 56.026]; bef = [0.187, 0.203, 0.284, 0.734]   # Finding 130 prediction (round 5 was for the wrong size)
c0b, cqb = 110.0, 4.9
P('| round | active | measured ms | before ms | callback cycles / query / SM (max, sum) | after ms (max) | after ms (sum) | error before | error after (max) | error after (sum) | warp ins per call (any hit, intersection) |'); P('|---|---|---|---|---|---|---|---|---|---|---|')
eb, ea, es, e1, e2 = [], [], [], [], []
for i in range(5):
    N = act[i]; r = CB.term('%s/knn_r%d' % (ND, i), N / 32, 'max'); r2 = CB.term('%s/knn_r%d' % (ND, i), N / 32, 'sum'); cm = r['cycles'] * 60 / N; cs = r2['cycles'] * 60 / N
    f = [x for x in CB.load('%s/knn_r%d' % (ND, i))[0] if x['calls'] > 1.5 * N / 32]
    if i < 4:
        miss = bef[i] * 1e-3 * 1.5e11 / N - (c0b + cqb * (qpr[i] + 1)); miss = max(miss, 0)
        ma = (f0[0] + cm + miss) * N / 1.5e11 * 1e3; msum = (f0[0] + cs + miss) * N / 1.5e11 * 1e3; bb = bef[i]
        m1 = (POST['P1'][0] + cs) * N / 1.5e11 * 1e3; m2 = (POST['P2'][0] + POST['P2'][1] * cs) * N / 1.5e11 * 1e3; e1.append(np.log(m1 / meas[i + 1])); e2.append(np.log(m2 / meas[i + 1]))
        eb.append(np.log(bb / meas[i + 1])); ea.append(np.log(ma / meas[i + 1])); es.append(np.log(msum / meas[i + 1]))
    else: ma = msum = (f0[0] + cm) * N / 1.5e11 * 1e3; bb = float('nan')   # round 5: the miss term of this launch is unknown (replay of 83 queries not run), a floor
    P('| %d | %d | %.2f | %.2f | %.0f, %.0f | %.2f | %.2f | %.2f | %.2f | %.2f | %s |' % (i + 1, N, meas[i + 1], bb, cm, cs, ma, msum, np.log(bb / meas[i + 1]) if i < 4 else float('nan'), np.log(ma / meas[i + 1]), np.log(msum / meas[i + 1]), ', '.join('%.0f/%.0f' % (x['ins'], x['tags']) for x in f)))
P('post hoc variants on rounds 1 to 4 (P1 c0 + callback, P2 c0 + a callback, fitted on RTNN): rms log error %.2f and %.2f' % (rms(e1), rms(e2)))
P('rounds 1 to 4: rms log error before %.2f, after (max) %.2f, after (sum) %.2f; sum of rounds 1 to 4: measured %.2f ms, before %.2f' % (rms(eb), rms(ea), rms(es), sum(meas[i] for i in range(1, 5)), sum(bef)))
# ---------------- BH
P('\n## RT-BarnesHut: launch time')
P('| bodies | measured ms | before (c 20, guessed shading) | bracket | callback ms (max, sum) | traversal ms (c 20) | after (max) ms | after (sum) ms | error before | error after (max) | error after (sum) |'); P('|---|---|---|---|---|---|---|---|---|---|---|')
for tag, N, trav, shade, brk, ms in (('g100k', 1e5, 1.58e7, 0.18, '0.78 to 6.76', 10.36), ('g1M', 1e6, 2.23e8, 2.13, '11.0 to 95.1', 74.3)):
    t = trav * 20 / (60 * 2.5e9) * 1e3; before = max(t, shade) + 0.25 * min(t, shade)
    rm = CB.term('%s/bh_%s' % (ND, tag), N / 32, 'max'); rs = CB.term('%s/bh_%s' % (ND, tag), N / 32, 'sum'); cmx = rm['cycles'] / 2.5e6; csm = rs['cycles'] / 2.5e6
    am = max(t, cmx) + 0.25 * min(t, cmx); asum = max(t, csm) + 0.25 * min(t, csm)
    P('| %d | %.1f | %.2f | %s | %.2f, %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f |' % (N, ms, before, brk, cmx, csm, t, am, asum, np.log(before / ms), np.log(am / ms), np.log(asum / ms)))
open('model/results_callback_term.md', 'w').write('# SM callback term, evaluation (queue 13), generated by model/callback_eval.py\n\n' + '\n'.join(L) + '\n')
