"""Blind test pipeline for model v1 (2026-11-18). One command per new dataset:

    python3 -I model/priorworks/blind_pipeline.py DATA.csv --name NAME [--n N] [--ks 5,50,128] [--gpu 1] [--no-push] [--no-gpu]

CALIBRATED version (Phase 13, predictions_phase13.md): step 1b measures the callback cost on a random 2 percent (and a 10 percent diagnostic) of the queries before the predictions are committed, and the full run is predicted with the unchanged v1 form (two ways: S = real build against stub build, D = per candidate cost solved from the subset launch).
Order of events (each step is written to disk and committed BEFORE the next one happens, so the commit history is the proof that nothing was seen first):
  1. CPU replay (no GPU): the F245 recipe on the dataset: RTNN radius per K (smallest radius at which 99 percent of 3,000 sampled queries have K other points inside), then the replay counts
     C (points in the query's cube of that radius) and S (points in its ball) for every query with a kd tree, and model v1's RTNN launch time (model_v1_rtnn.py, frozen form, callback cost = the uniform reference's, nothing fitted).
  2. COMMIT 1: model/blind/NAME/predictions_<ts>.md and .json (the numbers above, the data file's sha256 and row count, the code versions) are committed and pushed. Nothing has run on the GPU yet.
  3. hubs (CLOVER) is timed on the GPU (device span, 3 reps). Model v1 does not predict hubs.
  4. COMMIT 2: the winner call = "RTNN if v1's predicted launch time is below the measured hubs time, else hubs", written to model/blind/NAME/call_<ts>.md and committed before RTNN runs.
  5. RTNN knn mode is timed (3 runs, device scope = sort + geometry + search compute as in F247; the model predicts the search launch only, the sort and geometry are reported but not predicted).
  6. results_<ts>.md / .json: measured against predicted per K, and the winner (RTNN with its sort and geometry, against hubs) against the call. Committed and pushed.
Scope: RTNN only (TrueKNN needs per round warp cycle probes that v1 uses and this pipeline does not measure). One GPU, device scope, K in {5, 50, 128} (RTNN's K is a compile time macro: one build per K).
Not blind for any dataset whose F245 numbers are already known: the script only guarantees the ORDER for new data. What may be published: this script and the prediction and result files (numbers, no NVIDIA code or dumps).
Default paths (override with environment variables): RTNN builds PW/rebuild/rtnn{,k5,k128}_rt_2*/bin/optixNSearch, hubs bench PW/clvbench/benchh_clv3_*, PW=/local/scratch/a/cashman3/thirdparty/pw."""
import sys, os, re, json, time, glob, hashlib, argparse, statistics, subprocess
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_v1_rtnn as m1
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
CBREF = 1.0110154  # cycles per candidate, the uniform reference of score_model_v1.py (uni3_ref.csv, radius 0.0232, K 50 builds), recomputed 2026-11-18; used for every K as in F250
PW = os.environ.get('BLIND_PW', '/local/scratch/a/cashman3/thirdparty/pw')
def sh(cmd, **kw): return subprocess.run(cmd, capture_output=True, text=True, **kw)
def load(path, n):
    p = np.array(open(path).read().replace(',', ' ').split(), dtype=np.float64).reshape(-1, 3)
    if n: p = p[:n]
    return p.astype(np.float32).astype(np.float64)
def radius99(tree, pts, K, seed=5, nq=3000):
    q = pts[np.random.default_rng(seed).choice(len(pts), min(nq, len(pts)), replace=False)]
    d, _ = tree.query(q, k=K + 1, workers=-1); return float(np.quantile(d[:, K], 0.99))      # column 0 is the query itself
NOCOMMIT = False
def commit(paths, msg, push):
    if NOCOMMIT: return 'nocommit'
    sh(['git', '-C', REPO, 'add'] + paths); r = sh(['git', '-C', REPO, 'commit', '-q', '-m', msg + '\n\nCo-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>', '--'] + paths)
    if r.returncode != 0: raise SystemExit('commit failed: ' + r.stdout + r.stderr)
    h = sh(['git', '-C', REPO, 'rev-parse', '--short', 'HEAD']).stdout.strip()
    if push:
        r = sh([os.path.join(REPO, 'notes', 'push.sh')])
        print('push:', (r.stdout + r.stderr).strip().splitlines()[-1] if (r.stdout + r.stderr).strip() else 'no output', flush=True)
    return h
def idle(gpu):
    r = sh(['nvidia-smi', '-i', str(gpu), '--query-compute-apps=pid', '--format=csv,noheader']).stdout.strip()
    if r: raise SystemExit('GPU %s busy, stop: %s' % (gpu, r))
def run(cmd, env, gpu, tmo):
    idle(gpu); t0 = time.time(); r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=tmo + 2); return r.returncode, r.stdout + r.stderr, time.time() - t0
def exe(K, kind='rt'):
    pat = PW + '/rebuild/rtnn' + ('' if K == 50 else 'k%d' % K) + '_rt_' + ('stub_' if kind == 'stub' else '') + '2*/bin/optixNSearch'; c = sorted(glob.glob(pat))
    if not c: raise SystemExit('no RTNN %s build for K %d (%s)' % (kind, K, pat))
    return c[-1]
def rtnn_times(txt):
    g = lambda k: (float(re.search(k + r': ([\d.e+-]+) ms', txt)[1]) if re.search(k + r': ([\d.e+-]+) ms', txt) else None)
    return dict(sort=g('time sort and/or partition queries'), sortp=g('time sort points'), geom=g('time create and upload geometry'), search=g('time search compute'))
def calibrate(data, pts, N, tree, K, r, frac, env, gpu, tmpdir, tag):
    """Run RTNN real and stub on a random subset of frac of the N queries (points: the whole file); returns the measured quantities and the two v1 callback corrections (S and D)."""
    nq = max(32, int(round(frac * N))); idx = np.sort(np.random.default_rng(11).choice(N, nq, replace=False))
    lines = open(data).read().splitlines()[:N]; qf = os.path.join(tmpdir, 'calq_%s.csv' % tag); open(qf, 'x').write('\n'.join(lines[i] for i in idx) + '\n')
    try:
        sub = pts[idx]; C, S = m1.replay_counts(sub, r, tree); Ce = np.where(S > K, K * (C + 1) / (S + 1), C); qm = float((1.10 * Ce + 1).mean()); Cm = float(C.mean()); out = dict(frac=frac, nq=nq, q_mean_sub=qm, C_mean_sub=Cm, runs={})
        for kind in ('rt', 'stub'):
            rs = []
            for i in range(3):
                rc, txt, wt = run([exe(K, kind), '-f', data, '-q', qf, '-sm', 'knn', '-r', repr(r), '-k', str(K), '-p', '0', '-s', '0', '-m', '0', '-d', '0'], env, gpu, 300)
                t = rtnn_times(txt); t['rc'] = rc; rs.append(t)
                if rc != 0 or None in (t['search'], t['geom'], t['sort']): break
            out['runs'][kind] = rs
        ok = {k: [x for x in v if x['rc'] == 0 and x['search'] is not None and x['geom'] is not None and x['sort'] is not None] for k, v in out['runs'].items()}
        if not ok['rt'] or not ok['stub']: out['failed'] = True; return out
        Tr = statistics.median(x['search'] for x in ok['rt']); Ts = statistics.median(x['search'] for x in ok['stub']); cyc = 1e-3 * m1.SM * m1.CLK
        out.update(T_real_ms=Tr, T_stub_ms=Ts, fixed_ms=statistics.median(x['sort'] + (x['sortp'] or 0.0) + x['geom'] for x in ok['rt']))
        cbS = max(0.0, (Tr - Ts) * cyc / (nq * 1.10 * Cm)); out['cb_S_raw'] = (Tr - Ts) * cyc / (nq * 1.10 * Cm); out['cb_S'] = cbS; out['floor_binds_S'] = bool(out['cb_S_raw'] < 0)
        out['d_S'] = max(cbS - CBREF, -4.9)
        effD = max(0.0, (Tr * cyc / nq - 110.0) / qm); out['eff_D_raw'] = (Tr * cyc / nq - 110.0) / qm; out['d_D'] = effD - 4.9
        return out
    finally:
        os.remove(qf)
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('data'); ap.add_argument('--name', required=True); ap.add_argument('--n', type=int, default=0); ap.add_argument('--ks', default='5,50,128')
    ap.add_argument('--gpu', default='1'); ap.add_argument('--no-push', action='store_true'); ap.add_argument('--no-gpu', action='store_true', help='stop after commit 1 (predictions only; no calibration run, so the calibrated columns are empty)')
    ap.add_argument('--frac', type=float, default=0.02, help='primary calibration fraction of the queries (1 to 2 percent)'); ap.add_argument('--no-commit', action='store_true', help='smoke test only: write files, make no commit'); ap.add_argument('--frac2', type=float, default=0.10, help='diagnostic fraction (0 = skip)'); a = ap.parse_args()
    global NOCOMMIT; NOCOMMIT = a.no_commit
    Ks = [int(k) for k in a.ks.split(',')]; ts = time.strftime('%Y%m%dT%H%M%S'); out = os.path.join(REPO, 'model', 'blind', a.name); os.makedirs(out, exist_ok=True); rel = lambda f: os.path.relpath(f, REPO)
    pts = load(a.data, a.n); N = len(pts); sha = hashlib.sha256(open(a.data, 'rb').read()).hexdigest(); tree = cKDTree(pts); pred = {}
    env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=a.gpu)
    if 'RTNN_DRAM_GB' not in env:
        tot = int(sh(['nvidia-smi', '-i', a.gpu, '--query-gpu=memory.total', '--format=csv,noheader,nounits']).stdout.strip()); env['RTNN_DRAM_GB'] = '%.4f' % min(0.95 * tot / 1024, 11.5958)   # RTNN builds a grid sized to this; on a 48 GB card it faults (F167), so capped at the 4070 Ti's 11.6 GB
    tmpdir = '/dev/shm'
    for K in Ks:
        r = radius99(tree, pts, K); C, S = m1.replay_counts(pts, r, tree); v = m1.predict(N, K, C, S); pred[K] = dict(radius=r, uncal=v, **v)
        print('replay K', K, 'r', r, 'uncalibrated launch ms %.3f' % v['ms'], flush=True)
        if a.no_gpu: continue
        for tag, fr in (('p', a.frac), ('d', a.frac2)):
            if not fr: continue
            c = calibrate(a.data, pts, N, tree, K, r, fr, env, a.gpu, tmpdir, '%s_%s_%d' % (a.name, tag, K)); pred[K]['cal_' + tag] = c
            if c.get('failed'): print('calibration', tag, 'K', K, 'FAILED', flush=True); continue
            for var in ('S', 'D'):
                p = m1.predict(N, K, C, S, c['d_' + var]); c['pred_' + var] = p; c['ms_' + var] = p['ms']
            print('calibration %s K %d: nq %d real %.3f stub %.3f ms; d_S %.2f d_D %.2f -> launch S %.3f D %.3f ms; fixed %.1f ms' % (tag, K, c['nq'], c['T_real_ms'], c['T_stub_ms'], c['d_S'], c['d_D'], c['ms_S'], c['ms_D'], c['fixed_ms']), flush=True)
        cpp, cdd = pred[K].get('cal_p', {}), pred[K].get('cal_d', {})
        if 'T_real_ms' in cpp and 'T_real_ms' in cdd and cdd['nq'] != cpp['nq']:   # L: POST HOC, not pre registered (F259): straight line through the two real subset launch times, evaluated at all N queries; does not feed the call
            sl = (cdd['T_real_ms'] - cpp['T_real_ms']) / (cdd['nq'] - cpp['nq']); pred[K]['L_ms'] = max(cpp['T_real_ms'] + sl * (N - cpp['nq']), 1e-6)
        pc = pred[K].get('cal_p', {})
        pred[K]['ms'] = pc['ms_S'] if 'ms_S' in pc else pred[K]['uncal']['ms']       # the call uses variant S (v1's own definition of the callback cost); falls back to uncalibrated if the calibration failed
        pred[K]['fixed_ms'] = pc.get('fixed_ms', 0.0)
    pj = os.path.join(out, 'predictions_%s.json' % ts); pm = os.path.join(out, 'predictions_%s.md' % ts)
    json.dump(dict(name=a.name, data=os.path.basename(a.data), sha256_prefix12=sha[:12], N=N, ts=ts, cbref=CBREF, frac=a.frac, frac2=a.frac2, model='v1 (FORM_combined_v1.md) unchanged; callback cost measured on a query subset (variants S, D; predictions_phase13.md), K generalized from 50', pred={str(k): v for k, v in pred.items()}), open(pj, 'x'), indent=1)
    with open(pm, 'x') as f:
        f.write('# Model v1 predictions for %s, written %s BEFORE any hubs or full RTNN run\n\nData: %s (%d points used, sha256 prefix %s). Model: model_v1_rtnn.py, frozen v1 form, replay counts on the CPU, RTNN knn mode at the 99 percent radius, search launch only, SM 142, 2.5 GHz. Callback cost measured on %.0f percent of the queries (RTNN real and stub builds, 3 reps; diagnostic %.0f percent). S = real minus stub as v1 defines it (floored at 0), D = per candidate cost solved from the real subset launch. "uncal" = cb = cbref (F250). L = POST HOC (not pre registered, F259): straight line through the real subset launch times at the two fractions, evaluated at all N queries; reported only, never used for the call. Call = S launch + calibrated fixed part (query sort + point sort + geometry of the subset run) against hubs.\n\n| K | radius | mean q | max q | uncal ms | chain ms | S ms (%.0f%%) | D ms (%.0f%%) | S ms (%.0f%%) | D ms (%.0f%%) | fixed ms (cal) | L ms (post hoc) |\n|---|---|---|---|---|---|---|---|---|---|---|---|\n' % (a.name, ts, os.path.basename(a.data), N, sha[:12], 100 * a.frac, 100 * a.frac2, 100 * a.frac, 100 * a.frac, 100 * a.frac2, 100 * a.frac2))
        for K in Ks:
            v = pred[K]; cp, cd = v.get('cal_p', {}), v.get('cal_d', {}); g = lambda c, k: ('%.3f' % c[k]) if k in c else '-'
            f.write('| %d | %.6g | %.2f | %.1f | %.3f | %.3f | %s | %s | %s | %s | %s | %s |\n' % (K, v['radius'], v['uncal']['mean_q'], v['uncal']['max_q'], v['uncal']['ms'], v['uncal']['chain_ms'], g(cp, 'ms_S'), g(cp, 'ms_D'), g(cd, 'ms_S'), g(cd, 'ms_D'), g(cp, 'fixed_ms'), g(v, 'L_ms')))
    h1 = commit([rel(pj), rel(pm)], 'Blind predictions for %s (model v1, callback calibrated on a query subset, before any hubs or full run)' % a.name, not a.no_push); print('COMMIT 1 (predictions):', h1, flush=True)
    if a.no_gpu: return
    hb = sorted(glob.glob(PW + '/clvbench/benchh_clv3_*[0-9]'))[-1]; res = {}
    for K in Ks:
        rc, txt, wt = run([hb, '2', a.data, str(N), str(N), str(K), '3', '0', '7'], env, a.gpu, 600); ns = [int(m[1]) / 1e6 for m in re.finditer(r'^(\d+), rep', txt, re.M)]; res[K] = dict(hubs_ms=statistics.median(ns) if ns else None, hubs_rc=rc)
    callB = {K: ('RTNN' if res[K]['hubs_ms'] and pred[K]['ms'] + pred[K]['fixed_ms'] < res[K]['hubs_ms'] else 'hubs') for K in Ks}; callA = {K: ('RTNN' if res[K]['hubs_ms'] and pred[K]['ms'] < res[K]['hubs_ms'] else 'hubs') for K in Ks}
    cm = os.path.join(out, 'call_%s.md' % ts)
    with open(cm, 'x') as f:
        f.write('# Winner call for %s, written %s after the hubs timing and BEFORE any full RTNN run\n\nRule (the call): RTNN if the S launch prediction + the calibrated fixed part is below hubs\' measured device time, else hubs. Call A (launch only) is shown for comparison.\n\n| K | predicted RTNN launch ms (S) | calibrated fixed ms | hubs ms (measured) | call | call A |\n|---|---|---|---|---|---|\n' % (a.name, ts))
        for K in Ks: f.write('| %d | %.3f | %.1f | %s | %s | %s |\n' % (K, pred[K]['ms'], pred[K]['fixed_ms'], '-' if res[K]['hubs_ms'] is None else '%.2f' % res[K]['hubs_ms'], callB[K], callA[K]))
    h2 = commit([rel(cm)], 'Blind winner call for %s (after hubs timing, before the full RTNN run)' % a.name, not a.no_push); print('COMMIT 2 (call):', h2, flush=True)
    for K in Ks:
        runs = []
        for i in range(3):
            rc, txt, wt = run([exe(K), '-f', a.data, '-sm', 'knn', '-r', repr(pred[K]['radius']), '-k', str(K), '-p', '0', '-s', '0', '-m', '0', '-d', '0'], env, a.gpu, 300)
            t = rtnn_times(txt); t['rc'] = rc; runs.append(t)
            if rc != 0: break
        ok = [r for r in runs if r['rc'] == 0 and None not in (r['sort'], r['geom'], r['search'])]
        res[K].update(rtnn_runs=runs, search_ms=statistics.median(r['search'] for r in ok) if ok else None, fixed_ms=statistics.median(r['sort'] + r['geom'] for r in ok) if ok else None, total_dev_ms=statistics.median(r['sort'] + r['geom'] + r['search'] for r in ok) if ok else None)
    rj = os.path.join(out, 'results_%s.json' % ts); rm = os.path.join(out, 'results_%s.md' % ts); json.dump({str(k): dict(v, callB=callB[k], callA=callA[k]) for k, v in res.items()}, open(rj, 'x'), indent=1)
    with open(rm, 'x') as f:
        f.write('# Results for %s (%s). Predictions commit %s, call commit %s\n\nDevice scope as F247. Ratio = measured search compute / predicted launch (within a factor 2 = 0.5 to 2). Winner = RTNN (sort + geometry + search) against hubs. S, D = calibrated on %.0f percent of the queries.\n\n| K | uncal ms | S ms | D ms | measured search ms | L ms (post hoc) | ratio S | ratio D | ratio L (post hoc) | fixed cal / measured ms | RTNN total ms | hubs ms | call | call A | actual winner | call right |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n' % (a.name, ts, h1, h2, 100 * a.frac))
        for K in Ks:
            r = res[K]; win = ('RTNN' if r['total_dev_ms'] < r['hubs_ms'] else 'hubs') if r.get('total_dev_ms') and r['hubs_ms'] else '-'; c = pred[K].get('cal_p', {}); m = r['search_ms']
            f.write('| %d | %.3f | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n' % (K, pred[K]['uncal']['ms'], '%.3f' % c['ms_S'] if 'ms_S' in c else '-', '%.3f' % c['ms_D'] if 'ms_D' in c else '-', '-' if not m else '%.3f' % m, '-' if 'L_ms' not in pred[K] else '%.3f' % pred[K]['L_ms'],
                '-' if not m or 'ms_S' not in c else '%.2f' % (m / c['ms_S']), '-' if not m or 'ms_D' not in c else '%.2f' % (m / c['ms_D']), '-' if not m or 'L_ms' not in pred[K] else '%.2f' % (m / pred[K]['L_ms']), '%.1f / %s' % (pred[K]['fixed_ms'], '-' if r['fixed_ms'] is None else '%.1f' % r['fixed_ms']),
                '-' if not r['total_dev_ms'] else '%.2f' % r['total_dev_ms'], '-' if not r['hubs_ms'] else '%.2f' % r['hubs_ms'], callB[K], callA[K], win, 'yes' if win == callB[K] else 'no'))
    h3 = commit([rel(rj), rel(rm)], 'Blind results for %s (timed after the committed predictions and call)' % a.name, not a.no_push); print('COMMIT 3 (results):', h3, flush=True); print(open(rm).read())
main()
