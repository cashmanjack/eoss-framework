"""Blind test pipeline for model v1 (2026-11-18). One command per new dataset:

    python3 -I model/priorworks/blind_pipeline.py DATA.csv --name NAME [--n N] [--ks 5,50,128] [--gpu 1] [--no-push] [--no-gpu]

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
PW = os.environ.get('BLIND_PW', '/local/scratch/a/cashman3/thirdparty/pw')
def sh(cmd, **kw): return subprocess.run(cmd, capture_output=True, text=True, **kw)
def load(path, n):
    p = np.array(open(path).read().replace(',', ' ').split(), dtype=np.float64).reshape(-1, 3)
    if n: p = p[:n]
    return p.astype(np.float32).astype(np.float64)
def radius99(tree, pts, K, seed=5, nq=3000):
    q = pts[np.random.default_rng(seed).choice(len(pts), min(nq, len(pts)), replace=False)]
    d, _ = tree.query(q, k=K + 1, workers=-1); return float(np.quantile(d[:, K], 0.99))      # column 0 is the query itself
def commit(paths, msg, push):
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
def exe(K):
    pat = PW + '/rebuild/rtnn' + ('' if K == 50 else 'k%d' % K) + '_rt_2*/bin/optixNSearch'; c = sorted(glob.glob(pat))
    if not c: raise SystemExit('no RTNN build for K %d (%s)' % (K, pat))
    return c[-1]
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('data'); ap.add_argument('--name', required=True); ap.add_argument('--n', type=int, default=0); ap.add_argument('--ks', default='5,50,128')
    ap.add_argument('--gpu', default='1'); ap.add_argument('--no-push', action='store_true'); ap.add_argument('--no-gpu', action='store_true', help='stop after commit 1 (predictions only)'); a = ap.parse_args()
    Ks = [int(k) for k in a.ks.split(',')]; ts = time.strftime('%Y%m%dT%H%M%S'); out = os.path.join(REPO, 'model', 'blind', a.name); os.makedirs(out, exist_ok=True); rel = lambda f: os.path.relpath(f, REPO)
    pts = load(a.data, a.n); N = len(pts); sha = hashlib.sha256(open(a.data, 'rb').read()).hexdigest(); tree = cKDTree(pts); pred = {}
    for K in Ks:
        r = radius99(tree, pts, K); C, S = m1.replay_counts(pts, r, tree); pred[K] = dict(radius=r, **m1.predict(N, K, C, S)); print('replay K', K, 'r', r, 'pred launch ms %.3f' % pred[K]['ms'], flush=True)
    pj = os.path.join(out, 'predictions_%s.json' % ts); pm = os.path.join(out, 'predictions_%s.md' % ts)
    json.dump(dict(name=a.name, data=os.path.basename(a.data), sha256=sha, N=N, ts=ts, model='v1 (FORM_combined_v1.md), callback cost = uniform reference, K generalized from 50', pred={str(k): v for k, v in pred.items()}), open(pj, 'x'), indent=1)
    with open(pm, 'x') as f:
        f.write('# Model v1 predictions for %s, written %s BEFORE any timed run\n\nData: %s (%d points used, sha256 %s). Model: model_v1_rtnn.py, frozen v1 form, replay counts on the CPU, RTNN knn mode at the 99 percent radius, search launch only (no sort, no geometry, no result copy), SM 142, 2.5 GHz, cb = cbref.\n\n| K | radius | mean q | max q | throughput ms | chain ms | predicted launch ms |\n|---|---|---|---|---|---|---|\n' % (a.name, ts, os.path.basename(a.data), N, sha))
        for K in Ks: v = pred[K]; f.write('| %d | %.6g | %.2f | %.1f | %.3f | %.3f | %.3f |\n' % (K, v['radius'], v['mean_q'], v['max_q'], v['thr_ms'], v['chain_ms'], v['ms']))
    h1 = commit([rel(pj), rel(pm)], 'Blind predictions for %s (model v1, before any timed run)' % a.name, not a.no_push); print('COMMIT 1 (predictions):', h1, flush=True)
    if a.no_gpu: return
    env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=a.gpu)
    if 'RTNN_DRAM_GB' not in env:
        tot = int(sh(['nvidia-smi', '-i', a.gpu, '--query-gpu=memory.total', '--format=csv,noheader,nounits']).stdout.strip()); env['RTNN_DRAM_GB'] = '%.4f' % (0.95 * tot / 1024)
    hb = sorted(glob.glob(PW + '/clvbench/benchh_clv3_*[0-9]'))[-1]; res = {}
    for K in Ks:
        rc, txt, wt = run([hb, '2', a.data, str(N), str(N), str(K), '3', '0', '7'], env, a.gpu, 600); ns = [int(m[1]) / 1e6 for m in re.finditer(r'^(\d+), rep', txt, re.M)]; res[K] = dict(hubs_ms=statistics.median(ns) if ns else None, hubs_rc=rc)
    call = {K: ('RTNN' if res[K]['hubs_ms'] and pred[K]['ms'] < res[K]['hubs_ms'] else 'hubs') for K in Ks}
    cm = os.path.join(out, 'call_%s.md' % ts)
    with open(cm, 'x') as f:
        f.write('# Winner call for %s, written %s after the hubs timing and BEFORE any RTNN run\n\nRule: RTNN if model v1\'s predicted launch time is below hubs\' measured device time, else hubs (v1 does not predict hubs; the call ignores RTNN\'s sort and geometry time, which v1 does not model).\n\n| K | predicted RTNN launch ms | hubs ms (measured) | call |\n|---|---|---|---|\n' % (a.name, ts))
        for K in Ks: f.write('| %d | %.3f | %s | %s |\n' % (K, pred[K]['ms'], '-' if res[K]['hubs_ms'] is None else '%.2f' % res[K]['hubs_ms'], call[K]))
    h2 = commit([rel(cm)], 'Blind winner call for %s (after hubs timing, before RTNN)' % a.name, not a.no_push); print('COMMIT 2 (call):', h2, flush=True)
    for K in Ks:
        runs = []
        for i in range(3):
            rc, txt, wt = run([exe(K), '-f', a.data, '-sm', 'knn', '-r', repr(pred[K]['radius']), '-k', str(K), '-p', '0', '-s', '0', '-m', '0', '-d', '0'], env, a.gpu, 300)
            g = lambda k: (float(re.search(k + r': ([\d.e+-]+) ms', txt)[1]) if re.search(k + r': ([\d.e+-]+) ms', txt) else None)
            runs.append(dict(rc=rc, sort=g('time sort and/or partition queries'), geom=g('time create and upload geometry'), search=g('time search compute')))
            if rc != 0: break
        ok = [r for r in runs if r['rc'] == 0 and None not in (r['sort'], r['geom'], r['search'])]
        res[K].update(rtnn_runs=runs, search_ms=statistics.median(r['search'] for r in ok) if ok else None, total_dev_ms=statistics.median(r['sort'] + r['geom'] + r['search'] for r in ok) if ok else None)
    rj = os.path.join(out, 'results_%s.json' % ts); rm = os.path.join(out, 'results_%s.md' % ts); json.dump({str(k): v for k, v in res.items()}, open(rj, 'x'), indent=1)
    with open(rm, 'x') as f:
        f.write('# Results for %s (%s). Predictions commit %s, call commit %s\n\nDevice scope as F247. Ratio = measured search compute / predicted launch (within a factor 2 = 0.5 to 2). Winner = RTNN (sort + geometry + search) against hubs.\n\n| K | predicted launch ms | measured search ms | ratio | RTNN total ms | hubs ms | call | actual winner | call right |\n|---|---|---|---|---|---|---|---|---|\n' % (a.name, ts, h1, h2))
        for K in Ks:
            r = res[K]; win = ('RTNN' if r['total_dev_ms'] < r['hubs_ms'] else 'hubs') if r.get('total_dev_ms') and r['hubs_ms'] else '-'
            f.write('| %d | %.3f | %s | %s | %s | %s | %s | %s | %s |\n' % (K, pred[K]['ms'], '-' if not r['search_ms'] else '%.3f' % r['search_ms'], '-' if not r['search_ms'] else '%.2f' % (r['search_ms'] / pred[K]['ms']), '-' if not r['total_dev_ms'] else '%.2f' % r['total_dev_ms'], '-' if not r['hubs_ms'] else '%.2f' % r['hubs_ms'], call[K], win, 'yes' if win == call[K] else 'no'))
    h3 = commit([rel(rj), rel(rm)], 'Blind results for %s (timed after the committed predictions and call)' % a.name, not a.no_push); print('COMMIT 3 (results):', h3, flush=True); print(open(rm).read())
main()
