"""Blind test step 1: prediction (model v1, model/FROZEN.md; rule: model/blind/RULE.md). CPU only, no GPU run, no measurement of any kind.

    python3 -I model/blind_tools/blind_predict.py DATA.csv --name NAME [--ks 5,50,128] [--n N] [--gpu 1 | --sm 128] [--dry-run] [--no-push] [--no-commit]

Reads the dataset (3 columns, comma or space separated, one point per line, queries = points), then for every K:
  radius = the F245 rule (99 percent of 3,000 sampled queries have K other points inside), replay counts C and S (kd tree), model v1 RTNN search launch
  (model_v1_rtnn.predict with its frozen constants and cb - cbref = 0; only SM is set, from the GPU's specification).
Writes model/blind/NAME/predictions_<ts>.json and .md (never overwritten), commits and pushes them. Refuses to run if a frozen file changed, if NAME already
has predictions, or if the working tree has uncommitted changes to the frozen files. --dry-run writes to model/blind/dry_run_NAME and labels every file DRY RUN."""
import os, sys, json, time, argparse, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'priorworks'))
import blind_common as bc
import model_v1_rtnn as m1          # the frozen model, imported not copied
from scipy.spatial import cKDTree

def load(path, n):
    p = np.array(open(path).read().replace(',', ' ').split(), dtype=np.float64).reshape(-1, 3)
    if n: p = p[:n]
    return p.astype(np.float32).astype(np.float64)
def radius99(tree, pts, K, seed=5, nq=3000):                       # identical to blind_pipeline.radius99 (the F245 rule)
    q = pts[np.random.default_rng(seed).choice(len(pts), min(nq, len(pts)), replace=False)]
    d, _ = tree.query(q, k=K + 1, workers=-1); return float(np.quantile(d[:, K], 0.99))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('data'); ap.add_argument('--name', required=True); ap.add_argument('--ks', default='5,50,128'); ap.add_argument('--n', type=int, default=0)
    ap.add_argument('--gpu', default='1'); ap.add_argument('--sm', default=None); ap.add_argument('--dry-run', action='store_true'); ap.add_argument('--no-push', action='store_true'); ap.add_argument('--no-commit', action='store_true')
    a = ap.parse_args(); label = 'DRY RUN' if a.dry_run else 'BLIND'
    frozen = bc.verify_frozen()
    sm = bc.sm_for(a.gpu, a.sm); m1.SM = sm                                  # the only parameter passed into the frozen model
    if m1.CLK != bc.CLK: raise SystemExit('model clock constant differs from the frozen 2.5e9')
    out = os.path.join(bc.REPO, 'model', 'blind', ('dry_run_' if a.dry_run else '') + a.name)
    if os.path.isdir(out) and any(f.startswith('predictions_') for f in os.listdir(out)): raise SystemExit('%s already has predictions: a blind prediction is made once' % out)
    os.makedirs(out, exist_ok=True)
    Ks = [int(k) for k in a.ks.split(',')]; ts = time.strftime('%Y%m%dT%H%M%S')
    pts = load(a.data, a.n); N = len(pts); tree = cKDTree(pts); rows = {}
    for K in Ks:
        r = radius99(tree, pts, K); C, S = m1.replay_counts(pts, r, tree); v = m1.predict(N, K, C, S)      # cb - cbref = 0 (RULE.md item 1)
        rows[K] = dict(radius=bc.six(r), **{k: bc.six(x) for k, x in v.items()}); print('K %d radius %.6g mean C %.2f mean S %.2f predicted launch %.4f ms (throughput %.4f, chain %.4f)' % (K, r, v['mean_C'], v['mean_S'], v['ms'], v['thr_ms'], v['chain_ms']), flush=True)
    doc = dict(label=label, name=a.name, ts=ts, data=os.path.basename(a.data), data_sha256_12=bc.sha12(a.data), N=N, ks=Ks, sm=sm, clk=bc.CLK, model='model v1 (model/FROZEN.md), RTNN search launch, cb - cbref = 0',
               rule='model/blind/RULE.md', rule_sha12=bc.sha12(os.path.join(bc.REPO, 'model', 'blind', 'RULE.md')), frozen_files_checked=len(frozen), repo_head=bc.short_head(),
               tool_sha12=bc.sha12(os.path.abspath(__file__)), measured='none: no GPU run of any program on this dataset before this file is committed', pred={str(k): v for k, v in rows.items()})
    pj = os.path.join(out, 'predictions_%s.json' % ts); pm = os.path.join(out, 'predictions_%s.md' % ts)
    with open(pj, 'x') as f: json.dump(doc, f, indent=1)
    with open(pm, 'x') as f:
        f.write('# %s: model v1 predictions for %s, written %s before any measurement\n\n' % (label, a.name, ts))
        if a.dry_run: f.write('**DRY RUN: the dataset was measured before. This tests the tools, not the model. Not a blind result.**\n\n')
        f.write('Data: %s (%d points, sha256 prefix %s). Model: `model/FROZEN.md` (%d tracked files hash checked), cb - cbref = 0, SM %d, clock 2.5e9, RTNN search launch only (ms). Rule: `model/blind/RULE.md`. Repo head %s.\n\n' % (doc['data'], N, doc['data_sha256_12'], len(frozen), sm, doc['repo_head']))
        f.write('| K | radius | mean C | mean S | mean q | max q | throughput ms | chain ms | predicted ms |\n|---|---|---|---|---|---|---|---|---|\n')
        for K in Ks:
            v = rows[K]; f.write('| %d | %.6g | %.2f | %.2f | %.2f | %.1f | %.4f | %.4f | %.4f |\n' % (K, v['radius'], v['mean_C'], v['mean_S'], v['mean_q'], v['max_q'], v['thr_ms'], v['chain_ms'], v['ms']))
    if a.no_commit: print('written (no commit):', pj); return
    h = bc.commit([pj, pm], '%s: predictions for %s (model v1 frozen, CPU only, before any measurement)' % (label, a.name), not a.no_push)
    print('COMMIT (predictions):', h, flush=True)
main()
