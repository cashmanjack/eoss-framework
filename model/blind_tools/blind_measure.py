"""Blind test step 2: measurement of RTNN's "time search compute" (the quantity model v1 predicts). GPU, OptiX only, no TTU code.

    python3 -I model/blind_tools/blind_measure.py NAME --data DATA.csv --rtnn 50=/path/optixNSearch [--rtnn 5=... --rtnn 128=...] [--gpu 1] [--reps 3] [--dry-run] [--no-push] [--no-commit]

Refuses to start unless: the frozen files are unchanged, the newest predictions_*.json of NAME is tracked, unmodified and contained in origin/main, the data
file's hash matches the one in it. RTNN's K is a compile time macro, so one build per K is given with --rtnn K=path; a K without a build is recorded as missing
(never as a pass). Per run: idle check (nvidia-smi PIDs and utilization, never names), a 2 s warmup process, the program under `timeout -k 2`, SM clock sampled every 50 ms.
Flags: -sm knn -r <radius from the prediction file> -k K -p 0 -s 0 -m 0 -d 0 (as the F245 pipeline). Writes model/blind/NAME/measured_<ts>.json (never overwritten), commits and pushes it.
Uses the helpers of ttu/bench/Base_1/base_common.py (tethys paths for the OptiX library and CUDA)."""
import os, re, sys, json, time, argparse, statistics, subprocess
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import blind_common as bc
def parse_args():
    ap = argparse.ArgumentParser(); ap.add_argument('name'); ap.add_argument('--data', required=True); ap.add_argument('--rtnn', action='append', default=[], help='K=path to an optixNSearch built with KNN=K')
    ap.add_argument('--gpu', default='1'); ap.add_argument('--reps', type=int, default=3); ap.add_argument('--dry-run', action='store_true'); ap.add_argument('--no-push', action='store_true'); ap.add_argument('--no-commit', action='store_true')
    return ap.parse_args()
a = parse_args(); os.environ['BASE_GPU'] = a.gpu
sys.path.insert(0, os.path.join(bc.REPO, 'ttu', 'bench', 'Base_1')); import base_common as B
def t(txt, key):
    m = re.search(r'time ' + key + r': ([0-9.eE+-]+) ms', txt); return float(m.group(1)) if m else None
def main():
    bc.verify_frozen(); label = 'DRY RUN' if a.dry_run else 'BLIND'
    out = os.path.join(bc.REPO, 'model', 'blind', ('dry_run_' if a.dry_run else '') + a.name)
    preds = sorted(f for f in os.listdir(out) if f.startswith('predictions_') and f.endswith('.json')) if os.path.isdir(out) else []
    if not preds: raise SystemExit('no predictions for %s: predict first' % a.name)
    pf = os.path.join(out, preds[-1]); rel = os.path.relpath(pf, bc.REPO); ok, why = bc.committed_and_pushed(rel)
    if not ok: raise SystemExit('prediction file %s is not committed and pushed (%s): refusing to measure' % (rel, why))
    P = json.load(open(pf))
    if P['label'] != label: raise SystemExit('label mismatch: prediction is %s, this run is %s' % (P['label'], label))
    if bc.sha12(a.data) != P['data_sha256_12']: raise SystemExit('data file hash differs from the prediction file: refusing')
    builds = {}
    for s in a.rtnn: k, p = s.split('=', 1); builds[int(k)] = p
    warm = '/dev/shm/blind_warm'
    if not os.path.exists(warm):
        r = bc.sh([os.path.expanduser('~/opt/cuda-12.8/bin/nvcc'), '-O2', '-arch=sm_89', '-o', warm, os.path.join(bc.REPO, 'ttu', 'bench', 'Base_1', 'warm.cu')])
        if r.returncode != 0: raise SystemExit('warm build failed: ' + r.stderr[-300:])
    B.WARM_BIN = warm
    env = B.env(extra={'RTNN_DRAM_GB': '11.5958'}); gi = B.gpu_info(); ts = time.strftime('%Y%m%dT%H%M%S'); res = {}
    for K in P['ks']:
        r = P['pred'][str(K)]['radius']
        if K not in builds: res[str(K)] = dict(missing='no RTNN build for K %d given' % K); continue
        cmd = [builds[K], '-f', a.data, '-sm', 'knn', '-r', repr(r), '-k', str(K), '-p', '0', '-s', '0', '-m', '0', '-d', '0']; runs = []
        for i in range(a.reps):
            x = B.run(cmd, os.path.dirname(builds[K]), env, 600); txt = x.pop('stdout') + '\n' + x['stderr']
            x['search_ms'] = t(txt, 'search compute'); x['sort_ms'] = t(txt, 'sort and/or partition queries'); x['geom_ms'] = t(txt, 'create and upload geometry'); x['total_ms'] = t(txt, 'total search time')
            m = re.search(r'KERNEL queries \d+ ms ([0-9.]+)', txt); x['kernel_ms'] = float(m.group(1)) if m else None
            runs.append(x)
            if x['rc'] != 0 or x['search_ms'] is None: break
        good = [x for x in runs if x['rc'] == 0 and x['search_ms'] is not None]; sm = [x['search_ms'] for x in good]
        res[str(K)] = dict(cmd=' '.join(cmd).replace(B.HOME, '~'), runs=runs, ok_runs=len(good), search_ms_median=statistics.median(sm) if sm else None, search_ms_min=min(sm) if sm else None, search_ms_max=max(sm) if sm else None,
                           all_runs_full_clock=bool(good) and all(x['clock']['full_clock'] for x in good), lowest_clock_after_start=min((x['clock']['min_after_start'] or 0) for x in good) if good else None)
        print('K %d: search compute median %s ms (%d ok runs, full clock %s)' % (K, res[str(K)]['search_ms_median'], len(good), res[str(K)]['all_runs_full_clock']), flush=True)
    doc = dict(label=label, name=a.name, ts=ts, predictions_file=rel, predictions_commit=ok and why, data_sha256_12=P['data_sha256_12'], gpu=gi, quantity='RTNN time search compute (ms), median of reps',
               tool_sha12=bc.sha12(os.path.abspath(__file__)), repo_head=bc.short_head(), rows=res)
    mj = os.path.join(out, 'measured_%s.json' % ts)
    with open(mj, 'x') as f: json.dump(doc, f, indent=1)
    if a.no_commit: print('written (no commit):', mj); return
    print('COMMIT (measured):', bc.commit([mj], '%s: measured RTNN search compute for %s (after the committed predictions)' % (label, a.name), not a.no_push), flush=True)
main()
