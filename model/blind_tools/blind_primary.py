"""Blind test, PRIMARY version (model v1-S, calibrated; model/FROZEN.md) for RTNN, TrueKNN and Arkade, plus the relative rows R1 and R2. Rule: model/blind/RULE.md.

    blind_primary.py calibrate NAME --data D [--mappings rtnn,trueknn,arkade] [--ks 50] [--tk-k 16] [--dry-run]
    blind_primary.py predict   NAME --data D [same]            (CPU only; needs the committed and pushed calibration file)
    blind_primary.py measure   NAME --data D [same]            (full runs; needs committed and pushed calibration AND prediction files)
    blind_primary.py score     NAME [--dry-run]                (CPU only)
  common options: --gpu 1  --sm N  --no-push  --no-commit  --dry-run  and build paths (defaults are the tethys builds used for the 2026-11-24 baselines).

Every step writes a file that is never overwritten, commits it and pushes it before the next step may run (guards below). The only GPU runs before the predictions are the
sample procedures of RULE.md and the Arkade reference launch. All constants are in model/FROZEN.md; the frozen model_v1_rtnn.py is imported, not copied."""
import os, re, sys, json, math, time, argparse, statistics, subprocess
import numpy as np
from scipy.spatial import cKDTree
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'priorworks'))
import blind_common as bc
import model_v1_rtnn as m1                       # frozen
HOME = os.path.expanduser('~'); TP = HOME + '/thirdparty'
CBREF = 1.0110154; CLK = bc.CLK                  # FROZEN.md
SEEDS = dict(rtnn_sample=11, tk_sample=21, ark_queries=7, radius=5)   # RULE.md
LTK, LAR, FLOAT_FACTOR, PIN_MS_PER_MB, REG_SPEEDUP = 5800., 21000., 0.09, 0.33, 4.02   # FROZEN.md

def args():
    ap = argparse.ArgumentParser(); ap.add_argument('step', choices=['calibrate', 'predict', 'measure', 'score']); ap.add_argument('name')
    ap.add_argument('--data'); ap.add_argument('--mappings', default='rtnn,trueknn,arkade'); ap.add_argument('--ks', default='50'); ap.add_argument('--tk-k', type=int, default=16)
    ap.add_argument('--gpu', default='1'); ap.add_argument('--sm', default=None); ap.add_argument('--dry-run', action='store_true'); ap.add_argument('--no-push', action='store_true'); ap.add_argument('--no-commit', action='store_true')
    ap.add_argument('--rtnn', action='append', default=[], help='K=real build'); ap.add_argument('--rtnn-stub', action='append', default=[], help='K=stub build')
    ap.add_argument('--tk-bin', default=TP + '/div1/tk/ark/b/sample03-trueknn'); ap.add_argument('--ark-probe', default=TP + '/blind/ark_probe/sample01-knn')
    ap.add_argument('--ark-ref', default=TP + '/base_data/ark2e5.txt'); ap.add_argument('--knn-orig', default=TP + '/blind/knn_orig'); ap.add_argument('--knn-reg', default=TP + '/blind/knn_reg')
    a = ap.parse_args()
    if not a.rtnn: a.rtnn = ['50=' + TP + '/div1/rtnn/src/build/bin/optixNSearch']
    if not a.rtnn_stub: a.rtnn_stub = ['50=' + TP + '/div1_stub/rtnn/src/build/bin/optixNSearch']
    return a
A = args(); os.environ['BASE_GPU'] = A.gpu
sys.path.insert(0, os.path.join(bc.REPO, 'ttu', 'bench', 'Base_1')); import base_common as B
LABEL = 'DRY RUN' if A.dry_run else 'BLIND'
OUT = os.path.join(bc.REPO, 'model', 'blind', ('dry_run_' if A.dry_run else '') + A.name)
KS = [int(k) for k in A.ks.split(',')]; MAPS = [m for m in A.mappings.split(',') if m]
RT = {int(s.split('=')[0]): s.split('=', 1)[1] for s in A.rtnn}; RTS = {int(s.split('=')[0]): s.split('=', 1)[1] for s in A.rtnn_stub}
TS = time.strftime('%Y%m%dT%H%M%S'); SCR = '/dev/shm/blind_%s_%d' % (A.name, os.getpid())

# ---------------------------------------------------------------- data and replay (CPU)
def load(path):
    rows = [l.replace(',', ' ').split() for l in open(path).read().splitlines() if l.strip()]
    return np.array(rows, dtype=np.float64)
def radius99(tree, pts, K, seed=5, nq=3000):                       # RULE.md radius rule (F245), identical to blind_pipeline.radius99
    q = pts[np.random.default_rng(seed).choice(len(pts), min(nq, len(pts)), replace=False)]
    d, _ = tree.query(q, k=K + 1, workers=-1); return float(np.quantile(d[:, K], 0.99))
def r0_rule(tree, pts):                                            # RULE.md r0 (as predict_priorworks)
    ids = np.random.default_rng(5).choice(len(pts), 100, replace=False); d = tree.query(pts[ids], k=5)[0][:, 1:]; return float(d[d > 0].min())
def cube_len(tree, q, r): return tree.query_ball_point(q, r, p=np.inf, return_length=True, workers=-1).astype(float)
def tk_rounds(pts, r0, k, maxr=60):
    """F158 stop rule: per round the replay box hits, largest active box count, active count; query stops when its box count - 1 >= k; radius doubles."""
    t = cKDTree(pts); n = len(pts); active = np.ones(n, bool); r = r0; out = []
    for rd in range(maxr):
        c = np.zeros(n); c[active] = cube_len(t, pts[active], r)
        out.append(dict(radius=r, hits=float(c[active].sum()), maxc=float(c[active].max()), active=int(active.sum())))
        active = active & (c - 1 < k)
        if not active.any(): break
        r *= 2
    return out
def six(x): return float('%.6g' % x) if isinstance(x, float) else x
def write(path, text):
    with open(path, 'x') as f: f.write(text)
def wjson(path, doc):
    with open(path, 'x') as f: json.dump(doc, f, indent=1)
def latest(prefix):
    fs = sorted(f for f in os.listdir(OUT) if f.startswith(prefix) and f.endswith('.json')) if os.path.isdir(OUT) else []
    return os.path.join(OUT, fs[-1]) if fs else None
def pushed(path):
    ok, why = bc.committed_and_pushed(os.path.relpath(path, bc.REPO))
    if not ok: raise SystemExit('%s is not committed and pushed (%s): refusing' % (os.path.relpath(path, bc.REPO), why))
    return why
def commit(paths, msg):
    if A.no_commit: print('written, NOT committed:', *[os.path.relpath(p, bc.REPO) for p in paths]); return 'nocommit'
    return bc.commit(paths, '%s: %s' % (LABEL, msg), not A.no_push)
def common_doc(step):
    return dict(label=LABEL, name=A.name, step=step, ts=TS, data=os.path.basename(A.data) if A.data else None, data_sha256_12=bc.sha12(A.data) if A.data else None,
                gpu=B.gpu_info(), sm=bc.sm_for(A.gpu, A.sm), clk=CLK, repo_head=bc.short_head(), rule_sha12=bc.sha12(os.path.join(bc.REPO, 'model', 'blind', 'RULE.md')),
                frozen_files_checked=len(bc.verify_frozen()), tool_sha12=bc.sha12(os.path.abspath(__file__)), seeds=SEEDS)
def warm_ready():
    w = '/dev/shm/blind_warm'
    if not os.path.exists(w):
        r = bc.sh([HOME + '/opt/cuda-12.8/bin/nvcc', '-O2', '-arch=sm_89', '-o', w, os.path.join(bc.REPO, 'ttu', 'bench', 'Base_1', 'warm.cu')])
        if r.returncode != 0: raise SystemExit('warm build failed: ' + r.stderr[-300:])
    B.WARM_BIN = w
def run(cmd, cwd, extra=None, tmo=600, warm=True):
    return B.run(cmd, cwd, B.env(extra=extra), tmo, warm=warm)
def strip(x):
    s = x.pop('stdout') + '\n' + x['stderr']; x['clock'].pop('samples_mhz', None); return s     # clock summary kept, 50 ms samples dropped to keep the files small
def full_clock(runs): return bool(runs) and all(x['clock']['full_clock'] for x in runs)
def med(xs): xs = [x for x in xs if x is not None]; return statistics.median(xs) if xs else None
def need_data():
    if not A.data: raise SystemExit('--data is required')
    if not os.path.exists(A.data): raise SystemExit('no such data file')

# ---------------------------------------------------------------- parsers
def t_rtnn(txt, key): m = re.search(r'time ' + key + r': ([0-9.eE+-]+) ms', txt); return float(m.group(1)) if m else None
def tk_parse(txt):
    rr = [dict(round=int(m[0]), radius=float(m[1]), gt_span_ms=float(m[2]), f0_cycles=float(m[3]), f0_calls=float(m[4])) for m in re.findall(r'RND (\d+) radius ([0-9.]+) GT span_ms (\S+) CBK warps \d+ total_warp_cycles \S+ f0_cycles (\S+) f0_calls (\S+)', txt)]
    return dict(rounds=rr, gt_span_sum_ms=sum(r['gt_span_ms'] for r in rr), f0_cycles_sum=sum(r['f0_cycles'] for r in rr), rounds_n=len(rr))
def ark_parse(txt):
    m = re.search(r'KERNEL queries \d+ ms (\S+)', txt); c = re.search(r'f0_cycles (\S+) f0_calls (\S+)', txt)
    return dict(kernel_ms=float(m.group(1)) if m else None, f0_cycles=float(c.group(1)) if c else None, f0_calls=float(c.group(2)) if c else None)
def knn_parse(txt): m = re.search(r'Total Search Time: (\d+)ms', txt); return dict(total_search_ms=float(m.group(1)) if m else None)

# ---------------------------------------------------------------- calibrate
def cal_rtnn(pts, N, doc):
    out = {}; tree = cKDTree(pts)
    for K in KS:
        if K not in RT or K not in RTS: out[str(K)] = dict(missing='no real or stub build for K %d' % K); continue
        r = radius99(tree, pts, K); nq = max(32, int(round(0.02 * N))); idx = np.sort(np.random.default_rng(SEEDS['rtnn_sample']).choice(N, nq, replace=False))
        lines = [l for l in open(A.data).read().splitlines() if l.strip()]; qf = '%s_rtnn_q%d.csv' % (SCR, K); write(qf, '\n'.join(lines[i] for i in idx) + '\n')
        cube = cube_len(tree, pts[idx], r); Cm = float(cube.mean()); res = {}
        for kind, exe in (('real', RT[K]), ('stub', RTS[K])):
            runs = []
            for i in range(3):
                x = run([exe, '-f', A.data, '-q', qf, '-sm', 'knn', '-r', repr(r), '-k', str(K), '-p', '0', '-s', '0', '-m', '0', '-d', '0'], os.path.dirname(exe), {'RTNN_DRAM_GB': '11.5958'}, 300)
                txt = strip(x); x['search_ms'] = t_rtnn(txt, 'search compute'); runs.append(x)
                if x['rc'] != 0 or x['search_ms'] is None: break
            res[kind] = dict(runs=runs, median_ms=med([x['search_ms'] for x in runs if x['rc'] == 0]), full_clock=full_clock(runs))
        os.remove(qf); Tr, Ts = res['real']['median_ms'], res['stub']['median_ms']
        if Tr is None or Ts is None: out[str(K)] = dict(failed=True, radius=r, runs=res); continue
        sm = doc['sm']; raw = (Tr - Ts) * 1e-3 * sm * CLK / (nq * 1.10 * Cm); cbS = max(0.0, raw)
        out[str(K)] = dict(radius=six(r), nq=nq, mean_cube_sub=six(Cm), T_real_ms=Tr, T_stub_ms=Ts, cb_S_raw=six(raw), cb_S=six(cbS), floor_binds=bool(raw < 0), d_S=six(max(cbS - CBREF, -4.9)), runs=res)
        print('RTNN K %d: sample %d queries real %.4f stub %.4f ms -> cbS %.3f d_S %.3f' % (K, nq, Tr, Ts, cbS, out[str(K)]['d_S']), flush=True)
    return out
def cal_tk(pts, N, doc):
    tree = cKDTree(pts); r0 = r0_rule(tree, pts); k = A.tk_k; m = max(5000, int(round(0.02 * N))); idx = np.sort(np.random.default_rng(SEEDS['tk_sample']).choice(N, min(m, N), replace=False))
    lines = [l for l in open(A.data).read().splitlines() if l.strip()]; sf = '%s_tk_sample.csv' % SCR; write(sf, '\n'.join(lines[i] for i in idx) + '\n'); sub = pts[idx]
    rounds_sub = tk_rounds(sub, r0, k); runs = []
    for i in range(3):
        x = run([A.tk_bin, sf, str(len(idx)), '3', repr(r0), str(k), sf + '.out%d' % i], os.path.dirname(A.tk_bin), None, 600); txt = strip(x); x.update(tk_parse(txt)); runs.append(x)
        try: os.remove(sf + '.out%d' % i)
        except OSError: pass
    os.remove(sf); cpcs = []
    for x in runs:
        if x['rc'] != 0 or not x['rounds_n']: continue
        h = sum(r['hits'] for r in rounds_sub[:x['rounds_n']]); cpcs.append(x['f0_cycles_sum'] / h if h else None)
    cpc = med(cpcs)
    print('TrueKNN: sample %d points, r0 %.5g, k %d, runs rounds %s, cycles per replay box hit %s' % (len(idx), r0, k, [x['rounds_n'] for x in runs], cpc), flush=True)
    return dict(r0=six(r0), k=k, sample_points=int(len(idx)), replay_rounds_sample=[{kk: six(v) for kk, v in r.items()} for r in rounds_sub], cycles_per_replay_hit_runs=cpcs, cpc=six(cpc) if cpc else None, runs=runs, full_clock=full_clock(runs), failed=cpc is None)
def ark_inputs(pts, N):
    p2 = pts[:, :2]; tree = cKDTree(p2); r0 = r0_rule(tree, p2)
    Q = np.random.default_rng(SEEDS['ark_queries']).choice(N, min(10000, N), replace=False); q = p2[Q]; kd = tree.query(q, k=50)[0][:, -1]; r = r0
    while (kd <= r).mean() < 0.99: r *= 2
    return p2, tree, Q, r0, r
def ark_file(p2, q, path): write(path, '\n'.join('%.9g,%.9g' % (x, y) for x, y in np.vstack([p2, q])) + '\n')
def cal_ark(pts, N, doc):
    p2, tree, Q, r0, R = ark_inputs(pts, N); nq = len(Q); ns = min(1000, nq); ref = []
    if not os.path.exists(A.ark_ref): raise SystemExit('Arkade reference file missing: ' + A.ark_ref)
    for i in range(3):
        x = run([A.ark_probe, A.ark_ref, '100000', '100000', '0.005'], os.path.dirname(A.ark_probe), {'DEVOUT': '1'}, 300); txt = strip(x); x.update(ark_parse(txt)); ref.append(x)
    sf = '%s_ark_sample.txt' % SCR; ark_file(p2, p2[Q[:ns]], sf); runs = []
    for i in range(3):
        x = run([A.ark_probe, sf, str(N), str(ns), repr(R)], os.path.dirname(A.ark_probe), {'DEVOUT': '1'}, 300); txt = strip(x); x.update(ark_parse(txt)); runs.append(x)
    os.remove(sf)
    T_ref = med([x['kernel_ms'] for x in ref if x['rc'] == 0]); F_ref = med([x['f0_cycles'] for x in ref if x['rc'] == 0]); F_s = med([x['f0_cycles'] for x in runs if x['rc'] == 0])
    print('Arkade: radius %.5g, reference T_ref %s ms F_ref %s, sample F_s %s (x%.0f)' % (R, T_ref, F_ref, F_s, nq / ns), flush=True)
    return dict(r0=six(r0), radius=six(R), queries=nq, sample_queries=ns, T_ref_ms=T_ref, F_ref=F_ref, F_sample=F_s, F_d=F_s * nq / ns if F_s else None, reference_runs=ref, sample_runs=runs,
                full_clock=full_clock(ref + runs), failed=None in (T_ref, F_ref, F_s))
def do_calibrate():
    need_data(); bc.verify_frozen(); os.makedirs(OUT, exist_ok=True)
    if any(f.startswith('calibration_') for f in os.listdir(OUT)): raise SystemExit('%s already has a calibration: the sample procedures run once per dataset (RULE.md)' % OUT)
    warm_ready(); pts = load(A.data); N = len(pts); doc = common_doc('calibrate'); doc.update(N=N, ks=KS, tk_k=A.tk_k, mappings=MAPS)
    if 'rtnn' in MAPS: doc['rtnn'] = cal_rtnn(pts[:, :3], N, doc)
    if 'trueknn' in MAPS: doc['trueknn'] = cal_tk(pts[:, :3], N, doc)
    if 'arkade' in MAPS: doc['arkade'] = cal_ark(pts, N, doc)
    cj = os.path.join(OUT, 'calibration_%s.json' % TS); wjson(cj, doc); print('COMMIT (calibration):', commit([cj], 'sample calibration for %s (RULE.md procedures, no full run)' % A.name), flush=True)

# ---------------------------------------------------------------- predict (CPU)
def pred_rtnn(pts, N, cal, sm):
    m1.SM = sm; tree = cKDTree(pts); out = {}
    for K in KS:
        c = cal['rtnn'].get(str(K), {})
        if c.get('failed') or 'd_S' not in c: out[str(K)] = dict(missing='no calibration for K %d' % K); continue
        r = radius99(tree, pts, K); C, S = m1.replay_counts(pts, r, tree)
        p = m1.predict(N, K, C, S, c['d_S']); u = m1.predict(N, K, C, S, 0.0)
        out[str(K)] = dict(radius=six(r), d_S=c['d_S'], primary_ms=six(p['ms']), thr_ms=six(p['thr_ms']), chain_ms=six(p['chain_ms']), mean_q=six(p['mean_q']), secondary_uncalibrated_ms=six(u['ms']))
    return out
def pred_tk(pts, N, cal, sm):
    c = cal['trueknn']
    if c.get('failed'): return dict(missing='no calibration')
    rounds = tk_rounds(pts, c['r0'], c['k']); T = 0.0; rows = []
    for r in rounds:
        B_ = c['cpc'] * r['hits'] * 1.485 / (sm * 12.8) / CLK * 1e3; ch = r['maxc'] * LTK / CLK * 1e3; T += max(B_, ch); rows.append(dict(radius=six(r['radius']), active=r['active'], hits=six(r['hits']), callback_ms=six(B_), chain_ms=six(ch)))
    return dict(primary_ms=six(T), rounds=rows, cpc=c['cpc'], k=c['k'])
def pred_ark(pts, N, cal, sm):
    c = cal['arkade']
    if c.get('failed'): return dict(missing='no calibration')
    p2, tree, Q, r0, R = ark_inputs(pts, N)
    if abs(R - c['radius']) > 1e-5 * R: raise SystemExit('Arkade radius differs from the calibration file (%g vs %g)' % (R, c['radius']))
    cnt = 1.10 * cube_len(tree, p2[Q], R); nw = (len(cnt) + 31) // 32; pad = np.zeros(nw * 32); pad[:len(cnt)] = cnt
    Aa = c['T_ref_ms'] * c['F_d'] / c['F_ref']; Bb = pad.reshape(-1, 32).max(1).max() * LAR / CLK * 1e3 * FLOAT_FACTOR; P = max(Aa, Bb); MB = len(Q) * 50 * 8 / 1e6
    return dict(primary_ms=six(P), A_ms=six(Aa), B_ms=six(Bb), radius=six(R), mean_candidates=six(float(cnt.mean())), R1_pinned_over_device=six((P + PIN_MS_PER_MB * MB) / P), result_MB=six(MB))
def do_predict():
    need_data(); bc.verify_frozen(); cf = latest('calibration_')
    if not cf: raise SystemExit('no calibration file: calibrate first')
    why = pushed(cf); cal = json.load(open(cf)); doc = common_doc('predict'); sm = doc['sm']
    if cal['label'] != LABEL or cal['data_sha256_12'] != doc['data_sha256_12']: raise SystemExit('calibration is for another label or data file')
    if any(f.startswith('primary_predictions_') for f in os.listdir(OUT)): raise SystemExit('primary predictions exist already: made once')
    pts = load(A.data); N = len(pts); doc.update(N=N, calibration_file=os.path.relpath(cf, bc.REPO), calibration_commit=why, ks=KS, mappings=MAPS, version='PRIMARY v1-S (+ secondary uncalibrated v1 for RTNN, + relative rows)')
    if 'rtnn' in MAPS: doc['rtnn'] = pred_rtnn(pts[:, :3], N, cal, sm)
    if 'trueknn' in MAPS: doc['trueknn'] = pred_tk(pts[:, :3], N, cal, sm)
    if 'arkade' in MAPS: doc['arkade'] = pred_ark(pts, N, cal, sm)
    doc['relative'] = dict(R2_kNN_register_heap_speedup=REG_SPEEDUP, R1_pinned_over_device=doc.get('arkade', {}).get('R1_pinned_over_device'))
    pj = os.path.join(OUT, 'primary_predictions_%s.json' % TS); pm = os.path.join(OUT, 'primary_predictions_%s.md' % TS); wjson(pj, doc)
    L = ['# %s: PRIMARY (calibrated v1-S) predictions for %s, %s, before any full run\n' % (LABEL, A.name, TS)]
    if A.dry_run: L.append('**DRY RUN: the dataset was measured before. Tests the tools and procedures, not the model. Not a blind result.**\n')
    L.append('Data %s (%d points, sha256 prefix %s), SM %d, clock 2.5e9, calibration file %s (commit %s), rule hash %s.\n' % (doc['data'], N, doc['data_sha256_12'], sm, doc['calibration_file'], why, doc['rule_sha12']))
    for K, v in doc.get('rtnn', {}).items(): L.append('- RTNN K %s: primary %s ms (d_S %s, radius %s); secondary uncalibrated %s ms' % (K, v.get('primary_ms'), v.get('d_S'), v.get('radius'), v.get('secondary_uncalibrated_ms')) if 'primary_ms' in v else '- RTNN K %s: missing (%s)' % (K, v.get('missing')))
    if 'trueknn' in doc: L.append('- TrueKNN k %s: primary %s ms over %d rounds' % (doc['trueknn'].get('k'), doc['trueknn'].get('primary_ms'), len(doc['trueknn'].get('rounds', []))))
    if 'arkade' in doc: L.append('- Arkade: primary %s ms (A %s, B %s), radius %s' % (doc['arkade'].get('primary_ms'), doc['arkade'].get('A_ms'), doc['arkade'].get('B_ms'), doc['arkade'].get('radius')))
    L.append('- Relative R1 (Arkade pinned over device kernel time): %s; R2 (kNN register heap over original): %s' % (doc['relative']['R1_pinned_over_device'], REG_SPEEDUP))
    write(pm, '\n'.join(L) + '\n'); print('\n'.join(L)); print('COMMIT (predictions):', commit([pj, pm], 'primary predictions for %s (calibrated v1-S, before any full run)' % A.name), flush=True)

# ---------------------------------------------------------------- measure (full runs)
def meas_rtnn(pred):
    out = {}
    for K in KS:
        p = pred['rtnn'].get(str(K), {})
        if 'primary_ms' not in p or K not in RT: out[str(K)] = dict(missing='no prediction or build'); continue
        runs = []
        for i in range(3):
            x = run([RT[K], '-f', A.data, '-sm', 'knn', '-r', repr(p['radius']), '-k', str(K), '-p', '0', '-s', '0', '-m', '0', '-d', '0'], os.path.dirname(RT[K]), {'RTNN_DRAM_GB': '11.5958'}, 900)
            txt = strip(x); x['search_ms'] = t_rtnn(txt, 'search compute'); runs.append(x)
            if x['rc'] != 0 or x['search_ms'] is None: break
        out[str(K)] = dict(runs=runs, median_ms=med([x['search_ms'] for x in runs if x['rc'] == 0]), full_clock=full_clock(runs))
    return out
def meas_tk(pred, N):
    k = pred['trueknn']['k']; r0 = json.load(open(latest('calibration_')))['trueknn']['r0']; runs = []
    for i in range(3):
        o = '%s_tkfull%d.out' % (SCR, i); x = run([A.tk_bin, A.data, str(N), '3', repr(r0), str(k), o], os.path.dirname(A.tk_bin), None, 3600); txt = strip(x); x.update(tk_parse(txt)); runs.append(x)
        try: os.remove(o)
        except OSError: pass
    return dict(runs=runs, median_ms=med([x['gt_span_sum_ms'] for x in runs if x['rc'] == 0 and x['rounds_n']]), full_clock=full_clock(runs))
def meas_ark(pts, N, pred):
    p2, tree, Q, r0, R = ark_inputs(pts, N); f = '%s_ark_full.txt' % SCR; ark_file(p2, p2[Q], f); out = {}
    for name, ex in (('device', {'DEVOUT': '1', 'CB_OFF': '1'}), ('pinned', {'CB_OFF': '1'})):
        runs = []
        for i in range(3):
            x = run([A.ark_probe, f, str(N), str(len(Q)), repr(R)], os.path.dirname(A.ark_probe), ex, 900); txt = strip(x); x.update(ark_parse(txt)); runs.append(x)
        out[name] = dict(runs=runs, median_ms=med([x['kernel_ms'] for x in runs if x['rc'] == 0]), full_clock=full_clock(runs))
    os.remove(f); return out
def ncu_launch_ns(txt):
    """Sum of gpu__time_duration over all optixLaunch kernels in an ncu --csv output (ns)."""
    import csv, io
    tot = 0.0; n = 0
    for rec in csv.reader(io.StringIO(txt)):
        if len(rec) > 5 and rec[-1].replace(',', '').replace('.', '').isdigit() and 'gpu__time_duration' in ''.join(rec): tot += float(rec[-1].replace(',', '')); n += 1
    return dict(launch_sum_ms=tot / 1e6 if n else None, launches=n)
def meas_knn(pts, N, pred):
    cal = json.load(open(latest('calibration_'))); r0 = cal['trueknn']['r0'] if 'trueknn' in cal else r0_rule(cKDTree(pts[:, :3]), pts[:, :3]); out = {}
    for name, d in (('original', A.knn_orig), ('register_heap', A.knn_reg)):
        runs = []
        for i in range(3):
            x = run(['ncu', '--clock-control', 'none', '--cache-control', 'none', '--kernel-name', 'regex:optixLaunch', '--metrics', 'gpu__time_duration.sum', '--csv', '--print-units', 'base', d + '/knn_host', A.data, A.data, '16', repr(r0)],
                    d, {'TMPDIR': HOME + '/ncutmp'}, 1800); txt = strip(x); x.update(ncu_launch_ns(txt)); runs.append(x)
        out[name] = dict(runs=runs, median_ms=med([x['launch_sum_ms'] for x in runs if x['rc'] == 0]), full_clock=full_clock(runs))
    return out
def do_measure():
    need_data(); bc.verify_frozen(); cf = latest('calibration_'); pf = latest('primary_predictions_')
    if not cf or not pf: raise SystemExit('calibration and primary predictions are needed first')
    c1 = pushed(cf); c2 = pushed(pf)
    if bc.sh(['git', '-C', bc.REPO, 'merge-base', '--is-ancestor', c1, c2]).returncode != 0: raise SystemExit('calibration commit is not an ancestor of the prediction commit')
    pred = json.load(open(pf)); doc = common_doc('measure')
    if pred['label'] != LABEL or pred['data_sha256_12'] != doc['data_sha256_12']: raise SystemExit('prediction file is for another label or data file')
    warm_ready(); pts = load(A.data); N = len(pts); doc.update(N=N, predictions_file=os.path.relpath(pf, bc.REPO), predictions_commit=c2, calibration_commit=c1, mappings=MAPS)
    if 'rtnn' in MAPS: doc['rtnn'] = meas_rtnn(pred)
    if 'trueknn' in MAPS: doc['trueknn'] = meas_tk(pred, N)
    if 'arkade' in MAPS: doc['arkade'] = meas_ark(pts, N, pred)
    doc['knn'] = meas_knn(pts, N, pred)
    mj = os.path.join(OUT, 'primary_measured_%s.json' % TS); wjson(mj, doc); print('COMMIT (measured):', commit([mj], 'full measurements for %s (after the committed calibration and predictions)' % A.name), flush=True)

# ---------------------------------------------------------------- score (CPU)
def row(version, label, pred, meas):
    if pred is None or meas is None or pred <= 0: return dict(version=version, row=label, predicted=pred, measured=meas, missing=True)
    r = meas / pred; return dict(version=version, row=label, predicted=six(pred), measured=six(meas), ratio=six(r), ln_ratio=six(math.log(r)), inside_2=0.5 <= r <= 2.0)
def summary(rows):
    have = [r for r in rows if not r.get('missing')]; ln = [r['ln_ratio'] for r in have]
    return dict(rows=len(rows), measured_rows=len(have), missing=len(rows) - len(have), inside_factor_2=sum(r['inside_2'] for r in have), rms_ln_ratio=six(math.sqrt(sum(x * x for x in ln) / len(ln))) if ln else None,
                median_abs_ln_ratio=six(statistics.median(abs(x) for x in ln)) if ln else None, worst_ratio=six(max((max(r['ratio'], 1 / r['ratio']) for r in have), default=0.0)) if have else None)
def do_score():
    pf, mf = latest('primary_predictions_'), latest('primary_measured_')
    if not pf or not mf: raise SystemExit('need primary predictions and measurements')
    P, M = json.load(open(pf)), json.load(open(mf))
    if P['label'] != M['label'] or M['predictions_file'] != os.path.relpath(pf, bc.REPO): raise SystemExit('measured file was taken against a different predictions file')
    rows = []
    for K, v in P.get('rtnn', {}).items():
        m = M.get('rtnn', {}).get(K, {}).get('median_ms')
        rows.append(row('PRIMARY', 'RTNN K%s' % K, v.get('primary_ms'), m)); rows.append(row('SECONDARY (uncalibrated v1)', 'RTNN K%s' % K, v.get('secondary_uncalibrated_ms'), m))
    if 'trueknn' in P: rows.append(row('PRIMARY', 'TrueKNN k%s' % P['trueknn'].get('k'), P['trueknn'].get('primary_ms'), M.get('trueknn', {}).get('median_ms')))
    if 'arkade' in P: rows.append(row('PRIMARY', 'Arkade', P['arkade'].get('primary_ms'), M.get('arkade', {}).get('device', {}).get('median_ms')))
    rel = []
    if 'arkade' in P and P['arkade'].get('R1_pinned_over_device') and M.get('arkade'):
        d, p = M['arkade']['device']['median_ms'], M['arkade']['pinned']['median_ms']; rel.append(row('RELATIVE', 'R1 Arkade pinned over device', P['arkade']['R1_pinned_over_device'], p / d if d and p else None))
    o, g = M['knn']['original']['median_ms'], M['knn']['register_heap']['median_ms']; rel.append(row('RELATIVE', 'R2 kNN register heap speedup', REG_SPEEDUP, o / g if o and g else None))
    prim = [r for r in rows if r['version'] == 'PRIMARY']; sec = [r for r in rows if r['version'].startswith('SECONDARY')]
    ps = open(os.path.join(bc.REPO, 'model', 'blind', 'PASS.md')).read(); passset = 'STATUS: SET' in ps
    doc = dict(label=P['label'], name=A.name, predictions=os.path.basename(pf), measured=os.path.basename(mf), rows=rows + rel, summary=dict(primary=summary(prim), secondary=summary(sec), relative=summary(rel)), pass_md_status='SET' if passset else 'UNSET: no verdict printed',
               all_runs_full_clock=all(v for v in _clock_flags(M)))
    sj = os.path.join(OUT, 'primary_scores_%s.json' % TS); sm_ = os.path.join(OUT, 'primary_scores_%s.md' % TS); wjson(sj, doc)
    L = ['# %s: scores for %s (primary, secondary, relative)\n' % (P['label'], A.name)]
    if P['label'] == 'DRY RUN': L.append('**DRY RUN: tests the tools and procedures only; not a blind result.**\n')
    L.append('| version | row | predicted | measured | ratio | inside 2x |\n|---|---|---|---|---|---|')
    for r in rows + rel: L.append('| %s | %s | %s | %s | %s | %s |' % (r['version'], r['row'], r.get('predicted'), r.get('measured'), r.get('ratio', '-'), 'missing' if r.get('missing') else ('yes' if r['inside_2'] else 'no')))
    for k, s in doc['summary'].items(): L.append('\n%s: rows %d, measured %d, missing %d, inside 2x %d, rms ln ratio %s, median |ln| %s, worst ratio %s' % (k, s['rows'], s['measured_rows'], s['missing'], s['inside_factor_2'], s['rms_ln_ratio'], s['median_abs_ln_ratio'], s['worst_ratio']))
    L.append('\nPASS.md: %s. All timed runs at full clock: %s.' % (doc['pass_md_status'], doc['all_runs_full_clock']))
    write(sm_, '\n'.join(L) + '\n'); print('\n'.join(L)); print('COMMIT (scores):', commit([sj, sm_], 'primary scores for %s' % A.name), flush=True)
def _clock_flags(M):
    f = []
    def walk(x):
        if isinstance(x, dict):
            if 'full_clock' in x and isinstance(x['full_clock'], bool): f.append(x['full_clock'])
            for v in x.values(): walk(v)
        elif isinstance(x, list):
            for v in x: walk(v)
    walk(M); return f or [False]

{'calibrate': do_calibrate, 'predict': do_predict, 'measure': do_measure, 'score': do_score}[A.step]()
