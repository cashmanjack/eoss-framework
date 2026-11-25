"""DIAGNOSTIC on an already measured dataset (dry run only): how well does the RTNN real minus stub difference resolve the callback cost at different query sample sizes?
    diag_sample_size.py DATA NAME --fracs 0.02,0.1,0.25,1.0 [--k 50]
Writes model/blind/dry_run_NAME/diag_sample_size_<ts>.json (never overwritten). Not part of the blind procedure; it does not change RULE.md."""
import os, sys, json, time, argparse, statistics, numpy as np
from scipy.spatial import cKDTree
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import blind_common as bc
ap = argparse.ArgumentParser(); ap.add_argument('data'); ap.add_argument('name'); ap.add_argument('--fracs', default='0.02,0.1,0.25,0.5,1.0'); ap.add_argument('--k', type=int, default=50); ap.add_argument('--mode', default='knn', choices=['knn', 'radius']); ap.add_argument('--gpu', default='1'); a = ap.parse_args()
os.environ['BASE_GPU'] = a.gpu; sys.path.insert(0, os.path.join(bc.REPO, 'ttu', 'bench', 'Base_1')); import base_common as B
import re
HOME = os.path.expanduser('~'); TP = HOME + '/thirdparty'; B.WARM_BIN = '/dev/shm/blind_warm'
if not os.path.exists(B.WARM_BIN): bc.sh([HOME + '/opt/cuda-12.8/bin/nvcc', '-O2', '-arch=sm_89', '-o', B.WARM_BIN, os.path.join(bc.REPO, 'ttu', 'bench', 'Base_1', 'warm.cu')])
pts = np.array([l.replace(',', ' ').split() for l in open(a.data).read().splitlines() if l.strip()], dtype=np.float64); N = len(pts); tree = cKDTree(pts)
q = pts[np.random.default_rng(5).choice(N, 3000, replace=False)]; r = float(np.quantile(tree.query(q, k=a.k + 1, workers=-1)[0][:, a.k], 0.99)); sm = bc.sm_for(a.gpu); lines = [l for l in open(a.data).read().splitlines() if l.strip()]
real = TP + '/div1/rtnn/src/build/bin/optixNSearch'; stub = TP + '/div1_stub/rtnn/src/build/bin/optixNSearch'; out = dict(label='DIAGNOSTIC (dry run dataset)', mode=a.mode, data=os.path.basename(a.data), N=N, K=a.k, radius=r, sm=sm, rows={})
for fr in [float(x) for x in a.fracs.split(',')]:
    nq = N if fr >= 1 else max(32, int(round(fr * N))); idx = np.sort(np.random.default_rng(11).choice(N, nq, replace=False)); qf = '/dev/shm/diag_q_%d.csv' % os.getpid()
    open(qf, 'w').write('\n'.join(lines[i] for i in idx) + '\n'); Cm = float(tree.query_ball_point(pts[idx], r, p=np.inf, return_length=True, workers=-1).mean()); res = {}
    for kind, exe in (('real', real), ('stub', stub)):
        ts = []
        for i in range(5):
            x = B.run([exe, '-f', a.data, '-q', qf] + (['-sm', 'knn'] if a.mode == 'knn' else []) + ['-r', repr(r), '-k', str(a.k), '-p', '0', '-s', '0', '-m', '0', '-d', '0'], os.path.dirname(exe), B.env(extra={'RTNN_DRAM_GB': '11.5958'}), 300)
            m = re.search(r'time search compute: ([0-9.eE+-]+) ms', x['stdout'] + x['stderr']); ts.append(float(m.group(1)) if m else None)
        res[kind] = ts
    os.remove(qf); Tr, Ts = statistics.median(res['real']), statistics.median(res['stub']); raw = (Tr - Ts) * 1e-3 * sm * bc.CLK / (nq * 1.10 * Cm)
    out['rows'][str(fr)] = dict(nq=nq, real_ms=res['real'], stub_ms=res['stub'], real_median=Tr, stub_median=Ts, diff_over_real=(Tr - Ts) / Tr, cb_S_raw=raw, spread_real=(max(res['real']) - min(res['real'])) / Tr)
    print('frac %g nq %d: real %.4f stub %.4f ms, difference %.1f%% of real, raw cb %.2f cycles per candidate, real spread %.1f%%' % (fr, nq, Tr, Ts, 100 * (Tr - Ts) / Tr, raw, 100 * out['rows'][str(fr)]['spread_real']), flush=True)
p = os.path.join(bc.REPO, 'model', 'blind', 'dry_run_' + a.name, 'diag_sample_size_%s_%s.json' % (a.mode, time.strftime('%Y%m%dT%H%M%S')))
with open(p, 'x') as f: json.dump(out, f, indent=1)
print(p)
