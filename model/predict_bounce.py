"""Scene only prediction of the grouping speedups of bounce rays on another GPU (CPU replay over the decoded BVH; nothing fitted on the target GPU).
    predict_bounce.py WORK SCENE_N LOG2_PRIMARY MEASURED_JSON NSM L2_MB     (needs /dev/shm/rp/{assign,reuse}, MORTON_DIR host for the GAS dump and bounce rays, GPU only for those)
Everything about the performance model comes from the RTX 6000 Ada (parameters fitted on Memory_4, k = sectors per line request set on triton's random 1e5 ray set, 60 KB L1 = 480 lines);
the target GPU enters only through its SM count (NSM) and its L2 size. Inputs per launch order from the CPU replay: queries per ray q (replay.cpp, variant A), lines requested a and misses m per ray
(reuse.cpp histograms, one SM stream, nodes only, dedupe) -> S1 = k a, F = k m, D = 128 B x overflow x max(0, 1 - L2 / BVH). Predicted speedup of an order = c(pixel order) / c(order), c = 1 / predict_sm.
"""
import json, os, subprocess, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, '..')
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, 'ttu', 'bench', 'Morton')); sys.path.insert(0, os.path.join(ROOT, 'ttu', 'bench', 'Memory_4'))
import performance_model as PM, run_model as RM, reuse_inputs as RI  # noqa: E402
import morton_run as mr, netsort_run as ns  # noqa: E402
M = mr.M; RP = '/dev/shm/rp'


def hist(path):
    for l in open(path):
        if l.startswith('sm nodes dedupe'):
            p = l.split(); return int(p[3]), int(p[4]), int(p[5]), np.array(p[6:], np.int64)


def main(work, scene_n, lg, measured, nsm, l2_mb, bvh_mb):
    os.makedirs(work, exist_ok=True); scene_n = int(scene_n); nsm = int(nsm); l2_mb = float(l2_mb); bvh_mb = float(bvh_mb)
    # constants from triton only
    p = PM.fit(RM.memory4_rows())
    a_r = hist('notes/q5/reuse/reuse_1e5_random.txt')[0] / (142 * 2 * 39 * 32); kk = 82.0 / a_r
    R = ns.make(scene_n, 1 << int(lg), 'bounce'); N = len(R); base = M + '/p_rays.bin'; R.tofile(base); ref = M + '/p_ref'
    mr.host(scene_n, base, ref, ['plain'], ['--dump-gas', ref + '.gas', '--dump-tris', ref + '.tris'])
    canon = np.fromfile(ref + '.0.bin', np.uint32).reshape(-1, 8); tris = np.fromfile(ref + '.tris', np.float32).reshape(-1, 3, 3)
    Wn, kids, root = mr.prep.tree(open(ref + '.gas', 'rb').read()); codes = mr.oracle_codes(Wn, kids, root, R, canon, tris)
    subprocess.run(['/usr/bin/python3', os.path.join(HERE, 'replay', 'make_table.py'), ref + '.gas', ref + '.tris', M + '/p_table.bin'], env=dict(os.environ, ASSIGN=RP + '/assign'), check=True)
    rng = np.random.default_rng(7)
    orders = {'natural': np.arange(N), 'shuffled': rng.permutation(N), 'morton-o k5': mr.order_by(mr.code_o(R, 5)),
              'oracle d1': mr.order_by(codes['d1'][0]), 'oracle d2': mr.order_by(codes['d2'][0]), 'oracle d3': mr.order_by(codes['d3'][0]), 'oracle full': mr.order_by(codes['full'][0])}
    # queries per ray from the replay on a sample of 200,000 rays (variant A: any pending entry resumes)
    samp = R[:200000]; samp.tofile(M + '/p_s.bin'); qf = M + '/p_q.bin'
    subprocess.run([RP + '/replay', M + '/p_table.bin', M + '/p_s.bin', '200000', '0', qf], check=True)
    q = float(np.fromfile(qf, np.int32).reshape(-1, 8)[:, 3].mean()); print('replay queries per ray (variant A)', round(q, 3), flush=True)
    rays_stream = nsm * 2 * 39 * 32; feats = {}
    for name, o in orders.items():
        o.astype(np.int32).tofile(M + '/p_order.bin'); out = os.path.join(work, 'reuse_%d_%s.txt' % (scene_n, name.replace(' ', '_')))
        subprocess.run(['nice', '-n', '10', RP + '/reuse', M + '/p_table.bin', base, str(N), M + '/p_order.bin', out], env=dict(os.environ, NSM=str(nsm)), check=True)
        tot, cold, over, h = hist(out); mis = (cold + over + h[480:].sum()) / rays_stream
        share = max(0.0, 1 - l2_mb / bvh_mb)
        feats[name] = dict(q=q, S1=kk * tot / rays_stream, F=kk * mis, D=128 * (cold + over) / rays_stream * share, W=13); print(name, {k: round(v, 2) for k, v in feats[name].items()}, flush=True)
    c = {k: 1 / PM.predict_sm(p, v) for k, v in feats.items()}
    meas = {r['order']: r['ms'] for r in json.load(open(measured))['results']}
    nat = meas['natural']; rows = []
    for name in orders:
        if name == 'natural': continue
        key = name; mk = key if key in meas else None
        if mk is None: continue
        rows.append((name, c['natural'] / c[name], nat / meas[mk]))
        print('%-12s predicted speedup over pixel order %.2f   measured %.2f' % rows[-1], flush=True)
    e = np.sqrt(np.mean([np.log(a / b) ** 2 for _, a, b in rows])); print('rms log error %.3f over %d orders' % (e, len(rows)))
    json.dump(dict(scene=scene_n, nsm=nsm, l2_mb=l2_mb, q=q, feats=feats, rows=rows, rms_log_error=e), open(os.path.join(work, 'predict_bounce_%d.json' % scene_n), 'w'), indent=1)
    for f in os.listdir(M):
        if f.startswith('p_'): os.remove(os.path.join(M, f))


if __name__ == '__main__': main(*sys.argv[1:8])
