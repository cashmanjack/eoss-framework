"""Pre registered scene only check of the performance model on RANDOM rays (OptiX only, no TTU code). 2026-11-15, io.

    random_check.py predict WORK SCENE LOG2_RAYS TAG    scene: a soup size (triangles) or file:PATH.tris (scene2tris output)
    random_check.py measure WORK SCENE TAG              launch orders from the saved predict step, 10 launches each, median, every result compared with the natural order

predict writes WORK/predict_TAG.json and a text table; it needs the GPU only to build the GAS and dump it (one plain trace of the natural order whose time is NOT read) and the CPU replay tools in /dev/shm/rp.
The performance model parameters are the ones fitted on the RTX 6000 Ada (Memory_4 one SM data); k (sectors per line request) was set on triton's random 1e5 ray set; the target GPU enters through
its SM count (60) and L2 size (48 MB) only. Same code path as model/predict_bounce.py (Finding 113); this file only changes the ray kind (random: origin uniform in the scene box, direction uniform
on the sphere, tmax 1e16) and the order list (natural = random order; Morton of origin k5; oracle depth 1 to 4 and full).
The predict step is run and committed BEFORE the measure step is run.
"""
import json, os, subprocess, sys, statistics as st
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, '..', '..', '..')
sys.path.insert(0, os.path.join(ROOT, 'model')); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, 'ttu', 'bench', 'Memory_4'))
import performance_model as PM, run_model as RM  # noqa: E402
import morton_run as mr  # noqa: E402
M = mr.M; RP = '/dev/shm/rp'; CK = '/dev/shm/rt/ck'
NSM, L2_MB = 60, 48.0


def scene_box(scene):
    if scene.startswith('file:'):
        t = np.fromfile(scene[5:], np.float32).reshape(-1, 3); return t.min(0).astype(np.float64), t.max(0).astype(np.float64)
    return np.full(3, -1.0), np.full(3, 1.0)


def make_rays(scene, n):
    if not scene.startswith('file:'): return mr.prep.make_rays(n)
    lo, hi = scene_box(scene); g = np.random.default_rng(2026)
    o = g.uniform(lo, hi, (n, 3)); d = g.normal(size=(n, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
    r = np.zeros((n, 8), np.float32); r[:, 0:3] = o; r[:, 4:7] = d; r[:, 7] = 1e16
    return r


def unit_origin(R, scene):    # origins mapped into [-1,1]^3 by one scale (cubic cells) for the Morton code
    lo, hi = scene_box(scene); c = (lo + hi) / 2; s = float((hi - lo).max()) / 2
    Rn = R.copy(); Rn[:, 0:3] = ((R[:, 0:3].astype(np.float64) - c) / s).astype(np.float32); return Rn


def hist(path):
    for l in open(path):
        if l.startswith('sm nodes dedupe'):
            p = l.split(); return int(p[3]), int(p[4]), int(p[5]), np.array(p[6:], np.int64)


def predict(work, scene, lg, tag):
    os.makedirs(work, exist_ok=True); os.makedirs(CK, exist_ok=True); n = 1 << int(lg)
    p = PM.fit(RM.memory4_rows())
    a_r = hist(os.path.join(ROOT, 'notes/q5/reuse/reuse_1e5_random.txt'))[0] / (142 * 2 * 39 * 32); kk = 82.0 / a_r
    R = make_rays(scene, n); N = len(R); base = M + '/c_rays.bin'; R.tofile(base); ref = M + '/c_ref'
    mr.host(scene, base, ref, ['plain'], ['--dump-gas', ref + '.gas', '--dump-tris', ref + '.tris'])   # time of this launch is not read
    canon = np.fromfile(ref + '.0.bin', np.uint32).reshape(-1, 8); tris = np.fromfile(ref + '.tris', np.float32).reshape(-1, 3, 3)
    bvh_mb = os.path.getsize(ref + '.gas') / 2 ** 20
    print('%s: %d rays, %.1f%% hit, %d triangles, GAS %.1f MiB' % (tag, N, 100 * (canon[:, 0] != 0).mean(), len(tris), bvh_mb), flush=True)
    Wn, kids, root = mr.prep.tree(open(ref + '.gas', 'rb').read()); codes = mr.oracle_codes(Wn, kids, root, R, canon, tris)
    subprocess.run(['/usr/bin/python3', os.path.join(ROOT, 'model', 'replay', 'make_table.py'), ref + '.gas', ref + '.tris', M + '/c_table.bin'], env=dict(os.environ, ASSIGN=RP + '/assign'), check=True)
    Rn = unit_origin(R, scene)
    orders = {'natural': np.arange(N), 'morton-o k5': mr.order_by(mr.code_o(Rn, 5))}
    for d in ('d1', 'd2', 'd3', 'd4', 'full'): orders['oracle ' + d] = mr.order_by(codes[d][0])
    groups = {d: codes[d][1] for d in ('d1', 'd2', 'd3', 'd4', 'full')}
    samp = R[:200000]; samp.tofile(M + '/c_s.bin'); qf = M + '/c_q.bin'
    subprocess.run([RP + '/replay', M + '/c_table.bin', M + '/c_s.bin', '200000', '0', qf], check=True)
    q = float(np.fromfile(qf, np.int32).reshape(-1, 8)[:, 3].mean()); print('replay queries per ray (variant A)', round(q, 3), flush=True)
    rays_stream = NSM * 2 * 39 * 32; feats = {}
    for name, o in orders.items():
        o.astype(np.int32).tofile(M + '/c_order.bin'); out = os.path.join(work, 'reuse_%s_%s.txt' % (tag, name.replace(' ', '_')))
        subprocess.run(['nice', '-n', '10', RP + '/reuse', M + '/c_table.bin', base, str(N), M + '/c_order.bin', out], env=dict(os.environ, NSM=str(NSM)), check=True)
        tot, cold, over, h = hist(out); mis = (cold + over + h[480:].sum()) / rays_stream
        share = max(0.0, 1 - L2_MB / bvh_mb)
        feats[name] = dict(q=q, S1=kk * tot / rays_stream, F=kk * mis, D=128 * (cold + over) / rays_stream * share, W=13); print(name, {k: round(v, 2) for k, v in feats[name].items()}, flush=True)
    c = {k: 1 / PM.predict_sm(p, v) for k, v in feats.items()}
    rows = [(name, c['natural'] / c[name]) for name in orders if name != 'natural']
    for name, s in rows: print('%-12s predicted speedup over the natural (random) order: %.2f' % (name, s), flush=True)
    np.savez(os.path.join(CK, 'ck_%s.npz' % tag), R=R, canon=canon, **{k.replace(' ', '_'): v.astype(np.int32) for k, v in orders.items()})
    json.dump(dict(scene=scene, tag=tag, rays=N, q=q, bvh_mib=bvh_mb, l2_mib=L2_MB, nsm=NSM, hit_fraction=float((canon[:, 0] != 0).mean()), groups=groups, feats=feats, predicted=dict(rows)), open(os.path.join(work, 'predict_%s.json' % tag), 'w'), indent=1)
    for f in os.listdir(M):
        if f.startswith('c_'): os.remove(os.path.join(M, f))
    if os.path.exists(M + '/c_ref.0.bin'): os.remove(M + '/c_ref.0.bin')


def measure(work, scene, tag):
    z = np.load(os.path.join(CK, 'ck_%s.npz' % tag)); R = z['R']; canon = z['canon']; N = len(R); pr = json.load(open(os.path.join(work, 'predict_%s.json' % tag)))
    names = ['natural', 'morton-o k5', 'oracle d1', 'oracle d2', 'oracle d3', 'oracle d4', 'oracle full']; out = []
    for name in names:
        perm = z[name.replace(' ', '_')].astype(np.int64); f = M + '/c_rays.bin'; R[perm].tofile(f)
        r = mr.host(scene, f, M + '/c_out', ['plain'])[0]
        o = np.fromfile(M + '/c_out.0.bin', np.uint32).reshape(-1, 8); os.remove(M + '/c_out.0.bin')
        bad = int((o != canon[perm]).any(1).sum()); out.append(dict(order=name, ms=st.median(r['ms']), wrong=bad)); print(name, out[-1], flush=True)
    nat = out[0]['ms']; rows = []
    for r in out[1:]:
        s = nat / r['ms']; ps = pr['predicted'][r['order']]; rows.append((r['order'], ps, s)); print('%-12s predicted %.2f  measured %.2f  log error %+.3f  (wrong %d)' % (r['order'], ps, s, np.log(ps / s), r['wrong']), flush=True)
    e = float(np.sqrt(np.mean([np.log(a / b) ** 2 for _, a, b in rows]))); span = float(np.sqrt(np.mean([np.log(b) ** 2 for _, _, b in rows])))
    print('rms log error %.3f over %d orders; rms of the measured log speedups (the spread to explain) %.3f' % (e, len(rows), span))
    json.dump(dict(scene=scene, tag=tag, results=out, rows=rows, rms_log_error=e, rms_log_measured=span), open(os.path.join(work, 'measure_%s.json' % tag), 'w'), indent=1)
    for f in os.listdir(M):
        if f.startswith('c_'): os.remove(os.path.join(M, f))


if __name__ == '__main__':
    a = sys.argv[1:]
    predict(a[1], a[2], a[3], a[4]) if a[0] == 'predict' else measure(a[1], a[2], a[3])
