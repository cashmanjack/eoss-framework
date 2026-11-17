"""Pre registered check of the performance model on CAMERA and BOUNCE rays of graphics scenes (OptiX only, no TTU code). 2026-11-17, tethys.

    scene_check.py predict WORK SCENE KIND TAG      SCENE: file:PATH.tris (scene2tris output); KIND: camera | bounce
    scene_check.py measure WORK SCENE TAG           launch the orders saved by predict, 10 launches each, median, every result compared with the natural (pixel) order

Rays: camera rays 4096 x 2048 in pixel order from the auto picked camera of ttu/bench/RealScene/scene_run.py (same functions, y up), or the cosine weighted bounce rays from the closest hits of those
camera rays (survivors in pixel order, seed 5, as scene_run.py depth 2). Natural = pixel order. Orders: natural, shuffled, Morton (origin 5 bits for bounce rays; direction 5 bits for camera rays,
because every camera ray has the same origin), oracle depth 1, 2, 3, 4, full. The predict step uses the GPU only for the camera probe, a GAS dump and the closest hits (no time read), then the
CPU replay tools in /dev/shm/rp, as ttu/bench/Morton/random_check.py. Raw measure files are timestamped and never overwritten.
"""
import json, os, subprocess, sys, statistics as st, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, '..', '..', '..')
sys.path.insert(0, os.path.join(ROOT, 'model')); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, 'ttu', 'bench', 'Memory_4')); sys.path.insert(0, os.path.join(ROOT, 'ttu', 'bench', 'RealScene'))
import performance_model as PM, run_model as RM  # noqa: E402
import morton_run as mr  # noqa: E402
import random_check as rc  # noqa: E402
import scene_run as sr  # noqa: E402
M = mr.M; RP = '/dev/shm/rp'; CK = '/dev/shm/hs2/ck'; TD = '/dev/shm/hs2/t'; TRIS = '/dev/shm/hs2/tris'
sr.SC = TRIS; sr.T = TD; sr.M2 = M
NSM, L2_MB = int(os.environ.get('RC_NSM', 128)), float(os.environ.get('RC_L2_MB', 72.0))
NAMES = ['natural', 'shuffled', 'morton', 'oracle d1', 'oracle d2', 'oracle d3', 'oracle d4', 'oracle full']


def predict(work, scene, kind, tag):
    os.makedirs(work, exist_ok=True); os.makedirs(CK, exist_ok=True); os.makedirs(TD, exist_ok=True)
    path = scene[5:]; sname = os.path.basename(path)[:-5]; tris = np.fromfile(path, np.float32).reshape(-1, 3, 3)
    lo = tris.reshape(-1, 3).min(0).astype(np.float64); hi = tris.reshape(-1, 3).max(0).astype(np.float64); diag = float(np.linalg.norm(hi - lo))
    pos, fwd, camdesc = sr.pick_camera(sname, lo, hi, 1); upv = np.array([0, 1.0, 0])
    R = sr.camera_rays(pos, fwd, upv, 4096, 2048, 60.0)
    if kind == 'bounce':
        canon0 = sr.trace(sname, R, 'c0'); R, _ = sr.bounce(R, canon0, tris, np.random.default_rng(5), diag)
    N = len(R); base = M + '/c_rays.bin'; R.tofile(base); ref = M + '/c_ref'
    mr.host(scene, base, ref, ['plain'], ['--dump-gas', ref + '.gas', '--dump-tris', ref + '.tris'])   # time not read
    canon = np.fromfile(ref + '.0.bin', np.uint32).reshape(-1, 8); trisd = np.fromfile(ref + '.tris', np.float32).reshape(-1, 3, 3)
    bvh_mb = os.path.getsize(ref + '.gas') / 2 ** 20
    print('%s %s: %d rays, %.1f%% hit, %d triangles, GAS %.1f MiB, camera %s' % (tag, kind, N, 100 * (canon[:, 0] != 0).mean(), len(trisd), bvh_mb, camdesc), flush=True)
    p = PM.fit(RM.memory4_rows())
    a_r = rc.hist(os.path.join(ROOT, 'notes/q5/reuse/reuse_1e5_random.txt'))[0] / (142 * 2 * 39 * 32); kk = 82.0 / a_r
    Wn, kids, root = mr.prep.tree(open(ref + '.gas', 'rb').read()); codes = mr.oracle_codes(Wn, kids, root, R, canon, trisd)
    subprocess.run(['/usr/bin/python3', os.path.join(ROOT, 'model', 'replay', 'make_table.py'), ref + '.gas', ref + '.tris', M + '/c_table.bin'], env=dict(os.environ, ASSIGN=RP + '/assign'), check=True)
    morton = mr.order_by(mr.code_d(R, 5)) if kind == 'camera' else mr.order_by(mr.code_o(rc.unit_origin(R, scene), 5))
    orders = {'natural': np.arange(N), 'shuffled': np.random.default_rng(7).permutation(N), 'morton': morton}
    for d in ('d1', 'd2', 'd3', 'd4', 'full'): orders['oracle ' + d] = mr.order_by(codes[d][0])
    groups = {d: codes[d][1] for d in ('d1', 'd2', 'd3', 'd4', 'full')}
    samp = R[:200000]; samp.tofile(M + '/c_s.bin'); qf = M + '/c_q.bin'
    subprocess.run([RP + '/replay', M + '/c_table.bin', M + '/c_s.bin', '200000', '0', qf], check=True)
    q = float(np.fromfile(qf, np.int32).reshape(-1, 8)[:, 3].mean()); print('replay queries per ray (variant A)', round(q, 3), flush=True)
    rays_stream = NSM * 2 * 39 * 32; feats = {}
    for name, o in orders.items():
        o.astype(np.int32).tofile(M + '/c_order.bin'); out = os.path.join(work, 'reuse_%s_%s.txt' % (tag, name.replace(' ', '_')))
        subprocess.run(['nice', '-n', '10', RP + '/reuse', M + '/c_table.bin', base, str(N), M + '/c_order.bin', out], env=dict(os.environ, NSM=str(NSM)), check=True)
        tot, cold, over, h = rc.hist(out); mis = (cold + over + h[480:].sum()) / rays_stream
        share = max(0.0, 1 - L2_MB / bvh_mb)
        feats[name] = dict(q=q, S1=kk * tot / rays_stream, F=kk * mis, D=128 * (cold + over) / rays_stream * share, W=13); print(name, {k: round(v, 2) for k, v in feats[name].items()}, flush=True)
    c = {k: 1 / PM.predict_sm(p, v) for k, v in feats.items()}
    rows = [(name, c['natural'] / c[name]) for name in orders if name != 'natural']
    for name, s in rows: print('%-12s Memory_4 form predicted speedup over the natural (pixel) order: %.2f' % (name, s), flush=True)
    np.savez(os.path.join(CK, 'ck_%s.npz' % tag), R=R, canon=canon, **{k.replace(' ', '_'): v.astype(np.int32) for k, v in orders.items()})
    json.dump(dict(scene=scene, tag=tag, kind=kind, rays=N, q=q, bvh_mib=bvh_mb, l2_mib=L2_MB, nsm=NSM, hit_fraction=float((canon[:, 0] != 0).mean()), camera=camdesc, groups=groups, feats=feats, predicted=dict(rows)),
              open(os.path.join(work, 'predict_%s.json' % tag), 'x'), indent=1)
    for f in os.listdir(M):
        if f.startswith('c_'): os.remove(os.path.join(M, f))
    if os.path.exists(M + '/c_ref.0.bin'): os.remove(M + '/c_ref.0.bin')


def measure(work, scene, tag):
    z = np.load(os.path.join(CK, 'ck_%s.npz' % tag)); R = z['R']; canon = z['canon']; out = []
    for name in NAMES:
        perm = z[name.replace(' ', '_')].astype(np.int64); f = M + '/c_rays.bin'; R[perm].tofile(f)
        r = mr.host(scene, f, M + '/c_out', ['plain'])[0]
        o = np.fromfile(M + '/c_out.0.bin', np.uint32).reshape(-1, 8); os.remove(M + '/c_out.0.bin')
        bad = int((o != canon[perm]).any(1).sum()); out.append(dict(order=name, ms_all=r['ms'], ms=st.median(r['ms']), wrong=bad)); print(name, out[-1]['ms'], 'wrong', bad, flush=True)
    fn = os.path.join(work, 'measure_%s_%s.json' % (tag, time.strftime('%Y%m%dT%H%M%S')))
    json.dump(dict(scene=scene, tag=tag, results=out), open(fn, 'x'), indent=1); print('wrote', fn)
    for f in os.listdir(M):
        if f.startswith('c_'): os.remove(os.path.join(M, f))


if __name__ == '__main__':
    a = sys.argv[1:]
    predict(*a[1:5]) if a[0] == 'predict' else measure(*a[1:4])
