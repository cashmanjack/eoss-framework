"""Real scene test (queue 6): path tracing rays on Bistro, San Miguel, Power Plant, Lucy (OptiX only, no TTU code).

    scene_run.py WORK SCENE [LOG2_RAYS=23] [DEPTHS=5]        SCENE: bistro | san_miguel | powerplant | lucy   (triangle files made by scene2tris in /dev/shm/scenes/tris)
Rays: camera rays (4096 x 2048 at LOG2_RAYS 23, pixel order = launch order), then cosine weighted bounces from the real closest hits (geometric normal facing the ray,
origin offset 2e-6 of the scene diagonal along the normal), misses end the path, survivors keep pixel order. Depth 1 = camera rays, depth 2 = first bounce, and so on.
Per depth (all closest hit, 10 launches, median, trace time unless stated; every result compared with the pixel order result field by field):
  pixel order; shuffled; Morton of the origin, 4 to 7 bits per axis, and origin 4 bits + direction 2 bits (GPU key generation, CUB sort, gather, trace, scatter: net = all of them);
  oracle grouping by the ray's own hit subtree (depth 3 and full; needs the answer, so a bound);
  shadow rays (any hit, toward a fixed direction, from the same hit points, pixel order) and a minimal shade kernel (the traversal fraction inputs).
Camera: auto (best of 65,536 probe views: 70th percentile of median hit distance among views with >= 80% hits) or fixed (lucy).
"""
import json, os, subprocess, sys, statistics as st, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'Morton')); sys.path.insert(0, os.path.join(HERE, '..', 'Memory_4'))
import morton_run as mr  # noqa: E402
prep = mr.prep
SC = '/dev/shm/scenes/tris'; M2 = '/dev/shm/m2'; T = '/dev/shm/m2t'
SCENES = {'bistro': dict(up=1), 'san_miguel': dict(up=1), 'powerplant': dict(up=1),
          'lucy': dict(up=2, center=(690.7, -121.5, 192.5), cam_dist=3200.0, cam_dir=(0, -1, 0.1), vfov=30.0)}
CAMQ = float(os.environ.get('CAMQ', '0.7')); SUF = '' if CAMQ == 0.7 else '_q%d' % round(100 * CAMQ)   # camera view quantile (of the median hit distance among valid probe views), output name suffix
KEYS = ['gsort:4:0:o', 'gsort:5:0:o', 'gsort:6:0:o', 'gsort:7:0:o', 'gsort:4:2:o']


def host(scene, rays, out, modes, extra=(), tmo=900):
    mr.gpu_free()
    r = subprocess.run(['timeout', '-k', '2', str(tmo), M2 + '/morton', M2 + '/morton_dev.ptx', 'file:%s/%s.tris' % (SC, scene), rays, out, '10'] + list(extra) + modes, env=mr.ENV, capture_output=True, text=True)
    if r.returncode: sys.exit('host failed %d\n%s\n%s' % (r.returncode, r.stdout[-1500:], r.stderr[-1500:]))
    return [json.loads(l) for l in r.stdout.splitlines() if l.startswith('{')]


def trace(scene, R, tag):   # closest hit of R in file order (plain), returns canon
    f = T + '/q_%s.bin' % tag; R.astype(np.float32).tofile(f); o = T + '/qo_' + tag
    host(scene, f, o, ['plain']); c = np.fromfile(o + '.0.bin', np.uint32).reshape(-1, 8); os.remove(o + '.0.bin'); os.remove(f); return c


def camera_rays(pos, fwd, up, W, H, vfov):
    fwd = np.array(fwd, float); fwd /= np.linalg.norm(fwd); up = np.array(up, float); right = np.cross(fwd, up); right /= np.linalg.norm(right); up2 = np.cross(right, fwd)
    th = np.tan(np.radians(vfov) / 2); tw = th * W / H
    x, y = np.meshgrid((np.arange(W) + .5) / W * 2 - 1, 1 - (np.arange(H) + .5) / H * 2)
    d = fwd[None] + (x.ravel() * tw)[:, None] * right[None] + (y.ravel() * th)[:, None] * up2[None]
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    r = np.zeros((W * H, 8), np.float32); r[:, 0:3] = pos; r[:, 4:7] = d; r[:, 7] = 1e16
    return r


def pick_camera(scene, lo, hi, up_axis):
    g = np.random.default_rng(11); n = 8192; az = 8
    ext = hi - lo; diag = float(np.linalg.norm(ext)); pos = lo + g.uniform(0, 1, (n, 3)) * ext
    pos[:, up_axis] = lo[up_axis] + g.uniform(0.03, 0.35, n) * ext[up_axis]
    h = [i for i in range(3) if i != up_axis]; upv = np.zeros(3); upv[up_axis] = 1
    ang = 2 * np.pi * (np.arange(az)[None, :] + g.uniform(size=(n, 1))) / az                       # n x az
    fw = np.zeros((n, az, 3)); fw[:, :, h[0]] = np.cos(ang); fw[:, :, h[1]] = np.sin(ang)
    right = np.cross(fw, upv); hs = np.tan(np.radians(g.uniform(-30, 30, (n, az, 16)))); vs = np.tan(np.radians(g.uniform(-20, 10, (n, az, 16))))
    dd = fw[:, :, None, :] + hs[..., None] * right[:, :, None, :] + vs[..., None] * upv; dd /= np.linalg.norm(dd, axis=-1, keepdims=True)
    rays = np.zeros((n, az, 16, 8), np.float32); rays[..., 0:3] = pos[:, None, None, :]; rays[..., 4:7] = dd; rays[..., 7] = 1e16
    rays = rays.reshape(-1, 8); views = [(i, ang[i, a]) for i in range(n) for a in range(az)]
    c = trace(scene, rays, 'cam')
    hit = (c[:, 0] != 0).reshape(-1, 16); t = np.where(c[:, 0] != 0, c[:, 1].view(np.float32), np.inf).reshape(-1, 16)
    frac = hit.mean(1); tmin = t.min(1); med = np.array([np.median(x[np.isfinite(x)]) if np.isfinite(x).any() else 0 for x in t])
    ok = (frac >= 0.8) & (tmin > 0.002 * diag)
    if not ok.any(): sys.exit('no camera candidate')
    cand = np.nonzero(ok)[0]; order = cand[np.argsort(med[cand])]; v = order[int(CAMQ * (len(order) - 1))]
    i, ang = views[v]; fwd = np.zeros(3); fwd[h[0]] = np.cos(ang); fwd[h[1]] = np.sin(ang)
    print('camera: %d of %d probe views valid; chosen median hit distance %.4g (%.3f of the diagonal), hit fraction %.2f' % (len(cand), len(views), med[v], med[v] / diag, frac[v]), flush=True)
    return pos[i], fwd, dict(valid_views=int(len(cand)), median_hit_dist=float(med[v]), hit_fraction=float(frac[v]))


def bounce(R, canon, tris, g, diag):
    hit = canon[:, 0] != 0; Rh = R[hit]; t = canon[hit, 1].view(np.float32); tr = tris[canon[hit, 2].astype(np.int64)].astype(np.float64)
    n = np.cross(tr[:, 1] - tr[:, 0], tr[:, 2] - tr[:, 0]); n /= np.linalg.norm(n, axis=1, keepdims=True); d = Rh[:, 4:7].astype(np.float64)
    n = np.where((n * d).sum(1, keepdims=True) > 0, -n, n)
    p = Rh[:, 0:3].astype(np.float64) + t[:, None].astype(np.float64) * d + 2e-6 * diag * n
    a = np.where(np.abs(n[:, :1]) > 0.9, np.array([[0, 1.0, 0]]), np.array([[1.0, 0, 0]])); tt = np.cross(a, n); tt /= np.linalg.norm(tt, axis=1, keepdims=True); bb = np.cross(n, tt)
    u1 = g.uniform(size=len(p)); u2 = g.uniform(size=len(p)); rr = np.sqrt(u1); ph = 2 * np.pi * u2
    dn = (rr * np.cos(ph))[:, None] * tt + (rr * np.sin(ph))[:, None] * bb + np.sqrt(1 - u1)[:, None] * n
    nr = np.zeros((len(p), 8), np.float32); nr[:, 0:3] = p; nr[:, 4:7] = dn; nr[:, 7] = 1e16
    sh = nr.copy(); sh[:, 4:7] = np.array([0.3, 0.8, 0.5]) / np.linalg.norm([0.3, 0.8, 0.5])
    return nr, sh


def main(work, scene, lg=23, depths=5):
    os.makedirs(work, exist_ok=True); os.makedirs(T, exist_ok=True); lg = int(lg); depths = int(depths); cfg = SCENES[scene]; up_axis = cfg['up']
    tris = np.fromfile('%s/%s.tris' % (SC, scene), np.float32).reshape(-1, 3, 3); NT = len(tris)
    lo = tris.reshape(-1, 3).min(0).astype(np.float64); hi = tris.reshape(-1, 3).max(0).astype(np.float64); diag = float(np.linalg.norm(hi - lo))
    tiny = T + '/tiny.bin'; np.zeros((1, 8), np.float32).tofile(tiny)
    gasf = T + '/scene.gas'; info = T + '/info.json'
    host(scene, tiny, T + '/tiny_o', ['plain'], ['--info', info, '--dump-gas', gasf]); os.remove(T + '/tiny_o.0.bin')
    inf = json.load(open(info)); print('%s: %d triangles, GAS buffer %.1f MB = %.2f x the 96 MB L2' % (scene, NT, inf['gas_bytes'] / 2**20, inf['gas_bytes'] / 2**20 / 96), flush=True)
    t0 = time.time(); W, kids, root = prep.tree(open(gasf, 'rb').read()); print('tree: %d nodes, %.0f s' % (len(W), time.time() - t0), flush=True); os.remove(gasf)
    W_, H_ = {23: (4096, 2048), 22: (2048, 2048), 21: (2048, 1024), 20: (1024, 1024)}[lg]
    if 'center' in cfg:
        c = np.array(cfg['center']); fd = -np.array(cfg['cam_dir'], float); fd /= np.linalg.norm(fd); pos = c - fd * cfg['cam_dist']; camdesc = dict(fixed=True); vfov = cfg['vfov']
        fwd = fd
    else:
        pos, fwd, camdesc = pick_camera(scene, lo, hi, up_axis); vfov = 60.0
    upv = np.zeros(3); upv[up_axis] = 1
    R = camera_rays(pos, fwd, upv, W_, H_, vfov); g = np.random.default_rng(5); rows = []
    for depth in range(1, depths + 1):
        N = len(R)
        if N < 100000: print('depth %d: only %d rays, stop' % (depth, N)); break
        base = T + '/r_base.bin'; R.tofile(base); out = T + '/o_nat'
        rs = host(scene, base, out, ['plain'] + KEYS + ['shade'])
        canon = np.fromfile(out + '.0.bin', np.uint32).reshape(-1, 8); os.remove(out + '.0.bin')
        nat = st.median(rs[0]['ms']); row = dict(scene=scene, depth=depth, rays=N, hit_fraction=float((canon[:, 0] != 0).mean()), pixel_ms=nat, morton=[], oracle=[], gas_bytes=inf['gas_bytes'], triangles=NT)
        for j, r in enumerate(rs[1:-1], 1):
            o = np.fromfile('%s.%d.bin' % (out, j), np.uint32).reshape(-1, 8); bad = int((o != canon).any(1).sum()); os.remove('%s.%d.bin' % (out, j))
            m = {k: st.median(r[k]) for k in ('keygen', 'sort', 'gather', 'ms', 'scatter')}; m['mode'] = r['mode']; m['wrong'] = bad; m['net'] = m['keygen'] + m['sort'] + m['gather'] + m['ms'] + m['scatter']; row['morton'].append(m)
        row['shade_ms'] = st.median(rs[-1]['ms'])
        Rf = R.copy(); Rf[:, 7] = 0.0; f = T + '/r_floor.bin'; Rf.tofile(f); r = host(scene, f, T + '/o_floor', ['plain'])[0]; os.remove(T + '/o_floor.0.bin'); os.remove(f); row['floor_ms'] = st.median(r['ms'])   # same ray records, tmax 0: no traversal work, ray read and result write only
        for j in range(len(rs) - 1, len(KEYS) + 2):
            f = '%s.%d.bin' % (out, j)
            if os.path.exists(f): os.remove(f)
        # shuffled and oracle orders (trace only, same rays, results compared with the pixel order result)
        codes = mr.oracle_codes(W, kids, root, R, canon, tris)
        row['groups'] = {k: codes[k][1] for k in ('d1', 'd2', 'd3', 'd4', 'full')}
        orders = [('shuffled', np.random.default_rng(7).permutation(N))] + [('oracle ' + k, mr.order_by(codes[k][0])) for k in ('d3', 'full')]
        for name, perm in orders:
            f = T + '/r_ord.bin'; R[perm].tofile(f); r = host(scene, f, T + '/o_ord', ['plain'])[0]
            o = np.fromfile(T + '/o_ord.0.bin', np.uint32).reshape(-1, 8); os.remove(T + '/o_ord.0.bin'); bad = int((o != canon[perm]).any(1).sum()); ms = st.median(r['ms'])
            (row.__setitem__('shuffled_ms', ms) if name == 'shuffled' else row['oracle'].append(dict(order=name, ms=ms, wrong=bad)))
            if name == 'shuffled': row['shuffled_wrong'] = bad
        nr, sh = bounce(R, canon, tris, g, diag)
        sf = T + '/r_sh.bin'; sh.tofile(sf); r = host(scene, sf, T + '/o_sh', ['any'])[0]; os.remove(T + '/o_sh.0.bin'); os.remove(sf); row['shadow_ms'] = st.median(r['ms']); row['shadow_rays'] = len(sh)
        rows.append(row)
        print('depth %d: %d rays, %.1f%% hit; pixel %.3f ms; shuffled x%.2f; ' % (depth, N, 100 * row['hit_fraction'], nat, row['shuffled_ms'] / nat) +
              ' '.join('%s x%.2f (net x%.2f)' % (m['mode'][6:], nat / m['ms'], nat / m['net']) for m in row['morton']) + '; ' + ' '.join('%s x%.2f' % (o['order'], nat / o['ms']) for o in row['oracle']) +
              '; floor %.3f ms; shadow %.3f ms (%d), shade %.3f ms; wrong %d' % (row['floor_ms'], row['shadow_ms'], len(sh), row['shade_ms'], sum(m['wrong'] for m in row['morton']) + sum(o['wrong'] for o in row['oracle']) + row['shuffled_wrong']), flush=True)
        json.dump(dict(scene=scene, triangles=NT, gas_bytes=inf['gas_bytes'], camera=camdesc, rows=rows), open(os.path.join(work, 'scene_%s%s.json' % (scene, SUF)), 'w'), indent=1)
        R = nr
    for f in os.listdir(T): os.remove(os.path.join(T, f))


if __name__ == '__main__': main(*sys.argv[1:5])
