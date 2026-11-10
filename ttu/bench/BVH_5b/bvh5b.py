"""BVH_5b: local query starts for short rays (radius search style).

    bvh5b.py prep BIN_S0 WORK SCENE TMAX [N]   rays, OptiX only run with the GAS, start paths, CPU reference
    bvh5b.py run  BIN_S1 WORK                  root vs local starts, --sched compact, W 4 and 16, warm and cold
    bvh5b.py report WORK

Start node of a ray (segment o + t d, 0 <= t <= tmax): descend from the root while exactly one child
slot box (internal or leaf, decoded from the GAS, expanded by 1e-5) touches the segment and that child is
internal. Then no node outside the start node's subtree touches the segment, so the closest hit is inside
it (and no ancestor's leaf can hold it). Also recorded: whether the start node's box holds the whole segment.
CPU reference: float64 closest hit over the triangles near the segment (uniform grid, cell >= the largest
triangle extent, so checking the cells the segment's box touches plus one ring is exact).
"""
import json, os, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'BVH_5')); sys.path.insert(0, os.path.join(HERE, '..', 'Memory_4'))
import gas  # noqa: E402
from sched_analyze import analyze  # noqa: E402

HARNESS = os.path.join(HERE, '..', '..', 'harness', 'setup3', 'run.py')
EPS = 1e-5


def seg_box(o, d, tmax, box):
    t0, t1 = 0.0, tmax
    for a in range(3):
        lo, hi = box[a][0] - EPS, box[a][1] + EPS
        if abs(d[a]) < 1e-30:
            if o[a] < lo or o[a] > hi:
                return False
            continue
        u, v = (lo - o[a]) / d[a], (hi - o[a]) / d[a]
        if u > v:
            u, v = v, u
        t0, t1 = max(t0, u), min(t1, v)
        if t0 > t1:
            return False
    return True


def start_path(G, root, o, d, tmax):
    cur, path = root, ()
    while True:
        hits = [(kind, child, s) for s, kind, child, box in gas.slots(G, cur) if seg_box(o, d, tmax, box)]
        if len(hits) != 1 or hits[0][0] != 'node':
            return cur, path
        cur, path = hits[0][1], path + (hits[0][2],)


def node_box(G, off):
    """Box of a node = union of its child slot boxes."""
    bs = [box for _, _, _, box in gas.slots(G, off)]
    return [(min(b[a][0] for b in bs), max(b[a][1] for b in bs)) for a in range(3)]


def cpu_ref(tris, rays):
    sys.path.insert(0, os.path.join(HERE, '..', '..', 'harness', 'setup3'))
    from resume_check import closest
    ext = (tris.max(1) - tris.min(1)).max(); cell = max(ext, 1e-3)
    lo = tris.min((0, 1)) - 1e-6
    cmin = np.floor((tris.min(1) - lo) / cell).astype(np.int64)
    dims = cmin.max(0) + 2
    cid = (cmin[:, 0] * dims[1] + cmin[:, 1]) * dims[2] + cmin[:, 2]
    order = np.argsort(cid, kind='stable'); scid = cid[order]
    out = []
    for r in rays:
        o, dd, tmax = r[0:3].astype(np.float64), r[4:7].astype(np.float64), float(r[7])
        e = o + tmax * dd
        a = np.floor((np.minimum(o, e) - lo) / cell).astype(np.int64) - 1
        b = np.floor((np.maximum(o, e) - lo) / cell).astype(np.int64) + 1
        idx = []
        for x in range(max(a[0], 0), min(b[0], dims[0] - 1) + 1):
            for y in range(max(a[1], 0), min(b[1], dims[1] - 1) + 1):
                c0 = (x * dims[1] + y) * dims[2]
                i0 = np.searchsorted(scid, c0 + max(a[2], 0)); i1 = np.searchsorted(scid, c0 + min(b[2], dims[2] - 1) + 1)
                idx.append(order[i0:i1])
        idx = np.concatenate(idx) if idx else np.zeros(0, np.int64)
        if len(idx) == 0:
            out.append(None); continue
        t, k = closest(tris[idx], o, dd, tmax)
        out.append(None if t is None else (t, int(idx[k])))
    return out


def prep(b0, work, scene, tmax, n=8192):
    n = int(n); tmax = float(tmax); d = os.path.join(work, '%s_t%g' % (scene.replace(':', ''), tmax)); os.makedirs(d, exist_ok=True)
    g = np.random.default_rng(2026)
    o = g.uniform(-1, 1, (n, 3)); dd = g.normal(size=(n, 3)); dd /= np.linalg.norm(dd, axis=1, keepdims=True)
    rays = np.zeros((n, 8), np.float32); rays[:, 0:3] = o; rays[:, 4:7] = dd; rays[:, 7] = tmax
    rays.tofile(os.path.join(d, 'rays.bin'))
    ref = os.path.join(d, 'ref')
    r = subprocess.run([sys.executable, HARNESS, b0, 'file:' + os.path.join(d, 'rays.bin'), 'zero', ref + '.json', '--scene', scene,
                        '--warps', '16', '--no-ttu', '--dump-gas', ref + '.gas', '--dump-tris', ref + '.tris'], capture_output=True, text=True)
    if r.returncode:
        sys.exit(r.stdout + r.stderr)
    G = open(ref + '.gas', 'rb').read(); W = gas.walk(G); root = [k for k, v in W.items() if v['depth'] == 0][0]
    paths, depth, inside = [], [], 0
    for i in range(n):
        off, p = start_path(G, root, rays[i, 0:3].astype(np.float64), rays[i, 4:7].astype(np.float64), tmax)
        paths.append(p); depth.append(len(p))
        nb = node_box(G, off); e = rays[i, 0:3] + tmax * rays[i, 4:7]
        inside += all(nb[a][0] - EPS <= min(rays[i, a], e[a]) and max(rays[i, a], e[a]) <= nb[a][1] + EPS for a in range(3))
    open(os.path.join(d, 'starts_local.txt'), 'w').write(''.join((','.join(map(str, p)) if p else 'root') + '\n' for p in paths))
    tris = np.fromfile(ref + '.tris', np.float32).reshape(-1, 3, 3).astype(np.float64)
    cpu = cpu_ref(tris, rays)
    j = json.load(open(ref + '.json'))
    agree = sum((c is None and not h['hit']) or (c is not None and h['hit'] and c[1] == h['prim']) for c, h in zip(cpu, j['ref']))
    json.dump(dict(scene=scene, tmax=tmax, n=n, nodes=len(W), max_depth=max(v['depth'] for v in W.values()), start_depth=depth,
                   segment_inside_start=inside, cpu=cpu, cpu_hits=sum(c is not None for c in cpu), cpu_optix_agree=agree),
              open(os.path.join(d, 'prep.json'), 'w'))
    hist = {k: depth.count(k) for k in sorted(set(depth))}
    print('%s tmax %g: %d nodes (depth %d); start depth %s, mean %.2f; segment inside start box %d of %d; CPU hits %d; CPU = OptiX for %d of %d rays' % (
        scene, tmax, len(W), max(v['depth'] for v in W.values()), hist, np.mean(depth), inside, n, sum(c is not None for c in cpu), agree, n))


def run(b1, work):
    for d in sorted(os.listdir(work)):
        D = os.path.join(work, d)
        if not os.path.exists(os.path.join(D, 'prep.json')):
            continue
        scene = json.load(open(os.path.join(D, 'prep.json')))['scene']
        for w in (4, 16):
            for st, ex in (('warm', ['--passes', '2']), ('cold', ['--flush', '256'])):
                for mode in ('root', 'local'):
                    out = os.path.join(D, '%s_w%d_%s.json' % (mode, w, st))
                    sf = ['--start-file', os.path.join(D, 'starts_local.txt')] if mode == 'local' else []
                    p = subprocess.run([sys.executable, HARNESS, b1, 'file:' + os.path.join(D, 'rays.bin'), 'zero', out, '--scene', scene, '--warps', str(w),
                                        '--sms', '0,0', '--sched', 'compact', '--resume', '16'] + ex + sf, capture_output=True, text=True)
                    ok = 'exit 0' in p.stdout and ' error' not in p.stdout.replace('no error', '')
                    print(d, os.path.basename(out), 'OK' if ok else 'FAILED', flush=True)
                    if not ok:
                        sys.exit(p.stdout[-2000:] + p.stderr[-2000:])


def report(work):
    for d in sorted(os.listdir(work)):
        D = os.path.join(work, d)
        if not os.path.exists(os.path.join(D, 'root_w4_warm.json')):
            continue
        m = json.load(open(os.path.join(D, 'prep.json'))); cpu = m['cpu']
        print('\n## %s, tmax %g: start depth mean %.2f (max tree depth %d), segment inside start box %d of %d, CPU hits %d' % (
            m['scene'], m['tmax'], np.mean(m['start_depth']), m['max_depth'], m['segment_inside_start'], m['n'], m['cpu_hits']))
        for w in (4, 16):
            for st in ('warm', 'cold'):
                rs = {}
                for mode in ('root', 'local'):
                    p = os.path.join(D, '%s_w%d_%s.json' % (mode, w, st)); r = analyze(p); j = json.load(open(p))
                    f32 = lambda h: np.frombuffer(np.uint32(int(h, 16)).tobytes(), np.float32)[0]
                    cbad = 0
                    for c, (bt, bp) in zip(cpu, j['ray_best']):
                        if c is None:
                            cbad += bt != '0xffffffff'
                        else:
                            cbad += bt == '0xffffffff' or bp != c[1] or abs(float(f32(bt)) - c[0]) > 1e-5 * max(1.0, c[0])
                    r['cpu_wrong'] = cbad; rs[mode] = r
                a, b = rs['root'], rs['local']
                print('W%-2d %-4s | root %.4f rays/cyc, q/ray %.3f, lat/ray %.0f | local %.4f rays/cyc, q/ray %.3f, lat/ray %.0f | speedup %.2fx | wrong vs OptiX %d/%d, vs CPU %d/%d' % (
                    w, st, a['rays_per_cycle'], a['queries_per_ray'], a['lat_mean'], b['rays_per_cycle'], b['queries_per_ray'], b['lat_mean'],
                    b['rays_per_cycle'] / a['rays_per_cycle'], a['wrong'] + a['unfinished'], b['wrong'] + b['unfinished'], a['cpu_wrong'], b['cpu_wrong']))


if __name__ == '__main__':
    {'prep': prep, 'run': run, 'report': report}[sys.argv[1]](*sys.argv[2:])
