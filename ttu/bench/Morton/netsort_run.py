"""Net Morton speedup with a GPU sort (key generation + CUB radix sort + ray gather + result scatter), OptiX only, no TTU code.

    netsort_run.py WORK SCENE_N LOG2_RAYS SET        SET: random | camera | bounce | radius05 | radius20
Ray sets (all in the soup scene's [-1,1]^3 cube):
  random    origin uniform in the cube, uniform random direction, tmax 1e16 (file order = random)
  camera    pinhole at (0,0,-4) looking +z, tan(half angle) 0.4, W x H pixels (row major = natural launch order), tmax 1e16
  bounce    from every camera ray that hits: origin = hit point, uniform random direction on the sphere (no normals), tmin 1e-4, tmax 1e16
  radius05 / radius20   random origin, random direction, tmax 0.05 / 0.20 (short search style queries)
Baselines (plain launch): natural file order, and a random shuffle of the same rays. Sorted modes use the natural file on the GPU.
Time per sorted run = keygen + sort + gather (before the trace), trace, scatter of results back to the original index; median of 10 launches.
Every sorted result is compared field by field against the plain natural order result of the same rays.
"""
import json, os, subprocess, sys, statistics as st
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'Memory_4'))
import morton_run as mr  # noqa: E402
M = mr.M
KEYS = ['gsort:4:0:o', 'gsort:5:0:o', 'gsort:6:0:o', 'gsort:5:1:o', 'gsort:4:2:o', 'gsort:0:4:d', 'gsort:0:6:d', 'gsort:0:8:d']


def host(scene_n, rays, out, modes):
    mr.gpu_free()
    r = subprocess.run(['timeout', '-k', '2', '400', M + '/morton', M + '/morton_dev.ptx', str(scene_n), rays, out, '10'] + modes, env=mr.ENV, capture_output=True, text=True)
    if r.returncode: sys.exit('host failed %d\n%s\n%s' % (r.returncode, r.stdout[-1500:], r.stderr[-1500:]))
    return [json.loads(l) for l in r.stdout.splitlines() if l.startswith('{')]


def sphere(g, n):
    d = g.normal(size=(n, 3)); return d / np.linalg.norm(d, axis=1, keepdims=True)


def camera(n):
    W = 1 << ((int(np.log2(n)) + 1) // 2); H = n // W
    x, y = np.meshgrid((np.arange(W) + .5) / W * 2 - 1, (np.arange(H) + .5) / H * 2 - 1)
    d = np.stack([0.4 * x.ravel(), 0.4 * y.ravel(), np.ones(W * H)], 1); d /= np.linalg.norm(d, axis=1, keepdims=True)
    r = np.zeros((W * H, 8), np.float32); r[:, 0:3] = (0, 0, -4); r[:, 4:7] = d; r[:, 7] = 1e16
    return r


def make(scene_n, n, kind):
    g = np.random.default_rng(2026)
    if kind == 'random': return mr.prep.make_rays(n)
    if kind in ('radius05', 'radius20'):
        r = mr.prep.make_rays(n); r[:, 7] = 0.05 if kind == 'radius05' else 0.20; return r
    P = camera(n)
    if kind == 'camera': return P
    f = M + '/prim.bin'; P.tofile(f)
    res = np.fromfile(None, np.uint32) if False else None
    host(scene_n, f, M + '/prim_out', ['plain']); o = np.fromfile(M + '/prim_out.0.bin', np.uint32).reshape(-1, 8); os.remove(M + '/prim_out.0.bin'); os.remove(f)
    hit = o[:, 0] != 0; t = o[hit, 1].view(np.float32)
    r = np.zeros((int(hit.sum()), 8), np.float32); r[:, 0:3] = P[hit, 0:3] + t[:, None] * P[hit, 4:7]
    r[:, 3] = 1e-4; r[:, 4:7] = sphere(g, len(r)); r[:, 7] = 1e16
    return r


def main(work, scene_n, lg, kind):
    os.makedirs(work, exist_ok=True); scene_n = int(scene_n); n = 1 << int(lg)
    R = make(scene_n, n, kind); N = len(R); res = {}
    f = M + '/rays.bin'; R.tofile(f)
    rs = host(scene_n, f, M + '/o_nat', ['plain'] + KEYS)
    canon = np.fromfile(M + '/o_nat.0.bin', np.uint32).reshape(-1, 8)
    nat = st.median(rs[0]['ms']); res['natural'] = nat; rows = []
    print('%s scene %d: %d rays, %.1f%% hit; natural order %.3f ms = %.0f rays/us' % (kind, scene_n, N, 100 * (canon[:, 0] != 0).mean(), nat, N / nat / 1000), flush=True)
    for j, r in enumerate(rs[1:], 1):
        o = np.fromfile('%s.%d.bin' % (M + '/o_nat', j), np.uint32).reshape(-1, 8); bad = int((o != canon).any(1).sum()); os.remove('%s.%d.bin' % (M + '/o_nat', j))
        m = {k: st.median(r[k]) for k in ('keygen', 'sort', 'gather', 'ms', 'scatter')}; m['mode'] = r['mode']; m['wrong'] = bad; rows.append(m)
    os.remove(M + '/o_nat.0.bin')
    perm = np.random.default_rng(7).permutation(N); R[perm].tofile(f)
    shuf = st.median(host(scene_n, f, M + '/o_shuf', ['plain'])[0]['ms']); os.remove(M + '/o_shuf.0.bin'); os.remove(f)
    res['shuffled'] = shuf
    print('shuffled order %.3f ms = %.0f rays/us' % (shuf, N / shuf / 1000))
    print('%-12s %7s %7s %7s %8s %7s %8s %8s %8s %6s' % ('key', 'keygen', 'sort', 'gather', 'trace', 'scatter', 'net ms', 'x shuf', 'x nat', 'wrong'))
    for m in rows:
        net = m['keygen'] + m['sort'] + m['gather'] + m['ms'] + m['scatter']; m['net'] = net
        print('%-12s %7.3f %7.3f %7.3f %8.3f %7.3f %8.3f %8.2f %8.2f %6d' % (m['mode'][6:], m['keygen'], m['sort'], m['gather'], m['ms'], m['scatter'], net, shuf / net, nat / net, m['wrong']), flush=True)
    json.dump(dict(scene=scene_n, rays=N, kind=kind, baseline=res, rows=rows), open(os.path.join(work, 'netsort_s%d_%s_%d.json' % (scene_n, kind, int(lg))), 'w'), indent=1)


if __name__ == '__main__': main(*sys.argv[1:5])
