"""Memory_4 prep: one fixed set of incoherent rays, written in two orders.

    prep.py BIN_S0 OUTDIR SCENE [N]      (BIN_S0: a checked stage 0 setup3 binary; no TTU code runs here)

1. N random rays (origin uniform in [-1,1]^3, uniform random direction, tmin 0, tmax 1e16; seed 2026).
2. An OptiX only run (--no-ttu) gives every ray's closest hit, the GAS and the triangles.
3. Key of a ray: slot path of the deepest internal node whose box contains the hit triangle's box
   (smallest such child at each level), or the ray origin for a miss.
4. grouped = rays sorted by key (each warp's rays enter the same subtree as far as group sizes allow);
   interleaved = a random permutation (seed 7) of the same set. Writes rays_<order>.bin and prep.json.
"""
import json, os, random, struct, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'BVH_5'))
import gas  # noqa: E402

HARNESS = os.path.join(HERE, '..', '..', 'harness', 'setup3', 'run.py')


def make_rays(n, seed=2026):
    g = np.random.default_rng(seed)
    o = g.uniform(-1, 1, (n, 3))
    d = g.normal(size=(n, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
    r = np.zeros((n, 8), np.float32)
    r[:, 0:3] = o; r[:, 3] = 0; r[:, 4:7] = d; r[:, 7] = 1e16
    return r


def tree(G):
    W = gas.walk(G)
    kids = {}
    for off, n in W.items():
        if n['parent'] is not None:
            kids.setdefault(n['parent'], []).append(off)
    root = [o for o, n in W.items() if n['depth'] == 0][0]
    return W, kids, root


def key_of(W, kids, root, lo, hi, tol=1e-6):
    path, cur = (), root
    while True:
        c = [k for k in kids.get(cur, []) if all(W[k]['box'][a][0] - tol <= lo[a] and hi[a] <= W[k]['box'][a][1] + tol for a in range(3))]
        if not c:
            return path
        cur = min(c, key=lambda k: sum(W[k]['box'][a][1] - W[k]['box'][a][0] for a in range(3)))
        path = W[cur]['path']


def main(bin0, out, scene, n=2048):
    n = int(n); os.makedirs(out, exist_ok=True)
    rays = make_rays(n)
    base = os.path.join(out, 'rays_base.bin'); rays.tofile(base)
    ref = os.path.join(out, 'ref')
    r = subprocess.run([sys.executable, HARNESS, bin0, 'file:' + base, 'zero', ref + '.json', '--scene', scene, '--warps', '16',
                        '--no-ttu', '--dump-gas', ref + '.gas', '--dump-tris', ref + '.tris'], capture_output=True, text=True)
    if r.returncode:
        sys.exit(r.stdout + r.stderr)
    j = json.load(open(ref + '.json'))
    W, kids, root = tree(open(ref + '.gas', 'rb').read())
    tris = np.fromfile(ref + '.tris', np.float32).reshape(-1, 3, 3)
    keys = []
    for i in range(n):
        h = j['ref'][i]
        if h['hit']:
            t = tris[h['prim']]; keys.append(key_of(W, kids, root, t.min(0), t.max(0)))
        else:
            p = rays[i, 0:3]; keys.append(key_of(W, kids, root, p, p))
    grouped = sorted(range(n), key=lambda i: (keys[i], i))
    inter = list(range(n)); random.Random(7).shuffle(inter)
    for name, perm in (('grouped', grouped), ('interleaved', inter)):
        rays[perm].tofile(os.path.join(out, 'rays_%s.bin' % name))

    def per_warp(perm, depth):
        return float(np.mean([len({keys[i][:depth] for i in perm[w:w + 32]}) for w in range(0, n, 32)]))
    depths = [len(k) for k in keys]
    meta = dict(scene=scene, n=n, gas_size=j['gas_size'], nodes=len(W), max_depth=max(v['depth'] for v in W.values()),
                hits=sum(h['hit'] for h in j['ref']), key_depth=dict(sorted({d: depths.count(d) for d in set(depths)}.items())),
                orders=dict(grouped=grouped, interleaved=inter), keys=[list(k) for k in keys],
                distinct_per_warp={o: {d: per_warp(p, d) for d in (1, 2, 3, 4)} for o, p in (('grouped', grouped), ('interleaved', inter))})
    json.dump(meta, open(os.path.join(out, 'prep.json'), 'w'))
    print('%s: GAS %.1f MB, %d nodes, depth %d, %d of %d rays hit, key depth %s' % (
        scene, j['gas_size'] / 2 ** 20, len(W), meta['max_depth'], meta['hits'], n, meta['key_depth']))
    for o in ('grouped', 'interleaved'):
        print('  %-11s distinct subtrees per warp at depth 1..4: %s' % (o, ['%.1f' % v for v in meta['distinct_per_warp'][o].values()]))


if __name__ == '__main__':
    main(*sys.argv[1:])
