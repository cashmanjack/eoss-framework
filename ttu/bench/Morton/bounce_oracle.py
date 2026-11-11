"""Oracle bound for bounce rays (OptiX only, no TTU code).
    bounce_oracle.py WORK SCENE_N LOG2_PRIMARY
Bounce rays as in netsort_run (origin = camera hit point, random direction). Launch orders, trace time only (10 launches, median, every result compared to the natural order):
natural (pixel order), shuffled, Morton of origin k4/k5 (CPU sort), oracle depth 1,2,3 and full (grouped by the bounce ray's own hit subtree; needs the answer first, so it is a bound, not a method).
"""
import json, os, sys, statistics as st
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'Memory_4'))
import morton_run as mr, netsort_run as ns  # noqa: E402
M = mr.M

def main(work, scene_n, lg):
    os.makedirs(work, exist_ok=True); scene_n = int(scene_n); n = 1 << int(lg)
    R = ns.make(scene_n, n, 'bounce'); N = len(R)
    base = M + '/b_base.bin'; R.tofile(base); ref = M + '/b_ref'
    rs = mr.host(scene_n, base, ref, ['plain'], ['--dump-gas', ref + '.gas', '--dump-tris', ref + '.tris'])
    canon = np.fromfile(ref + '.0.bin', np.uint32).reshape(-1, 8)
    tris = np.fromfile(ref + '.tris', np.float32).reshape(-1, 3, 3)
    W, kids, root = mr.prep.tree(open(ref + '.gas', 'rb').read())
    codes = mr.oracle_codes(W, kids, root, R, canon, tris)
    print('scene %d: %d bounce rays, %.1f%% hit; groups d1 %d d2 %d d3 %d full %d' % (scene_n, N, 100 * (canon[:, 0] != 0).mean(), *[codes[k][1] for k in ('d1', 'd2', 'd3', 'full')]), flush=True)
    rng = np.random.default_rng(7)
    orders = [('natural', np.arange(N)), ('shuffled', rng.permutation(N)),
              ('morton-o k4', mr.order_by(mr.code_o(R, 4))), ('morton-o k5', mr.order_by(mr.code_o(R, 5)))]
    for d in (1, 2, 3): orders.append(('oracle d%d' % d, mr.order_by(codes['d%d' % d][0])))
    orders.append(('oracle full', mr.order_by(codes['full'][0])))
    out = []
    for name, perm in orders:
        f = M + '/b_rays.bin'; R[perm].tofile(f)
        r = mr.host(scene_n, f, M + '/b_out', ['plain'])[0]
        o = np.fromfile(M + '/b_out.0.bin', np.uint32).reshape(-1, 8); os.remove(M + '/b_out.0.bin')
        bad = int((o != canon[perm]).any(1).sum()); ms = st.median(r['ms'])
        out.append(dict(order=name, ms=ms, rays_per_us=N / ms / 1000, wrong=bad))
    nat = out[0]['ms']; shuf = out[1]['ms']
    for r in out: print('%-12s %8.3f ms %7.0f rays/us  x natural %.2f  x shuffled %.2f  wrong %d' % (r['order'], r['ms'], r['rays_per_us'], nat / r['ms'], shuf / r['ms'], r['wrong']), flush=True)
    json.dump(dict(scene=scene_n, rays=N, results=out), open(os.path.join(work, 'bounce_oracle_s%d.json' % scene_n), 'w'), indent=1)
    for f in os.listdir(M):
        if f.startswith('b_'): os.remove(os.path.join(M, f))

if __name__ == '__main__': main(*sys.argv[1:4])
