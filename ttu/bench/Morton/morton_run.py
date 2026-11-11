"""Morton sort on OptiX (key paper 2 test) and grouping depth. OptiX only, no TTU code.

    morton_run.py WORK SCENE_N LOG2_RAYS PART         PART: sweep | depth | all
Random rays (prep.make_rays seed 2026), N = 2^LOG2_RAYS. Orders (launch order = file order):
  random; morton-o k (Morton code of origin, k bits per axis, ties keep random order);
  morton-od k kd (origin code major, direction code minor, kd bits per axis); morton-do (direction major);
  oracle (full subtree key of the hit triangle's box; origin for a miss) and oracle depth 1, 2, 3 (key prefix).
Every order x {plain, SER hint} is timed (10 launches, CUDA events, median); every result is compared with the random order plain result (all fields, per ray).
"""
import json, os, subprocess, sys, statistics as st
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'Memory_4'))
import prep  # noqa: E402

M = '/dev/shm/m1'
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES='1', LD_LIBRARY_PATH=os.path.expanduser('~/opt/nvoptix-580.178.04') + ':' + os.environ.get('LD_LIBRARY_PATH', ''))


def gpu_free():
    p = subprocess.run(['nvidia-smi', '-i', '1', '--query-compute-apps=pid', '--format=csv,noheader'], capture_output=True, text=True).stdout.strip()
    if p: sys.exit('GPU 1 busy: ' + p)


def host(scene_n, rays, out, modes, extra=()):
    gpu_free()
    r = subprocess.run(['timeout', '-k', '2', '120', M + '/morton', M + '/morton_dev.ptx', str(scene_n), rays, out, '10'] + list(extra) + modes, env=ENV, capture_output=True, text=True)
    if r.returncode: sys.exit('host failed %d\n%s\n%s' % (r.returncode, r.stdout[-1500:], r.stderr[-1500:]))
    return [json.loads(l) for l in r.stdout.splitlines() if l.startswith('{')]


def spread(x, k):    # interleave k bits of each of 3 integer arrays x (list of 3 arrays) -> 3k bit code
    c = np.zeros(len(x[0]), np.uint64)
    for b in range(k):
        for a in range(3):
            c |= ((x[a] >> np.uint64(b)) & np.uint64(1)) << np.uint64(3 * b + a)
    return c


def q(v, k):   # [-1,1] -> k bit integer
    return np.minimum(((v + 1.0) * 0.5 * (1 << k)).astype(np.int64), (1 << k) - 1).astype(np.uint64)


def code_o(R, k): return spread([q(R[:, a].astype(np.float64), k) for a in range(3)], k)
def code_d(R, k): return spread([q(R[:, 4 + a].astype(np.float64), k) for a in range(3)], k)


def order_by(code):    # stable sort: ties keep the random base order
    return np.argsort(code, kind='stable')



def oracle_codes(W, kids, root, R, canon, tris, tol=1e-6):
    """Vectorized prep.key_of for every ray: key = deepest node reached by always taking the smallest child box that contains the hit triangle's box
    (the origin point for a miss). Returns {'d1','d2','d3','full': (group id per ray, number of groups)}; depth d group = ancestor at depth d (the final node if shallower)."""
    offs = sorted(W, key=lambda o: W[o]['path']); nid = {o: i for i, o in enumerate(offs)}
    nn = len(offs)
    cid = -np.ones((nn, 12), np.int64); box = np.zeros((nn, 12, 6))
    for o in offs:
        for j, c in enumerate(sorted(kids.get(o, []), key=lambda c: W[c]['path'])):
            cid[nid[o], j] = nid[c]; bx = W[c]['box']; box[nid[o], j] = [bx[0][0], bx[1][0], bx[2][0], bx[0][1], bx[1][1], bx[2][1]]
    n = len(R); hit = canon[:, 0] != 0
    lo = R[:, 0:3].astype(np.float64).copy(); hi = lo.copy()
    t = tris[canon[hit, 2].astype(np.int64)].astype(np.float64)
    lo[hit] = t.min(1); hi[hit] = t.max(1)
    cur = np.full(n, nid[root], np.int64); anc = [cur.copy()]
    for lvl in range(40):
        new = cur.copy(); CH = 1 << 19
        for s0 in range(0, n, CH):
            sl = slice(s0, s0 + CH); c = cur[sl]; b = box[c]; ch = cid[c]
            ok = (ch >= 0) & (b[:, :, 0:3] - tol <= lo[sl][:, None, :]).all(2) & (hi[sl][:, None, :] <= b[:, :, 3:6] + tol).all(2)
            vol = np.where(ok, (b[:, :, 3:6] - b[:, :, 0:3]).sum(2), np.inf)
            j = vol.argmin(1)
            new[sl] = np.where(ok.any(1), ch[np.arange(len(c)), j], c)
        if not (new != cur).any(): break
        cur = new; anc.append(cur.copy())
    final = cur; depth = np.array([len(W[o]['path']) for o in offs]); out = {}
    for name, d in (('d1', 1), ('d2', 2), ('d3', 3), ('d4', 4), ('full', 99)):
        gid = final if d == 99 else np.where(depth[final] <= d, final, anc[min(d, len(anc) - 1)])
        u, inv = np.unique(gid, return_inverse=True)
        out[name] = (inv.astype(np.uint64), len(u))
    return out

def main(work, scene_n, lg, part):
    os.makedirs(work, exist_ok=True)
    n = 1 << int(lg); scene_n = int(scene_n); tag = 's%d' % scene_n
    R = prep.make_rays(n)
    base = M + '/base_%s.bin' % tag; R.tofile(base)
    ref = M + '/ref_%s' % tag
    res = host(scene_n, base, ref, ['plain'], ['--dump-gas', ref + '.gas', '--dump-tris', ref + '.tris'])
    canon = np.fromfile(ref + '.0.bin', np.uint32).reshape(-1, 8)
    print('reference run: %.0f rays/us, %d of %d hit' % (n / st.median(res[0]['ms']) / 1000, int((canon[:, 0] != 0).sum()), n), flush=True)
    tris = np.fromfile(ref + '.tris', np.float32).reshape(-1, 3, 3)
    W, kids, root = prep.tree(open(ref + '.gas', 'rb').read())
    codes = oracle_codes(W, kids, root, R, canon, tris)
    print('groups: depth 1 %d, depth 2 %d, depth 3 %d, full %d' % (codes['d1'][1], codes['d2'][1], codes['d3'][1], codes['full'][1]), flush=True)

    def hint(kind, order=None):   # hint per ray in launch order, and bits
        if kind == 'o4': h, b = code_o(R, 4), 12
        elif kind == 'od': h, b = (code_o(R, 3) << np.uint64(3)) | code_d(R, 1), 12
        elif kind == 'o6': h, b = code_o(R, 6), 18
        elif kind == 'd2': h, b = codes['d2'][0], 8
        elif kind == 'd3': h, b = codes['d3'][0], 11
        return (h if order is None else h[order]).astype(np.uint32), b

    results = []

    def run(name, order, sers=()):
        """name: label; order: permutation (None = random); sers: SER hint kinds to time on this order."""
        perm = np.arange(n) if order is None else order
        f = M + '/rays_%s.bin' % tag; R[perm].tofile(f)
        modes = ['plain']; mnames = ['plain']
        for k in sers:
            h, b = hint(k, perm); hf = M + '/hint_%s_%s.bin' % (tag, k); h.tofile(hf); modes.append('ser:%s:%d' % (hf, b)); mnames.append('ser-' + k)
        out = M + '/out_' + tag
        rs = host(scene_n, f, out, modes)
        for j, r in enumerate(rs):
            o = np.fromfile('%s.%d.bin' % (out, j), np.uint32).reshape(-1, 8)
            bad = int((o != canon[perm]).any(axis=1).sum()); os.remove('%s.%d.bin' % (out, j))
            ms = st.median(r['ms']); rate = n / ms / 1000
            results.append(dict(order=name, mode=mnames[j], ms=ms, rays_per_us=rate, launches=r['ms'], wrong=bad))
            print('%-22s %-9s %8.3f ms %7.0f rays/us  wrong %d' % (name, mnames[j], ms, rate, bad), flush=True)

    sers = ('o4', 'od', 'd3')
    if part in ('sweep', 'all'):
        run('random', None, sers + ('o6', 'd2'))
        for k in (1, 2, 3, 4, 5, 6, 8, 10):
            run('morton-o k%d' % k, order_by(code_o(R, k)), sers if k == 4 else ())
        for k, kd in ((3, 1), (4, 1), (5, 1), (3, 2), (4, 2), (5, 2), (6, 1), (6, 2), (8, 1), (8, 2)):
            run('morton-od k%d d%d' % (k, kd), order_by((code_o(R, k) << np.uint64(3 * kd)) | code_d(R, kd)), sers if (k, kd) == (3, 1) else ())
        for k, kd in ((1, 3), (2, 3), (1, 4), (2, 4), (3, 3), (3, 4)):
            run('morton-do k%d d%d' % (k, kd), order_by((code_d(R, kd) << np.uint64(3 * k)) | code_o(R, k)))
        run('oracle full', order_by(codes['full'][0]), sers)
    if part in ('depth', 'all'):
        for d in (1, 2, 3):
            run('oracle depth %d (%d groups)' % (d, codes['d%d' % d][1]), order_by(codes['d%d' % d][0]))
        run('oracle full', order_by(codes['full'][0]))
        run('random', None)
    json.dump(dict(scene=scene_n, rays=n, groups={k: v[1] for k, v in codes.items()}, results=results), open(os.path.join(work, 'morton_%s_%s.json' % (tag, part)), 'w'), indent=1)
    for f in os.listdir(M):
        if f.endswith('.bin') or f.endswith('.gas') or f.endswith('.tris'): os.remove(os.path.join(M, f))


if __name__ == '__main__':
    main(*sys.argv[1:5])
