"""Leaf slot type byte vs the number of triangles in the slot's box (OptiX only; our own GAS built in our own process and copied out with cudaMemcpy by the morton host).
    leafscan.py WORKDIR NAME   (NAME.gas and NAME.tris in WORKDIR, tris = float3 x3 per triangle)
Triangles per leaf slot = triangles whose box centre lies inside the slot box (valid when triangles are well separated)."""
import sys, struct, collections, numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0]); import gas

def scan(d, tris):
    c = tris.mean(1)  # centroid
    W = gas.walk(d); rows = []
    for off in W:
        for s, kind, ty, box in [(s, k, c_, b) for s, k, c_, b in gas.slots(d, off) if k == 'leaf']:
            lo = np.array([b[0] for b in box]); hi = np.array([b[1] for b in box])
            m = ((c >= lo - 1e-6) & (c <= hi + 1e-6)).all(1)
            rows.append((ty, int(m.sum()), d[off + 0x20 + 8 * s + 6], off, s))
    return W, rows

if __name__ == '__main__':
    wd, name = sys.argv[1:3]
    d = open('%s/%s.gas' % (wd, name), 'rb').read(); tris = np.fromfile('%s/%s.tris' % (wd, name), np.float32).reshape(-1, 3, 3)
    W, rows = scan(d, tris)
    t = collections.defaultdict(collections.Counter)
    for ty, n, b6, off, s in rows: t[ty][n] += 1
    for ty in sorted(t): print('type 0x%02x  (hi %2d lo %d): triangles in box ->' % (ty, (ty >> 3) & 15, ty & 7), dict(sorted(t[ty].items())))
    print('leaf slots', len(rows), 'triangles', len(tris), 'sum of box counts', sum(r[1] for r in rows), 'header +0x60', struct.unpack_from('<I', d, 0x60)[0])
