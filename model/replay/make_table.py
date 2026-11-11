"""Build the replay table from a GAS dump and its triangles (CPU only).   make_table.py GAS TRIS OUT.bin
Triangles are put into leaf slots by box containment (smallest containing slot, leaf over node on a tie): the triangle to leaf mapping is not decoded."""
import os, struct, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'ttu', 'bench', 'BVH_5'))
import gas  # noqa: E402


def build(gas_path, tris_path, out, tol=1e-3):
    d = open(gas_path, 'rb').read(); tris = np.fromfile(tris_path, np.float32).reshape(-1, 3, 3)
    root = struct.unpack_from('<Q', d, 0x20)[0]
    idx = {root: 0}; order = [root]; i = 0
    while i < len(order):                     # breadth first node numbering, root = 0
        for s, k, ch, box in gas.slots(d, order[i]):
            if k == 'node': idx[ch] = len(order); order.append(ch)
        i += 1
    nn = len(order); typ = np.zeros((nn, 12), np.int32); child = np.zeros((nn, 12), np.int32); box = np.zeros((nn, 12, 6), np.float32)
    nl = 0
    for o in order:
        for s, k, ch, bx in gas.slots(d, o):
            typ[idx[o], s] = 1 if k == 'node' else 2
            child[idx[o], s] = idx[ch] if k == 'node' else nl
            if k == 'leaf': nl += 1
            box[idx[o], s] = [bx[0][0], bx[1][0], bx[2][0], bx[0][1], bx[1][1], bx[2][1]]
    n = len(tris)
    pre = out + '.pre'
    with open(pre, 'wb') as f:
        f.write(struct.pack('<3i', nn, n, nl))
        rec = np.zeros(nn * 12, dtype=[('t', '<i4'), ('c', '<i4'), ('b', '<f4', 6)]); rec['t'] = typ.ravel(); rec['c'] = child.ravel(); rec['b'] = box.reshape(-1, 6)
        f.write(rec.tobytes()); f.write(tris.astype('<f4').tobytes())
    import subprocess
    subprocess.run([os.environ.get('ASSIGN', os.path.join(HERE, 'assign')), pre, out + '.leaf', repr(tol)], check=True); os.remove(pre)
    pr = np.fromfile(out + '.leaf', np.int32).astype(np.int64).reshape(-1, 2); os.remove(out + '.leaf')
    order_t = np.lexsort((pr[:, 0], pr[:, 1])); pr = pr[order_t]; tri_ids = pr[:, 0]
    cnt = np.bincount(pr[:, 1], minlength=nl); start = np.concatenate([[0], np.cumsum(cnt)]).astype(np.int32)
    ass = np.zeros(n, bool); ass[tri_ids] = True
    with open(out, 'wb') as f:
        f.write(struct.pack('<3i', nn, n, nl))
        rec = np.zeros(nn * 12, dtype=[('t', '<i4'), ('c', '<i4'), ('b', '<f4', 6)]); rec['t'] = typ.ravel(); rec['c'] = child.ravel(); rec['b'] = box.reshape(-1, 6)
        f.write(rec.tobytes()); f.write(start.tobytes()); f.write(tri_ids.astype('<i4').tobytes()); f.write(tris.astype('<f4').tobytes())
    print('nodes %d, leaf slots %d (empty %d), triangles %d, assigned %d (%.2f%%), pairs %d, per leaf slot mean %.2f max %d' % (
        nn, nl, int((cnt == 0).sum()), n, int(ass.sum()), 100 * ass.mean(), len(pr), cnt.mean(), cnt.max()))


if __name__ == '__main__':
    build(*sys.argv[1:4])
