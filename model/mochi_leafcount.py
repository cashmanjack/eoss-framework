"""Mochi count gap diagnostic (F166): replay with the TRUE primitives per leaf slot (low 3 bits of the slot type byte, summing to the primitive count) instead of the box matched count.
    /usr/bin/python3 model/mochi_leafcount.py DUMP_DIR REC_DIR WORK     (DUMP_DIR gas_fNN.bin, particles_fNN.bin of run B; REC_DIR cand_fNN.bin of run B; needs /dev/shm/rp/assign and replay_aabb)
Table v2 = the make_table table with leafstart rebuilt from the type bytes (leaf slots numbered in the same breadth first order as make_table); the replay output column 2 (primitives in entered slots) is then the true count."""
import os, struct, subprocess, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../ttu/bench/BVH_5'); import gas
D, REC, W = sys.argv[1:4]; os.makedirs(W, exist_ok=True)
def run(tab, rays, n, out): subprocess.run(['/dev/shm/rp/replay_aabb', tab, rays, str(n), '0', out], check=True)
print('| frame | measured calls / query | box matched replay | true count replay | box matched error | true count error | per query corr (true) | frac queries true < measured | frac true > measured | mean nodes (true) |'); print('|---|---|---|---|---|---|---|---|---|---|')
for f in range(10):
    P = np.fromfile('%s/particles_f%02d.bin' % (D, f), np.float32).reshape(-1, 8); N = len(P); c, r = P[:, :3], P[:, 6:7]
    np.stack([c - 2 * r, c + 2 * r, c - 2 * r], 1).astype(np.float32).tofile('%s/f%02d.tris' % (W, f))
    gf = '%s/gas_f%02d.bin' % (D, f)
    subprocess.run([sys.executable, HERE + '/replay/make_table.py', gf, '%s/f%02d.tris' % (W, f), '%s/f%02d.table' % (W, f)], env=dict(os.environ, ASSIGN='/dev/shm/rp/assign'), check=True, capture_output=True)
    R = np.zeros((N, 8), np.float32); R[:, :3] = c; R[:, 5] = 1; R[:, 7] = 1e-16; R.tofile('%s/f%02d.rays' % (W, f))
    d = open(gf, 'rb').read(); root = struct.unpack_from('<Q', d, 0x20)[0]; order = [root]; i = 0; types = []
    while i < len(order):
        for s, k, ch, box in gas.slots(d, order[i]):
            if k == 'node': order.append(ch)
            else: types.append(ch)
        i += 1
    cnt = np.array(types) & 7; cnt = np.where(cnt == 0, 8, cnt)
    raw = open('%s/f%02d.table' % (W, f), 'rb').read(); nn, nt, nl = struct.unpack_from('<3i', raw, 0); assert nl == len(cnt), (nl, len(cnt))
    off = 12 + nn * 12 * 32; ls = np.frombuffer(raw, np.int32, nl + 1, off); lt_off = off + 4 * (nl + 1); nlt = ls[-1]
    ls2 = np.concatenate([[0], np.cumsum(cnt)]).astype(np.int32); tail = np.frombuffer(raw, np.uint8, offset=lt_off + 4 * nlt)   # the tris array
    with open('%s/f%02d.table2' % (W, f), 'wb') as o:
        o.write(raw[:off]); o.write(ls2.tobytes()); o.write(np.zeros(ls2[-1], np.int32).tobytes()); o.write(tail.tobytes())
    run('%s/f%02d.table' % (W, f), '%s/f%02d.rays' % (W, f), N, '%s/f%02d.rep' % (W, f)); run('%s/f%02d.table2' % (W, f), '%s/f%02d.rays' % (W, f), N, '%s/f%02d.rep2' % (W, f))
    m = np.fromfile('%s/cand_f%02d.bin' % (REC, f), np.uint32).astype(float)
    a = np.fromfile('%s/f%02d.rep' % (W, f), np.int32).reshape(-1, 8); b = np.fromfile('%s/f%02d.rep2' % (W, f), np.int32).reshape(-1, 8); ca, cb = a[:, 2] * 1.006, b[:, 2] * 1.006
    print('| %d | %.3f | %.3f | %.3f | %+.1f%% | %+.1f%% | %.3f | %.3f | %.3f | %.1f |' % (f, m.mean(), ca.mean(), cb.mean(), 100 * (ca.mean() / m.mean() - 1), 100 * (cb.mean() / m.mean() - 1), np.corrcoef(m, cb)[0, 1], (cb < m - .5).mean(), (cb > m + .5).mean(), b[:, 0].mean()))
