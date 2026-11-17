"""Box quantization false positives (predictions: model/predictions_struct_quant.md). CPU only.
usage: /usr/bin/python3 model/struct_quant.py DUMP_PREFIX RES_FILE WORK [segments]     (custom primitive scene: PREFIX.gas .aabb .rays from Struct_1, RES_FILE = per ray Out records of the hardware launch)
Needs /dev/shm/rp/assign and /dev/shm/rp/replay_quant (g++ -O2 model/replay/assign.cpp, replay_quant.cpp).
Q table = make_table (decoded quantized boxes, prims into leaf slots by box matching); E1 table = same, every slot box replaced by the tight float union of the primitive boxes below it."""
import os, struct, subprocess, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../ttu/bench/BVH_5'); import gas
pre, resf, W = sys.argv[1:4]; os.makedirs(W, exist_ok=True)
tag = os.path.basename(pre)
bx = np.fromfile(pre + '.aabb', np.float32).reshape(-1, 6); N = len(bx)
np.stack([bx[:, :3], bx[:, 3:], bx[:, :3]], 1).astype(np.float32).tofile(W + '/' + tag + '.tris')   # a "triangle" holds the box (lo, hi, lo)
env = dict(os.environ, ASSIGN='/dev/shm/rp/assign')
r = subprocess.run([sys.executable, HERE + '/replay/make_table.py', pre + '.gas', W + '/' + tag + '.tris', W + '/' + tag + '.table'], env=env, check=True, capture_output=True, text=True); print(r.stdout.strip())
raw = open(W + '/' + tag + '.table', 'rb').read(); nn, nt, nl = struct.unpack_from('<3i', raw, 0)
rec = np.frombuffer(raw, np.dtype([('t', '<i4'), ('c', '<i4'), ('b', '<f4', 6)]), nn * 12, 12).copy().reshape(nn, 12)
off = 12 + nn * 12 * 32; ls = np.frombuffer(raw, np.int32, nl + 1, off); lt = np.frombuffer(raw, np.int32, int(ls[-1]), off + 4 * (nl + 1))
# true prims per leaf slot from the type bytes (breadth first slot order = make_table's), as an assignment check
d = open(pre + '.gas', 'rb').read(); root = struct.unpack_from('<Q', d, 0x20)[0]; order = [root]; i = 0; types = []
while i < len(order):
    for s, k, ch, box in gas.slots(d, order[i]):
        if k == 'node': order.append(ch)
        else: types.append(ch)
    i += 1
cnt = np.array(types) & 7; cnt = np.where(cnt == 0, 8, cnt); asg = np.diff(ls)
print('leaf slots %d, assignment count equals the type byte count in %.4f of slots, prims %d assigned %d (type bytes sum %d)' % (nl, (cnt == asg).mean(), N, len(lt), cnt.sum()))
# exact boxes, bottom up (children of a node have larger table indices: breadth first numbering)
ex = np.zeros((nn, 12, 6), np.float32); has = np.zeros((nn, 12), bool); nodebox = np.zeros((nn, 6), np.float32); nodehas = np.zeros(nn, bool)
for n in range(nn - 1, -1, -1):
    lo = np.full(3, np.inf); hi = np.full(3, -np.inf)
    for s in range(12):
        t = rec['t'][n, s]
        if t == 0: continue
        if t == 2:
            p = lt[ls[rec['c'][n, s]]:ls[rec['c'][n, s] + 1]]
            if len(p) == 0: has[n, s] = False; continue
            b = np.concatenate([bx[p, :3].min(0), bx[p, 3:].max(0)])
        else:
            c = rec['c'][n, s]
            if not nodehas[c]: continue
            b = nodebox[c]
        ex[n, s] = b; has[n, s] = True; lo = np.minimum(lo, b[:3]); hi = np.maximum(hi, b[3:])
    if has[n].any(): nodebox[n] = np.concatenate([lo, hi]); nodehas[n] = True
live = (rec['t'] > 0) & has; q = rec['b']
eps = 1e-5 * max(1.0, float(np.abs(bx).max()) * 1e-3)
contains = (q[..., :3] <= ex[..., :3] + eps).all(-1) & (q[..., 3:] >= ex[..., 3:] - eps).all(-1)
slack = ((q[..., 3:] - q[..., :3]) - (ex[..., 3:] - ex[..., :3]))
print('slots live %d; quantized box contains the exact box in %.5f of them; mean extra width per axis / exact width: %.4f; slots with extra width > 0.1 of the exact width on some axis: %.4f' % (live.sum(), contains[live].mean(), np.mean(((slack / np.maximum(ex[..., 3:] - ex[..., :3], 1e-9))[live]).mean(-1)), (((slack / np.maximum(ex[..., 3:] - ex[..., :3], 1e-9)) > 0.1).any(-1)[live]).mean()))
rec2 = rec.copy(); rec2['b'] = np.where(has[..., None], ex, q)
with open(W + '/' + tag + '.tableE1', 'wb') as f:
    f.write(raw[:12]); f.write(rec2.tobytes()); f.write(raw[off:])
R = np.fromfile(pre + '.rays', np.float32).reshape(-1, 8); n = len(R)
def run(tab, out): subprocess.run(['/dev/shm/rp/replay_quant', tab, pre + '.rays', str(n), '0', out], check=True); return np.fromfile(out, np.int32).reshape(n, 8)
Qr = run(W + '/' + tag + '.table', W + '/' + tag + '.repQ'); Er = run(W + '/' + tag + '.tableE1', W + '/' + tag + '.repE1')
Hres = np.fromfile(resf, np.uint32).reshape(-1, 8); H = Hres[:n, 3].astype(float)
Qc, Ec, E0 = Qr[:, 2].astype(float), Er[:, 2].astype(float), Qr[:, 7].astype(float)
dif = Er[:, 7] != Qr[:, 7]; print('queries whose exact hit count differs between the Q and E1 tables: %d of %d (mean abs %.4f)' % (dif.sum(), n, np.abs(Er[:, 7] - Qr[:, 7]).mean()))
m = lambda x: x.mean()
print('| scene | H calls/q | Q | E1 | E0 | H/Q | Q/E1 | E1/E0 | H/E0 | H==Q per query | H<Q | H>Q | nodes Q | nodes E1 | Q/E1 nodes | leaf slots Q | E1 | Q/E1 | queries with extra node | extra leaf slot | extra call |')
print('|' + '---|' * 21)
print('| %s | %.3f | %.3f | %.3f | %.3f | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %.2f | %.2f | %.4f | %.2f | %.2f | %.4f | %.3f | %.3f | %.3f |' % (
    tag, m(H), m(Qc), m(Ec), m(E0), m(H) / m(Qc), m(Qc) / m(Ec), m(Ec) / max(m(E0), 1e-9), m(H) / max(m(E0), 1e-9), (H == Qc).mean(), (H < Qc).mean(), (H > Qc).mean(),
    m(Qr[:, 0]), m(Er[:, 0]), m(Qr[:, 0]) / m(Er[:, 0]), m(Qr[:, 1]), m(Er[:, 1]), m(Qr[:, 1]) / m(Er[:, 1]), (Qr[:, 0] > Er[:, 0]).mean(), (Qr[:, 1] > Er[:, 1]).mean(), (Qc > Ec).mean()))
print('closer: |H-Q| = %.4f calls/q, |H-E1| = %.4f; per query mean abs diff H-Q %.3f, H-E1 %.3f' % (abs(m(H) - m(Qc)), abs(m(H) - m(Ec)), np.abs(H - Qc).mean(), np.abs(H - Ec).mean()))
