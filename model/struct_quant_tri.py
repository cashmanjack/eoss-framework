"""Box quantization on triangles (predictions item 9: model/predictions_struct_quant.md). CPU only, no new hardware run.
usage: /usr/bin/python3 model/struct_quant_tri.py GAS TRIS RAYS WORK     (the F187 dumps: notes/tmaxreplay/soup_*.gas .tris .rays; needs /dev/shm/rp/assign and /dev/shm/rp/replay)
Q = decoded quantized boxes (make_table, ASSIGN_MODE=1: a triangle no slot contains goes into every slot it overlaps). E1 = every slot box replaced by the union over its triangles of (triangle box clipped to the quantized slot box), node slots the union of their children: the tightest box inside the quantized one that still holds the slot's triangle pieces (spatial split pieces are clipped, so the full triangle box would be too large)."""
import os, struct, subprocess, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
gasf, trisf, raysf, W = sys.argv[1:5]; os.makedirs(W, exist_ok=True)
env = dict(os.environ, ASSIGN='/dev/shm/rp/assign', ASSIGN_MODE='1')
r = subprocess.run([sys.executable, HERE + '/replay/make_table.py', gasf, trisf, W + '/soup.table'], env=env, check=True, capture_output=True, text=True); print(r.stdout.strip())
raw = open(W + '/soup.table', 'rb').read(); nn, nt, nl = struct.unpack_from('<3i', raw, 0)
rec = np.frombuffer(raw, np.dtype([('t', '<i4'), ('c', '<i4'), ('b', '<f4', 6)]), nn * 12, 12).copy().reshape(nn, 12)
off = 12 + nn * 12 * 32; ls = np.frombuffer(raw, np.int32, nl + 1, off); lt = np.frombuffer(raw, np.int32, int(ls[-1]), off + 4 * (nl + 1))
T = np.fromfile(trisf, np.float32).reshape(-1, 3, 3); tb = np.concatenate([T.min(1), T.max(1)], 1)
ex = np.zeros((nn, 12, 6), np.float32); has = np.zeros((nn, 12), bool); nb = np.zeros((nn, 6), np.float32); nh = np.zeros(nn, bool); q = rec['b']
for n in range(nn - 1, -1, -1):
    lo = np.full(3, np.inf); hi = np.full(3, -np.inf)
    for s in range(12):
        t = rec['t'][n, s]
        if t == 0: continue
        if t == 2:
            p = lt[ls[rec['c'][n, s]]:ls[rec['c'][n, s] + 1]]
            if len(p) == 0: continue
            lo_ = np.maximum(tb[p, :3], q[n, s, :3]); hi_ = np.minimum(tb[p, 3:], q[n, s, 3:]); ok = (lo_ <= hi_).all(1)
            if not ok.any(): continue
            b = np.concatenate([lo_[ok].min(0), hi_[ok].max(0)])
        else:
            c = rec['c'][n, s]
            if not nh[c]: continue
            b = nb[c]
        ex[n, s] = b; has[n, s] = True; lo = np.minimum(lo, b[:3]); hi = np.maximum(hi, b[3:])
    if has[n].any(): nb[n] = np.concatenate([lo, hi]); nh[n] = True
live = (rec['t'] > 0) & has
w = (q[..., 3:] - q[..., :3]); we = (ex[..., 3:] - ex[..., :3]); print('slots live %d of %d; mean extra width / exact width per axis %.4f' % (live.sum(), (rec['t'] > 0).sum(), np.mean(((w - we) / np.maximum(we, 1e-9))[live])))
rec2 = rec.copy(); rec2['b'] = np.where(has[..., None], ex, q)
with open(W + '/soup.tableE1', 'wb') as f: f.write(raw[:12]); f.write(rec2.tobytes()); f.write(raw[off:])
n = len(np.fromfile(raysf, np.float32).reshape(-1, 8))
def run(tab, out): subprocess.run(['/dev/shm/rp/replay', tab, raysf, str(n), '0', out], check=True); return np.fromfile(out, np.int32).reshape(n, 8)
Q = run(W + '/soup.table', W + '/soup.repQ'); E = run(W + '/soup.tableE1', W + '/soup.repE1')
wn, wl = 2.2, 9.9
for nm, o in (('Q', Q), ('E1', E)):
    print('%s nodes %.3f leaf slots %.3f tris %.3f hit %.4f | weighted requests %.2f (measured 80.8) ' % (nm, o[:, 0].mean(), o[:, 1].mean(), o[:, 2].mean(), (o[:, 5] >= 0).mean(), wn * o[:, 0].mean() + wl * o[:, 1].mean()))
print('Q/E1 nodes %.4f, leaf slots %.4f, weighted requests %.4f; queries with extra node %.3f, extra leaf slot %.3f; hit decision differs %d, primitive differs %d of %d' % (Q[:, 0].mean() / E[:, 0].mean(), Q[:, 1].mean() / E[:, 1].mean(),
      (wn * Q[:, 0].mean() + wl * Q[:, 1].mean()) / (wn * E[:, 0].mean() + wl * E[:, 1].mean()), (Q[:, 0] > E[:, 0]).mean(), (Q[:, 1] > E[:, 1]).mean(), ((Q[:, 5] >= 0) != (E[:, 5] >= 0)).sum(), (Q[:, 5] != E[:, 5]).sum(), n))
