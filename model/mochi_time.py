"""Mochi time gap (F168): span model with measured callback cost. Form: model/FORM_mochi_time.md (committed before the probe run). CPU only.
    /usr/bin/python3 model/mochi_time.py DUMP_C REC_C WORK        (DUMP_C gas_fNN, particles_fNN; REC_C launch_ns.bin, cb_full_fNN.bin, cb_small{0,1,2}_fNN.bin of run C; needs /dev/shm/rp binaries)"""
import os, struct, subprocess, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../ttu/bench/BVH_5'); import gas
D, REC, W = sys.argv[1:4]; os.makedirs(W, exist_ok=True)
SM, CLK, NS = 142, 2.5e9, 100000
ns = np.fromfile(REC + '/launch_ns.bin', np.uint64).astype(float).reshape(-1, 400) / 1e6
def replay(f):
    P = np.fromfile('%s/particles_f%02d.bin' % (D, f), np.float32).reshape(-1, 8); c, r = P[:, :3], P[:, 6:7]
    np.stack([c - 2 * r, c + 2 * r, c - 2 * r], 1).astype(np.float32).tofile('%s/f%02d.tris' % (W, f)); gf = '%s/gas_f%02d.bin' % (D, f)
    subprocess.run([sys.executable, HERE + '/replay/make_table.py', gf, '%s/f%02d.tris' % (W, f), '%s/f%02d.table' % (W, f)], env=dict(os.environ, ASSIGN='/dev/shm/rp/assign'), check=True, capture_output=True)
    R = np.zeros((len(P), 8), np.float32); R[:, :3] = c; R[:, 5] = 1; R[:, 7] = 1e-16; R.tofile('%s/f%02d.rays' % (W, f))
    d = open(gf, 'rb').read(); root = struct.unpack_from('<Q', d, 0x20)[0]; order = [root]; i = 0; types = []
    while i < len(order):
        for s, k, ch, box in gas.slots(d, order[i]):
            if k == 'node': order.append(ch)
            else: types.append(ch)
        i += 1
    cnt = np.array(types) & 7; cnt = np.where(cnt == 0, 8, cnt)
    raw = open('%s/f%02d.table' % (W, f), 'rb').read(); nn, nt, nl = struct.unpack_from('<3i', raw, 0); off = 12 + nn * 12 * 32; ls = np.frombuffer(raw, np.int32, nl + 1, off); lt_off = off + 4 * (nl + 1); nlt = ls[-1]
    ls2 = np.concatenate([[0], np.cumsum(cnt)]).astype(np.int32); tail = np.frombuffer(raw, np.uint8, offset=lt_off + 4 * nlt)
    with open('%s/f%02d.table2' % (W, f), 'wb') as o: o.write(raw[:off]); o.write(ls2.tobytes()); o.write(np.zeros(ls2[-1], np.int32).tobytes()); o.write(tail.tobytes())
    subprocess.run(['/dev/shm/rp/replay_aabb', '%s/f%02d.table2' % (W, f), '%s/f%02d.rays' % (W, f), str(len(P)), '0', '%s/f%02d.rep2' % (W, f)], check=True)
    return np.fromfile('%s/f%02d.rep2' % (W, f), np.int32).reshape(-1, 8)
print('| frame | span measured ms (median of 400) | span of the probed launch | probes off span (run B) | warp calls: probe / replay | L_ld cycles per call | L_unl | raygen cycles per warp loaded / small launch | W implied | callbacks / traversal / other share of warp time | T_rt ms | T_cb ms (L_ld) | T_hand ms | T_cb ms (L_unl) | model (L_ld) ms | model (L_unl) ms | log error (L_unl) |')
print('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
B = [0.1055, 0.0993, 0.0993, 0.0993, 0.0993, 0.0993, 0.1126, 0.1393, 0.1812, 0.2386]
for f in range(10):
    rep = replay(f); cw = rep[:, 2] * 1.006; Nrep = cw[: NS // 32 * 32].reshape(-1, 32).max(1).sum(); q = rep[:, 2].mean() + 1
    full = np.fromfile('%s/cb_full_f%02d.bin' % (REC, f), np.uint64).reshape(-1, 4)[:3125].astype(float)
    sm = [np.fromfile('%s/cb_small%d_f%02d.bin' % (REC, k, f), np.uint64) for k in range(3)]; Lu = np.mean([s[:-2].reshape(-1, 4)[:142, 0].sum() / s[:-2].reshape(-1, 4)[:142, 1].sum() for s in sm]); rg_small = np.mean([s[:-2].reshape(-1, 4)[:142, 2].mean() for s in sm])
    span = np.median(ns[f]) ; last = ns[f][-1]; cyc = span * 1e-3 * CLK
    Nprobe = full[:, 1].sum(); Lld = full[:, 0].sum() / Nprobe; Wimp = full[:, 2].sum() / (SM * cyc)
    cb, tr, rg = full[:, 0].sum(), full[:, 3].sum(), full[:, 2].sum(); sh = (cb / rg, (tr - cb) / rg, (rg - tr) / rg)
    Trt = (110 + 4.9 * q) * NS / (SM * CLK) * 1e3; Tcb = Lld * Nprobe / (SM * Wimp) / CLK * 1e3; Th = 875 * Nprobe / (SM * Wimp) / CLK * 1e3; Tcbu = Lu * Nprobe / (SM * Wimp) / CLK * 1e3
    m1 = Trt + Tcb + Th; m2 = Trt + Tcbu + Th
    print('| %d | %.4f | %.4f | %.4f | %d / %d | %.0f | %.0f | %.0f / %.0f | %.2f | %.2f / %.2f / %.2f | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %+.2f |' % (f, span, last, B[f], Nprobe, Nrep, Lld, Lu, rg / 3125, rg_small, Wimp, sh[0], sh[1], sh[2], Trt, Tcb, Th, Tcbu, m1, m2, np.log(m2 / span)), flush=True)
