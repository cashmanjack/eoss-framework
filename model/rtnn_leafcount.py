"""F167: recheck of the RTNN replays (F125, F127) with the TRUE primitives per leaf slot (low 3 bits of the slot type byte, 0 = 8) instead of box matching.
    /usr/bin/python3 model/rtnn_leafcount.py DUMP_ROOT WORK      DUMP_ROOT/<config>/{gas0.bin, points.bin, aabb_radius0.txt, rays0.bin, cand.bin} from the patched RTNN (triton, notes/q11/rtnn_patches.diff plus the RTNN_DRAM_GB line)
Writes WORK/<config>.rep (box matched, as before) and .rep2 (true counts); prints per configuration the mean hardware calls (recorder), both replays, lane efficiencies."""
import os, struct, subprocess, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../ttu/bench/BVH_5'); import gas
R_, W = sys.argv[1:3]; os.makedirs(W, exist_ok=True)
def eff(c): w = c[: len(c) // 32 * 32].reshape(-1, 32); return w.sum() / (32 * w.max(1).sum())
def table2(gf, tab, out):
    d = open(gf, 'rb').read(); root = struct.unpack_from('<Q', d, 0x20)[0]; order = [root]; i = 0; types = []
    while i < len(order):
        for s, k, ch, box in gas.slots(d, order[i]):
            if k == 'node': order.append(ch)
            else: types.append(ch)
        i += 1
    cnt = np.array(types) & 7; cnt = np.where(cnt == 0, 8, cnt)
    raw = open(tab, 'rb').read(); nn, nt, nl = struct.unpack_from('<3i', raw, 0); assert nl == len(cnt), (nl, len(cnt))
    off = 12 + nn * 12 * 32; ls = np.frombuffer(raw, np.int32, nl + 1, off); lt_off = off + 4 * (nl + 1); nlt = ls[-1]
    ls2 = np.concatenate([[0], np.cumsum(cnt)]).astype(np.int32); tail = np.frombuffer(raw, np.uint8, offset=lt_off + 4 * nlt)
    with open(out, 'wb') as o: o.write(raw[:off]); o.write(ls2.tobytes()); o.write(np.zeros(ls2[-1], np.int32).tobytes()); o.write(tail.tobytes())
    return int(cnt.sum()), nt, nl, int((cnt == 8).sum())
print('| config | hardware calls / query (recorder) | box matched replay x1.006 | true count replay x1.006 | box matched error | true count error | exact box hits | per query replays equal | lane eff (recorder) | lane eff box matched | lane eff true | sum of slot counts / primitives | slots with count 8 |'); print('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
for name in sorted(os.listdir(R_)):
    d = '%s/%s' % (R_, name); r = float(open(d + '/aabb_radius0.txt').read().split()[0]); P = np.fromfile(d + '/points.bin', np.float32).reshape(-1, 3)
    np.stack([P - np.float32(r), P + np.float32(r), P - np.float32(r)], 1).astype(np.float32).tofile('%s/%s.tris' % (W, name))
    subprocess.run([sys.executable, HERE + '/replay/make_table.py', d + '/gas0.bin', '%s/%s.tris' % (W, name), '%s/%s.table' % (W, name)], env=dict(os.environ, ASSIGN='/dev/shm/rp/assign'), check=True, capture_output=True)
    tot, nt, nl, n8 = table2(d + '/gas0.bin', '%s/%s.table' % (W, name), '%s/%s.table2' % (W, name)); nr = os.path.getsize(d + '/rays0.bin') // 32
    for t, o in (('table', 'rep'), ('table2', 'rep2')): subprocess.run(['/dev/shm/rp/replay_aabb', '%s/%s.%s' % (W, name, t), d + '/rays0.bin', str(nr), '0', '%s/%s.%s' % (W, name, o)], check=True)
    m = np.fromfile(d + '/cand.bin', np.uint32).astype(float); a = np.fromfile('%s/%s.rep' % (W, name), np.int32).reshape(-1, 8); b = np.fromfile('%s/%s.rep2' % (W, name), np.int32).reshape(-1, 8); ca, cb = a[:, 2] * 1.006, b[:, 2] * 1.006
    print('| %s | %.2f | %.2f | %.2f | %+.1f%% | %+.1f%% | %.2f | %.4f | %.3f | %.3f | %.3f | %d / %d | %d |' % (name, m.mean(), ca.mean(), cb.mean(), 100 * (ca.mean() / m.mean() - 1), 100 * (cb.mean() / m.mean() - 1), a[:, 7].mean(), (a[:, 2] == b[:, 2]).mean(), eff(m), eff(ca), eff(cb), tot, nt, n8), flush=True)
