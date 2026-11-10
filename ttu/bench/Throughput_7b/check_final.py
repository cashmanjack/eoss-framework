"""Check final results of a build_patch --loop run against a brute force CPU closest hit (float64).

    check_final.py FINAL.bin RAYS.bin TRIS.bin K [NSAMPLE]

FINAL.bin: setup3 --dump-final (14 u32 per lane: R3 R32 R33 R44 R45 R56..R61 R70 R71 rounds_left).
RAYS.bin: 8 float32 per ray (o, tmin, d, tmax). TRIS.bin: setup3 --dump-tris (9 float32 per triangle).
A lane is finished if rounds_left > 0 (the loop ended by itself, not by the round limit K).
Checks NSAMPLE lanes spread over the launch (all lanes if the launch is smaller); the host's OptiX
reference covers lane 0..31 of the block separately (the setup3 JSON).
"""
import struct, sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'harness', 'setup3'))
from resume_check import closest

f = lambda u: struct.unpack('<f', struct.pack('<I', int(u)))[0]


def main(fin, frays, ftris, K, ns=64):
    a = np.fromfile(fin, dtype=np.uint32).reshape(-1, 14)
    rays = np.fromfile(frays, dtype=np.float32).reshape(-1, 8)[:len(a)]
    tris = np.fromfile(ftris, dtype=np.float32).reshape(-1, 3, 3).astype(np.float64)
    rounds = K - a[:, 13].astype(np.int64)
    fin_ok = a[:, 13] > 0
    idx = np.arange(len(a)) if len(a) <= ns else np.unique(np.linspace(0, len(a) - 1, ns).astype(int))
    bad = []
    for i in idx:
        o = rays[i, :3].astype(np.float64); d = rays[i, 4:7].astype(np.float64)
        t, p = closest(tris, o, d)
        hit = a[i, 0] != 0
        if t is None:
            good = not hit
        else:
            good = hit and abs(abs(f(a[i, 4])) - t) <= 1e-5 * max(1.0, t) and (a[i, 3] & 0x1fffffff) == p
        if not good:
            bad.append((int(i), t, p, int(hit), abs(f(a[i, 4])), int(a[i, 3] & 0x1fffffff)))
    print('lanes %d, unfinished (hit the round limit K=%d): %d, mean rounds %.2f, max %d | checked %d vs CPU, wrong %d'
          % (len(a), K, int((~fin_ok).sum()), rounds.mean(), rounds.max(), len(idx), len(bad)))
    for b in bad[:8]:
        print('  wrong lane', b)
    return not bad and fin_ok.all()


if __name__ == '__main__':
    sys.exit(0 if main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]) if len(sys.argv) > 5 else 64) else 1)
