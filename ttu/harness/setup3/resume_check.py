"""Check resumed queries against a brute force CPU reference (float64, every triangle).

    resume_check.py OUT.json TRIS.bin

TRIS.bin: setup3 --dump-tris (9 float32 per triangle). Checks the first 32 lanes (the
lane rays recorded in OUT.json): final hit (closest over all queries) vs the CPU closest hit.
"""
import json, struct, sys
import numpy as np


def closest(tris, o, d, tmax=np.inf):
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    e1, e2 = b - a, c - a
    p = np.cross(d, e2); det = (e1 * p).sum(1)
    ok = np.abs(det) > 1e-12
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    s = o - a; u = (s * p).sum(1) * inv
    q = np.cross(s, e1); v = (q * d).sum(1) * inv; t = (q * e2).sum(1) * inv
    hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t >= 0) & (t <= tmax)
    if not hit.any():
        return None, None
    k = np.argmin(np.where(hit, t, np.inf))
    return float(t[k]), int(k)


def main(path, tpath):
    j = json.load(open(path))
    tris = np.fromfile(tpath, dtype=np.float32).reshape(-1, 3, 3).astype(np.float64)
    f = lambda h: struct.unpack('<f', struct.pack('<I', int(h, 16)))[0]
    bad = []; rows = []
    for i, r in enumerate(j['lane_rays']):
        o = np.array(r[:3]); d = np.array(r[3:6])
        t_cpu, p_cpu = closest(tris, o, d)
        bt, bp = j['lane_best'][i]
        q = j['lane_queries'][i]
        if t_cpu is None:
            good = bt == '0xffffffff'
        else:
            good = bt != '0xffffffff' and abs(f(bt) - t_cpu) <= 1e-5 * max(1.0, t_cpu) and bp == p_cpu
        rows.append((i, q, t_cpu, p_cpu, None if bt == '0xffffffff' else f(bt), bp, good))
        if not good: bad.append(i)
    qs = [abs(x) for x in j['lane_queries'][:32]]
    print('%s: %d lanes, CPU closest hit matched in %d, mismatches %s, unfinished lanes %s' % (
        path.split('/')[-1], len(rows), sum(r[-1] for r in rows), bad or 'none', [i for i, x in enumerate(j['lane_queries'][:32]) if x < 0] or 'none'))
    print('  queries per lane (first warp): %s' % qs)
    return not bad


if __name__ == '__main__':
    sys.exit(0 if main(sys.argv[1], sys.argv[2]) else 1)
