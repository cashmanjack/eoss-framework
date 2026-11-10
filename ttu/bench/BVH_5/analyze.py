"""BVH_5 analysis: did a traversal started at an internal node X return the closest hit among
the triangles inside X's box (and miss otherwise)?

    analyze.py OUT.json GAS.bin TRIS.bin [--root-ref]

X's box: decoded from its parent's slot (gas.py). A triangle is under X if its bounding box
lies inside X's box. CPU reference: float64 closest hit over those triangles only.
--root-ref: also compare every lane with the OptiX reference (closest over all triangles).
"""
import json, struct, sys
import numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0])
sys.path.insert(0, __file__.rsplit('/', 1)[0] + '/../../harness/setup3')
import gas
from resume_check import closest

f32 = lambda h: struct.unpack('<f', struct.pack('<I', int(h, 16)))[0]


def crosses(o, d, lo, hi, eps=1e-6):
    """Does the ray (t >= 0) touch the box [lo, hi]? Slab test; an axis with d = 0 only checks the origin."""
    lo = lo - eps; hi = hi + eps
    with np.errstate(divide='ignore', invalid='ignore'):
        t0 = (lo - o) / d; t1 = (hi - o) / d
    tn = np.minimum(t0, t1); tf = np.maximum(t0, t1)
    inside = (o >= lo) & (o <= hi)
    tn = np.where(d == 0, np.where(inside, -np.inf, np.inf), tn)
    tf = np.where(d == 0, np.where(inside, np.inf, -np.inf), tf)
    return max(tn.max(), 0.0) <= tf.min()


def main(path, gpath, tpath, root_ref=False):
    j = json.load(open(path)); G = open(gpath, 'rb').read()
    tris = np.fromfile(tpath, dtype=np.float32).reshape(-1, 3, 3).astype(np.float64)
    W = gas.walk(G); X = int(j['start_off'], 16)
    assert X in W, 'start offset is not an internal node of this GAS'
    node = W[X]
    want = j['start_path'] or 'root'
    assert want == 'root' and node['path'] == () or tuple(int(x) for x in want.split(',')) == node['path'], 'path mismatch'
    if node['box'] is None:
        under = np.ones(len(tris), bool)
    else:
        lo = np.array([b[0] for b in node['box']]); hi = np.array([b[1] for b in node['box']])
        tol = 1e-5
        under = ((tris.min(1) >= lo - tol) & (tris.max(1) <= hi + tol)).all(1)
    sub = tris[under]; idx = np.nonzero(under)[0]
    R0 = j['in'][0]
    word_ok = R0[70] == (int(j['start'], 16) >> 7) & 0xffffffff
    rows = dict(hit_ok=0, miss_ok=0, bad=[], unfinished=0, ref_bad=[], exit_ok=0, exit_n=0)
    q = []
    for i, r in enumerate(j['lane_rays']):
        o = np.array(r[:3], float); d = np.array(r[3:6], float)
        t, k = closest(sub, o, d) if len(sub) else (None, None)
        bt, bp = j['lane_best'][i]; nq = j['lane_queries'][i]; q.append(abs(nq))
        if nq < 0: rows['unfinished'] += 1
        if t is None:
            ok = bt == '0xffffffff'; rows['miss_ok'] += ok
            if node['box'] is not None and crosses(o, d, lo, hi) and closest(tris, o, d)[0] is not None:
                rows['exit_n'] += 1; rows['exit_ok'] += ok   # ray crosses X's box, nothing under X, a hit elsewhere
        else:
            ok = bt != '0xffffffff' and abs(f32(bt) - t) <= 1e-5 * max(1.0, t) and bp == int(idx[k]); rows['hit_ok'] += ok
        if not ok: rows['bad'].append((i, None if t is None else (round(t, 6), int(idx[k])), bt if bt == '0xffffffff' else (round(f32(bt), 6), bp)))
        if root_ref:
            ref = j['ref'][i]
            rok = (ref['hit'] == 0 and bt == '0xffffffff') or (ref['hit'] and bt != '0xffffffff' and ref['prim'] == bp and abs(f32(ref['t']) - f32(bt)) <= 1e-6 * max(1, f32(bt)))
            if not rok: rows['ref_bad'].append(i)
    n = len(j['lane_rays'])
    print('%-28s start %-6s depth %d (%d tris under it) | %d lanes: %d hits ok, %d misses ok (%d of %d crossing X and hitting outside it), %d wrong %s | queries %s | 0x180 word %s%s' % (
        path.split('/')[-1], want, node['depth'], int(under.sum()), n, rows['hit_ok'], rows['miss_ok'], rows['exit_ok'], rows['exit_n'], len(rows['bad']), rows['bad'][:4],
        dict(sorted({x: q.count(x) for x in set(q)}.items())), 'ok' if word_ok else 'WRONG',
        (' | vs OptiX: %d wrong' % len(rows['ref_bad'])) if root_ref else ''))
    return not rows['bad'] and word_ok and not rows['ref_bad']


if __name__ == '__main__':
    a = [x for x in sys.argv[1:] if not x.startswith('--')]
    sys.exit(0 if main(*a, root_ref='--root-ref' in sys.argv) else 1)
