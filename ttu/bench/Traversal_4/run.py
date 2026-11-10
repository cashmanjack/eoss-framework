"""Traversal_4: rays grazing triangle edges, vertices and box edges, hardware vs exact arithmetic.

    run.py BIN_S1_SMFILTER WORK      BIN: build_patch --stage 1 --clock --smfilter (needed by --sched)
    /usr/bin/python3 run.py --report WORK

Scene tri1 (a (-.5,-.5,0), b (.5,-.5,0), c (0,.5,0); its box edge y = -0.5 is also the edge ab) and grid:144 (triangles in the plane z = 0,
shared edges and vertices: watertightness). Rays go from z = -1 along +z through float32 points (x, y) placed ON an edge (exactly representable
points on the line) or within +-m float32 steps (ulps of the moved coordinate, m = 0..12 and 2^k for k = 4..20) of it, and at/around the
vertices. Exact answer: Fraction arithmetic on the float32 values (closed triangles; hit iff the point lies in some triangle). Compared with the
RT core's answer (first query is enough: one hit or none; --sched compact, resume 0) and OptiX's.
"""
import json, os, subprocess, sys
from fractions import Fraction as Fr
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.join(HERE, '..', '..', 'harness', 'setup3', 'run.py')
W = 16   # warps per block: 512 rays per launch
F = np.float32


def step(x, m):
    """x moved by m float32 steps (m may be negative)."""
    y = F(x)
    for _ in range(abs(int(m))):
        y = np.nextafter(y, F(np.inf) if m > 0 else F(-np.inf))
    return y


def tri1_points():
    a, b, c = (F(-.5), F(-.5)), (F(.5), F(-.5)), (F(0), F(.5))
    ms = list(range(-12, 13)) + [s * (1 << k) for k in range(4, 21) for s in (-1, 1)]
    pts = []   # (x, y, label)
    for x in (F(-.25), F(0), F(.25)):                                   # edge ab: y = -0.5, vary y; inside = larger y
        for m in ms: pts.append((x, step(F(-.5), m), 'ab', m))
    for x, y in ((F(-.375), F(-.25)), (F(-.25), F(0)), (F(-.125), F(.25))):   # edge ac: y = 2x + .5, vary x; inside = larger x
        for m in ms: pts.append((step(x, m), y, 'ac', m))
    for x, y in ((F(.125), F(.25)), (F(.25), F(0)), (F(.375), F(-.25))):      # edge bc: y = -2x + .5, vary x; inside = smaller x
        for m in ms: pts.append((step(x, -m), y, 'bc', m))
    for name, v in (('a', a), ('b', b), ('c', c)):                     # vertices and their eight neighbour directions
        for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)):
            for m in (1, 2, 4):
                if (dx, dy) == (0, 0) and m > 1: continue
                pts.append((step(v[0], dx * m), step(v[1], dy * m), 'v' + name, m * max(abs(dx), abs(dy))))
    return [a, b, c], pts


def inside(tri, x, y):
    (ax, ay), (bx, by), (cx, cy) = [(Fr(float(p[0])), Fr(float(p[1]))) for p in tri]
    x, y = Fr(float(x)), Fr(float(y))
    o = lambda px, py, qx, qy: (qx - px) * (y - py) - (qy - py) * (x - px)
    s = [o(ax, ay, bx, by), o(bx, by, cx, cy), o(cx, cy, ax, ay)]
    return all(v >= 0 for v in s) or all(v <= 0 for v in s)


def run(b1, work):
    os.makedirs(work, exist_ok=True)
    tri, pts = tri1_points()
    rays = np.zeros((len(pts), 8), F)
    for i, (x, y, _, _) in enumerate(pts): rays[i] = [x, y, -1, 0, 0, 0, 1, 1e16]
    n = (len(pts) + 32 * W - 1) // (32 * W) * 32 * W
    rays = np.concatenate([rays, np.tile(rays[:1], (n - len(pts), 1))])
    rays.tofile(os.path.join(work, 'rays_tri1.bin'))
    json.dump([(float(x), float(y), l, int(m)) for x, y, l, m in pts], open(os.path.join(work, 'pts_tri1.json'), 'w'))
    exec_(b1, os.path.join(work, 'rays_tri1.bin'), os.path.join(work, 'tri1'), 'tri1', work)
    # grid:144: every vertex of the triangle set, and the points one step off along x and y
    exec_(b1, 't2', os.path.join(work, 'gridtris'), 'grid:144', work, dump=os.path.join(work, 'grid.tris'), sched=False)
    T = np.fromfile(os.path.join(work, 'grid.tris'), F).reshape(-1, 3, 3)
    assert np.all(T[:, :, 2] == T[0, 0, 2]), 'grid triangles are not in one plane'
    z0 = float(T[0, 0, 2]); V = sorted({(float(p[0]), float(p[1])) for t in T for p in t})
    gp = [(F(x), F(y), 'gv', 0) for x, y in V]
    gp += [(step(F(x), 1), F(y), 'gvx+', 1) for x, y in V[:300]] + [(step(F(x), -1), F(y), 'gvx-', -1) for x, y in V[:300]]
    n = (len(gp) + 32 * W - 1) // (32 * W) * 32 * W
    r = np.zeros((n, 8), F)
    for i in range(n):
        x, y = (gp[i][0], gp[i][1]) if i < len(gp) else (gp[0][0], gp[0][1]); r[i] = [x, y, z0 - 1, 0, 0, 0, 1, 1e16]
    r.tofile(os.path.join(work, 'rays_grid.bin'))
    json.dump([(float(x), float(y), l, int(m)) for x, y, l, m in gp], open(os.path.join(work, 'pts_grid.json'), 'w'))
    exec_(b1, os.path.join(work, 'rays_grid.bin'), os.path.join(work, 'grid'), 'grid:144', work)


def exec_(b1, rays, out, scene, work, dump=None, sched=True):
    cmd = [sys.executable, HARNESS, b1, rays if rays.startswith('t') and not os.path.exists(rays) else 'file:' + rays, 'zero', out + '.json', '--scene', scene, '--warps', str(W)]
    cmd += ['--sched', 'compact', '--resume', '0'] if sched else ['--warps', '1']
    if dump: cmd += ['--dump-tris', dump]
    p = subprocess.run(cmd, capture_output=True, text=True)
    ok = p.returncode in (0, 1) and 'exit' in p.stdout and ' error' not in p.stdout.replace('no error', '')   # exit 1 = some rays still pending (a first query only)
    print(os.path.basename(out), 'OK' if ok else 'FAILED', [l for l in p.stdout.split('\n') if l.startswith('sched')], flush=True)
    if not ok: sys.exit(p.stdout[-2000:] + p.stderr[-2000:])
    if dump is None and sched is False: pass


def classify(work, name, tris):
    j = json.load(open(os.path.join(work, name + '.json'))); pts = json.load(open(os.path.join(work, 'pts_%s.json' % name.replace('tri1', 'tri1'))))
    res = []
    for i, (x, y, lab, m) in enumerate(pts):
        ex = any(inside(t, x, y) for t in tris)
        hw = j['ray_best'][i][0] != '0xffffffff'; ox = bool(j['ref'][i]['hit'])
        res.append((lab, m, ex, hw, ox))
    return res


def report(work):
    tri = [(( -.5, -.5), (.5, -.5), (0., .5))]
    for name, tris in (('tri1', tri),):
        res = classify(work, name, tri)
        print('== tri1: hardware (RT core) and OptiX vs exact arithmetic, by edge and distance in float32 steps (negative = outside the triangle, 0 = exactly on)')
        print('%-4s %6s | %5s | %-14s %-14s | exact hit?' % ('edge', 'steps', 'n', 'RT core hits', 'OptiX hits'))
        for lab in ('ab', 'ac', 'bc', 'va', 'vb', 'vc'):
            ms = sorted({m for l, m, *_ in res if l == lab})
            for m in ms:
                sel = [r for r in res if r[0] == lab and r[1] == m]
                if lab in ('ab', 'ac', 'bc') and abs(m) not in (0, 1, 2, 3, 4, 8, 12, 16, 256, 4096, 1048576): continue
                print('%-4s %6d | %5d | %5d of %-5d  %5d of %-5d | %s' % (lab, m, len(sel), sum(r[3] for r in sel), len(sel), sum(r[4] for r in sel), len(sel), sorted({r[2] for r in sel})))
        bad = [r for r in res if r[2] != r[3]]; badx = [r for r in res if r[2] != r[4]]
        print('tri1: %d points; RT core disagrees with exact on %d (%d false hits, %d false misses); OptiX on %d' % (len(res), len(bad), sum(1 for r in bad if r[3]), sum(1 for r in bad if r[2]), len(badx)))
    if os.path.exists(os.path.join(work, 'grid.json')):
        T = np.fromfile(os.path.join(work, 'grid.tris'), F).reshape(-1, 3, 3); gt = [tuple((float(p[0]), float(p[1])) for p in t) for t in T]
        res = classify(work, 'grid', gt)
        print('== grid:144 (%d triangles), shared vertices and one step off' % len(T))
        for lab in ('gv', 'gvx+', 'gvx-'):
            sel = [r for r in res if r[0] == lab]
            print('  %-5s n %4d | exact hit %4d | RT core hit %4d | OptiX hit %4d | RT core != exact %d | OptiX != exact %d' % (
                lab, len(sel), sum(r[2] for r in sel), sum(r[3] for r in sel), sum(r[4] for r in sel), sum(r[2] != r[3] for r in sel), sum(r[2] != r[4] for r in sel)))


if __name__ == '__main__':
    report(sys.argv[2]) if sys.argv[1] == '--report' else run(*sys.argv[1:3])
