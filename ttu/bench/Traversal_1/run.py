"""Traversal_1: visit order and early exit.

    run.py BIN_S1 WORK            BIN_S1: checked stage 1 build with --clock
    /usr/bin/python3 run.py --report WORK

(a) Order along the ray: stack:L (L copies of one triangle at z = 0.01 k, all on the ray) traced from below (+z) and from above (-z):
    which primitive does the first query return? (nearest to the ray origin = by distance along the ray, not by memory order)
(b) Closest hit in the near vs far child: clus:16 (a large slanted triangle A whose box is entered first but is hit at z 4.5, and 15 small
    triangles on the ray at z < 4) traced from below (A's box first, closest hit later) and from above (A is both first and closest).
(c) Early exit by tmax: par:256:s (256 triangles stacked along z, the ray passes every box, no hit) with tmax cutting the ray after a fraction
    of the stack: if boxes beyond tmax are culled the latency falls with the fraction.
One warp, every lane the same ray, --resume 12, 5 launches (first dropped).
"""
import json, os, struct, statistics as st, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.join(HERE, '..', '..', 'harness', 'setup3', 'run.py')
FR = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def go(b1, rays, out, scene, extra=()):
    p = subprocess.run([sys.executable, HARNESS, b1, rays, 'zero', out + '.json', '--scene', scene, '--warps', '1', '--repeat', '5', '--resume', '12'] + list(extra),
                       capture_output=True, text=True)
    ok = p.returncode == 0 and 'exit 0' in p.stdout and ' error' not in p.stdout.replace('no error', '')
    print(os.path.basename(out), 'OK' if ok else 'FAILED', flush=True)
    if not ok:
        sys.exit(p.stdout[-2000:] + p.stderr[-2000:])


def rayfile(path, o, d, tmax):
    np.tile(np.array([[*o, 0.0, *d, tmax]], np.float32), (32, 1)).tofile(path)
    return 'file:' + path


def run(b1, work):
    os.makedirs(work, exist_ok=True)
    for L in (8, 64):
        go(b1, 'hit', os.path.join(work, 'a_stack%d_up' % L), 'stack:%d' % L)
        go(b1, 'back', os.path.join(work, 'a_stack%d_down' % L), 'stack:%d' % L)
    go(b1, rayfile(os.path.join(work, 'r.bin'), (0, 0, -1), (0, 0, 1), 1e16), os.path.join(work, 'b_clus16_up'), 'clus:16')
    go(b1, rayfile(os.path.join(work, 'r.bin'), (0, 0, 12), (0, 0, -1), 1e16), os.path.join(work, 'b_clus16_down'), 'clus:16')
    # par:256:s extent: dump the triangles once
    go(b1, 't2', os.path.join(work, 'c_dump'), 'par:256:s', ['--dump-tris', os.path.join(work, 'c.tris')])
    z = np.fromfile(os.path.join(work, 'c.tris'), np.float32).reshape(-1, 3, 3)[:, :, 2]; z0, z1 = float(z.min()), float(z.max())
    json.dump(dict(z0=z0, z1=z1), open(os.path.join(work, 'c_extent.json'), 'w'))
    for f in FR:
        tmax = (z0 + 2.0) + f * (z1 - z0)   # origin z = -2
        go(b1, rayfile(os.path.join(work, 'r.bin'), (0, 0, -2), (0, 0, 1), tmax), os.path.join(work, 'c_par256_%g' % f), 'par:256:s')
    os.remove(os.path.join(work, 'r.bin'))


def report(work):
    fl = lambda u: struct.unpack('<f', struct.pack('<I', u))[0]
    ld = lambda n: json.load(open(os.path.join(work, n + '.json')))
    print('(a)(b) first query result, queries per ray, pending entries after the first query')
    for n in ('a_stack8_up', 'a_stack8_down', 'a_stack64_up', 'a_stack64_down', 'b_clus16_up', 'b_clus16_down'):
        j = ld(n); wo = j['warp_out'][0]
        print('  %-16s first query: hit %d prim %d t %.4f | queries %d | final best prim %s | pending pairs after q1 %d' % (
            n, wo[3], wo[44] & 0x1fffffff, abs(fl(wo[45])), j['lane_queries'][0], j['lane_best'][0][1], j['lane_pending_pairs_first'][0]))
    ex = json.load(open(os.path.join(work, 'c_extent.json')))
    print('(c) par:256:s, z extent %.4f..%.4f, latency (cycles) of one query vs the fraction of the stack the ray reaches' % (ex['z0'], ex['z1']))
    base = None
    for f in FR:
        j = ld('c_par256_%g' % f); lat = st.median(l['warp_clk'][0][1] - l['warp_clk'][0][0] for l in j['launches'][1:])
        base = lat if f == 1.0 else base
        print('  fraction %.2f: latency %8.0f   queries %d' % (f, lat, j['lane_queries'][0]))
    print('  (full ray = last row)')


if __name__ == '__main__':
    report(sys.argv[2]) if sys.argv[1] == '--report' else run(*sys.argv[1:3])
