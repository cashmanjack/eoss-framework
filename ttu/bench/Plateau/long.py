"""Plateau checks with long launches (persistent loop kernel, build_patch --outer R; see Throughput_6L/run.py).

    long.py WORK a SCENE            (a) RT only packed vs spread: 142 blocks launched, n active SMs (one per TPC), R = 100 rounds per lane
    long.py WORK b SCENE G R        (b) Memory_4 grouped (oracle key) vs interleaved at G blocks, R rounds per lane

(a) sets as in rt_pack.py: spread = TPCs 0..n-1 (round robin over the 12 GPCs), packed = TPCs GPC by GPC. Rates count only the rays the active blocks trace.
(b) the same random rays N = G x 896 x R in two orders (grouped: sorted by the slot path of the deepest node whose box holds the hit triangle's box, or the origin for a miss;
    interleaved: random permutation, seed 7); every lane is compared with OptiX, 200 with the CPU.
"""
import os, random, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'Throughput_6L')); sys.path.insert(0, os.path.join(HERE, '..', 'Memory_4'))
import long_run as T  # noqa: E402
import prep  # noqa: E402


def sets(n):
    by_gpc = [[t for t in range(71) if t % 12 == g] for g in range(12)]
    packed = [t for g in range(12) for t in by_gpc[g]][:n]
    return {'spread': [2 * t for t in range(n)], 'packed': [2 * t for t in packed]}


def part_a(work, scene, R=100):
    for n in [6, 12, 24, 36, 60]:
        for kind, sm in sets(n).items():
            T.run(work, scene, 'rnd', 142, R, tag='pk_%s_%s_%d' % (scene.replace(':', ''), kind, n), smset=sm)


def make_ordered(work, scene, G, R, tagp):
    """rays in two orders; returns (grouped file, interleaved file). Uses a stage 0 loop build only to get a checked binary for the OptiX reference."""
    n = G * T.LANES * R
    base = '/dev/shm/t8/pb_base.bin'; prep.make_rays(n).tofile(base)
    b0 = '/dev/shm/t8/s0_plain'
    T.build(0, 1, 1, b0)
    ref = '/dev/shm/t8/pb_ref'
    p = subprocess.run([sys.executable, T.os.path.join(T.S3, 'run.py'), b0, 'file:' + base, 'zero', ref + '.json', '--scene', scene, '--warps', '16', '--no-ttu', '--quick',
                        '--dump-gas', ref + '.gas', '--dump-tris', ref + '.tris', '--dump-ref', ref + '.ref'], capture_output=True, text=True)
    if p.returncode: sys.exit(p.stdout[-2000:] + p.stderr[-1000:])
    r = np.fromfile(ref + '.ref', dtype=np.uint32).reshape(-1, 8); assert len(r) == n
    rays = np.fromfile(base, dtype=np.float32).reshape(-1, 8)
    tris = np.fromfile(ref + '.tris', np.float32).reshape(-1, 3, 3)
    W, kids, root = prep.tree(open(ref + '.gas', 'rb').read())
    cache = {}; keys = [None] * n
    for i in range(n):
        if r[i, 0]:
            pr = int(r[i, 2])
            if pr not in cache:
                t = tris[pr]; cache[pr] = prep.key_of(W, kids, root, t.min(0), t.max(0))
            keys[i] = cache[pr]
        else:
            q = rays[i, 0:3]; keys[i] = prep.key_of(W, kids, root, q, q)
    grouped = sorted(range(n), key=lambda i: (keys[i], i))
    inter = list(range(n)); random.Random(7).shuffle(inter)
    out = []
    for name, perm in (('grouped', grouped), ('interleaved', inter)):
        f = '/dev/shm/t8/pb_%s.bin' % name; rays[np.array(perm)].tofile(f); out.append(f)
    for f in (base, ref + '.ref', ref + '.gas', ref + '.json', b0, b0 + '.checked', b0 + '.layout.json'):
        if os.path.exists(f): os.remove(f)
    if os.path.exists(ref + '.tris'): os.remove(ref + '.tris')
    return out


def part_b(work, scene, G, R):
    g, i = make_ordered(work, scene, G, R, scene)
    for name, f in (('interleaved', i), ('grouped', g)):
        T.run(work, scene, 'rnd', G, R, tag='pb_%s_%s_g%d' % (scene.replace(':', ''), name, G), rays_file=f)
        os.remove(f) if name == 'grouped' else None
    if os.path.exists(i): os.remove(i)


if __name__ == '__main__':
    work, part, scene = sys.argv[1:4]
    if part == 'a': part_a(work, scene)
    else: part_b(work, scene, int(sys.argv[4]), int(sys.argv[5]))
