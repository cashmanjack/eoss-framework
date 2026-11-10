"""BVH_5: start traversal at an internal node. Reruns the 2026-11-10 series.

    run.py BIN OUTDIR       BIN = a stage 1 setup3 binary that passed check_seq.py (--clock build)

Every run goes through ttu/harness/setup3/run.py (checked sha256, idle GPU 1, 30 s timeout).
No new TTU sequence: only the 0x180 input word changes (--start-path). Results: OUTDIR/*.json,
with the GAS and triangle dumps of the same process next to each. Then:
    analyze.py OUT.json OUT.gas OUT.tris     correctness vs a CPU reference limited to the subtree
    latency.py OUT.json ...                  first query latency per start node
"""
import os, random, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.join(HERE, '..', '..', 'harness', 'setup3', 'run.py')
sys.path.insert(0, HERE)
import gas  # noqa: E402


def grid_under(box, n):
    k = int(-(-n ** 0.5 // 1)); c = 2 / k
    return [i for i in range(n) if box[0][0] - 1e-5 <= -1 + (i % k) * c + 0.1 * c and -1 + (i % k) * c + 0.9 * c <= box[0][1] + 1e-5
            and box[1][0] - 1e-5 <= -1 + (i // k) * c + 0.1 * c and -1 + (i // k) * c + 0.9 * c <= box[1][1] + 1e-5]


def run(binp, out, name, scene, pairs, warps, path, extra=()):
    o = os.path.join(out, name)
    cmd = [sys.executable, HARNESS, binp, 'pairs:' + pairs, 'zero', o + '.json', '--scene', scene, '--warps', str(warps),
           '--resume', '8', '--start-path', path, '--dump-gas', o + '.gas', '--dump-tris', o + '.tris'] + list(extra)
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(name, 'exit', r.returncode)
    if r.returncode:
        sys.exit(r.stdout + r.stderr)
    return o


def main(binp, out):
    os.makedirs(out, exist_ok=True)
    P = lambda n, lines: (open(os.path.join(out, n), 'w').write(''.join('%d %d\n' % ab for ab in lines)), os.path.join(out, n))[1]
    r5, r7, r11 = random.Random(5), random.Random(7), random.Random(11)
    # A: grid:144, straight rays onto every triangle, root and each depth 1 child; smallest launch first
    pa = P('p144_all.txt', [(k, k) for k in range(144)])
    run(binp, out, 'a1_3', 'grid:144', pa, 1, '3')
    for p in ['root'] + [str(s) for s in range(12)]:
        run(binp, out, 'a5_' + p, 'grid:144', pa, 5, p)
    # B: grid:144, random oblique rays
    pb = P('p144_rand.txt', [(r5.randrange(144), r5.randrange(144)) for _ in range(160)])
    for p in ['root', '0', '6', '11']:
        run(binp, out, 'b5_' + p, 'grid:144', pb, 5, p)
    # C: four layers (rays can cross a subtree's box and leave it)
    pc = P('pl_rand.txt', [(r7.randrange(144), r7.randrange(144)) for _ in range(160)])
    for p in ['root'] + [str(s) for s in range(12)]:
        run(binp, out, 'c5_' + p, 'layers:36:4:0.3', pc, 5, p)
    # D: grid:2500, depth 1 and 2 starts (slot paths are stable across processes; boxes from a reference dump)
    ref = run(binp, out, 'd_ref', 'grid:2500', P('p_one.txt', [(0, 0)]), 1, 'root')
    W = gas.walk(open(ref + '.gas', 'rb').read())
    und = grid_under([n['box'] for n in W.values() if n['path'] == (0,)][0], 2500)
    pd = P('p2500.txt', [(i, i) for i in und] + [(r11.randrange(2500), r11.randrange(2500)) for _ in range(512 - len(und))])
    for p in ['root', '0', '5', '0,0', '0,6', '5,6']:
        run(binp, out, 'd5_' + p, 'grid:2500', pd, 16, p)
    # E: latency, one warp, one ray, warm (across launches) and after a 256 MB flush
    t00 = grid_under([n['box'] for n in W.values() if n['path'] == (0, 0)][0], 2500)[0]
    for fl in ('0', '256'):
        for n, t, paths in (('144', 39, ['root', '3']), ('2500', t00, ['root', '0', '0,0'])):
            pe = P('l%s.txt' % n, [(t, t)])
            for p in paths:
                run(binp, out, 'lat_%s_%s_%s' % (n, p, fl), 'grid:' + n, pe, 1, p, ['--repeat', '31', '--flush', fl])


if __name__ == '__main__':
    main(*sys.argv[1:3])
