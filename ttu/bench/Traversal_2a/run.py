"""Traversal_2a: when does a query return without a hit? Full traversal scenes (setup3 scene par:N:s|o):
N triangles parallel to a +z ray at x = y = 0 and 1e-3 off it, so the ray passes through every box and
hits nothing. One warp, every lane the same ray ("t2": origin (0,0,-2), direction +z), --resume 64,
--repeat 5 (first query latency from launches 1..4), GAS dumped in the same process.

    run.py BIN_S1 WORK
    /usr/bin/python3 run.py --report WORK
"""
import json, os, statistics as st, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.join(HERE, '..', '..', 'harness', 'setup3', 'run.py')
SETS = [('s', [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096, 16384, 65536, 262144]), ('o', [1, 2, 4, 8, 16, 32, 64, 128, 256, 1024, 4096, 16384, 65536])]


def run(b1, work):
    os.makedirs(work, exist_ok=True)
    for mode, ns in SETS:
        for n in ns:
            out = os.path.join(work, 'par_%s_%d' % (mode, n))
            p = subprocess.run([sys.executable, HARNESS, b1, 't2', 'zero', out + '.json', '--scene', 'par:%d:%s' % (n, mode), '--warps', '1',
                                '--repeat', '5', '--resume', '64', '--dump-gas', out + '.gas', '--dump-tris', out + '.tris'], capture_output=True, text=True)
            ok = p.returncode == 0 and 'exit 0' in p.stdout and ' error' not in p.stdout.replace('no error', '')
            print(os.path.basename(out), 'OK' if ok else 'FAILED', [l for l in p.stdout.split('\n') if l.startswith('resume')], flush=True)
            if not ok:
                sys.exit(p.stdout[-2000:] + p.stderr[-2000:])


def report(work):
    sys.path.insert(0, os.path.join(HERE, '..', 'BVH_5'))
    sys.path.insert(0, os.path.join(HERE, '..', '..', 'harness', 'setup3'))
    import gas
    import numpy as np
    from resume_check import closest
    for mode, ns in SETS:
        print('\n## par:N:%s' % mode)
        print('%7s | %6s %5s %6s | %6s %6s %5s | %8s %8s | %s' % ('N', 'nodes', 'depth', 'leaves', 'q/ray', 'pend1', 'hit', 'lat q1', 'lat qN', 'per round latency'))
        for n in ns:
            b = os.path.join(work, 'par_%s_%d' % (mode, n))
            j = json.load(open(b + '.json')); G = open(b + '.gas', 'rb').read(); W = gas.walk(G)
            leaves = sum(1 for off in W for _, k, _, _ in gas.slots(G, off) if k == 'leaf')
            tris = np.fromfile(b + '.tris', np.float32).reshape(-1, 3, 3).astype(np.float64)
            r = j['lane_rays'][0]; cpu = closest(tris, np.array(r[:3]), np.array(r[3:6]))[0]
            q = j['lane_queries']; assert len(set(q)) == 1, 'lanes differ'
            hit = j['lane_best'][0][0] != '0xffffffff'
            l1 = st.median(l['warp_clk'][0][1] - l['warp_clk'][0][0] for l in j['launches'][1:])
            rl = [r_['warp_clk'][0][1] - r_['warp_clk'][0][0] for r_ in j['resume_rounds']]
            print('%7d | %6d %5d %6d | %6d %6d %5s | %8.0f %8s | %s' % (n, len(W), max(v['depth'] for v in W.values()), leaves, q[0], j['lane_pending_pairs_first'][0],
                  'YES' if hit else ('cpu!' if cpu is not None else 'no'), l1, rl[-1] if rl else '-', rl[:6] + (['...'] if len(rl) > 6 else [])))


if __name__ == '__main__':
    report(sys.argv[2]) if sys.argv[1] == '--report' else run(*sys.argv[1:3])
