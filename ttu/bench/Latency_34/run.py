"""Latency_3 (children per node, nodes visited at fixed depth) and Latency_4 (triangles per leaf).

    run.py BIN_S1 WORK            BIN_S1: checked stage 1 build with --clock (one session, L2 warm, L1 cold per launch)
    /usr/bin/python3 run.py --report WORK

Scenes par:N:s (N triangles parallel to a +z ray, 1e-3 off it; the ray passes through every box and hits nothing):
the query visits every node and leaf and tests every triangle. N 1..48 is ONE node (at most 12 leaf slots), so
leaves grow to 12 and then triangles per leaf grow (Latency_4); N 1..12 is children per node (Latency_3);
N 33..400 has 3..13 nodes at depth 1 (nodes visited at fixed depth). 1 warp, 5 launches (first dropped).
"""
import json, os, statistics as st, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.join(HERE, '..', '..', 'harness', 'setup3', 'run.py')
NS = list(range(1, 49)) + [56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 400]


def run(b1, work):
    os.makedirs(work, exist_ok=True)
    for n in NS:
        out = os.path.join(work, 'par_s_%d' % n)
        p = subprocess.run([sys.executable, HARNESS, b1, 't2', 'zero', out + '.json', '--scene', 'par:%d:s' % n, '--warps', '1', '--repeat', '5',
                            '--dump-gas', out + '.gas'], capture_output=True, text=True)
        ok = p.returncode == 0 and 'exit 0' in p.stdout and ' error' not in p.stdout.replace('no error', '')
        print(os.path.basename(out), 'OK' if ok else 'FAILED', flush=True)
        if not ok:
            sys.exit(p.stdout[-2000:] + p.stderr[-2000:])


def rows(work):
    sys.path.insert(0, os.path.join(HERE, '..', 'BVH_5'))
    import gas
    out = []
    for n in NS:
        b = os.path.join(work, 'par_s_%d' % n)
        if not os.path.exists(b + '.json'):
            continue
        j = json.load(open(b + '.json')); G = open(b + '.gas', 'rb').read(); W = gas.walk(G)
        leaves = sum(1 for off in W for _, k, _, _ in gas.slots(G, off) if k == 'leaf')
        types = {}
        for off in W:
            for _, k, t, _ in gas.slots(G, off):
                if k == 'leaf': types[t] = types.get(t, 0) + 1
        wo = j['warp_out'][0]; hit = bool(wo[3]) or bool(j['ref'][0]['hit']); pend = any(wo[x] for x in (56, 57, 58, 59, 60, 61, 70, 71))
        lat = st.median(l['warp_clk'][0][1] - l['warp_clk'][0][0] for l in j['launches'][1:])
        out.append(dict(n=n, nodes=len(W), depth=max(v['depth'] for v in W.values()), leaves=leaves, hit=hit, lat=lat, pend=pend))
    return out


def report(work):
    import numpy as np
    r = rows(work)
    print('%4s | %5s %5s %6s | %7s | %8s | %s' % ('N', 'nodes', 'depth', 'leaves', 'tri/leaf', 'latency', 'checks'))
    for x in r:
        print('%4d | %5d %5d %6d | %7.2f | %8.0f | hit %s, pending %s' % (x['n'], x['nodes'], x['depth'], x['leaves'], x['n'] / x['leaves'], x['lat'], x['hit'], x['pend']))
    A = lambda rows_, cols: np.array([[1.0] + [row[c] for c in cols] for row in rows_])
    for name, sel, cols in (('one node: lat = a + b leaves + c triangles', [x for x in r if x['nodes'] == 1], ['leaves', 'n']),
                            ('one node, N <= 12 (1 triangle per leaf): lat = a + b children', [x for x in r if x['nodes'] == 1 and x['n'] <= 12], ['n']),
                            ('one node, N >= 12 (12 leaves, more triangles per leaf): lat = a + c triangles', [x for x in r if x['nodes'] == 1 and x['n'] >= 12], ['n']),
                            ('all scenes (N 1..400): lat = a + b nodes + c leaves + d triangles', r, ['nodes', 'leaves', 'n'])):
        if len(sel) < len(cols) + 2: continue
        X = A(sel, cols); y = np.array([x['lat'] for x in sel]); co, res, *_ = np.linalg.lstsq(X, y, rcond=None)
        print('%-85s n=%2d  a=%7.0f  %s  rms %.0f' % (name, len(sel), co[0], '  '.join('%s=%.1f' % (c, v) for c, v in zip(cols, co[1:])), (np.sqrt(np.mean((X @ co - y) ** 2)))))


if __name__ == '__main__':
    report(sys.argv[2]) if sys.argv[1] == '--report' else run(*sys.argv[1:3])
