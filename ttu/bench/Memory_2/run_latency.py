"""Memory_2: size of the node cache seen by the RT core (latency steps vs number of distinct nodes).

    run.py BIN_S0 BIN_S1 WORK        BIN: build_patch --clock --twice --compact-out (session A warms, session B timed in the same kernel)
    /usr/bin/python3 run.py --report WORK

Scene soup:100000 (6.2 MB GAS, random rays are all different paths). R distinct random rays; lane i traces ray i mod R, always W = 16 warps
(512 lanes, one block, one SM, 128 registers: the most that fits), so only the number of distinct nodes changes, not the work in flight.
Ray paths differ in length, so every R is run with 4 different random pools (seeds) and the median of the 4 is reported. Second trace latency per warp, median over warps and 10
launches. L1 size: no reservation (about 90 KB) or --smem 98304 (reserves shared memory: about 20 KB L1, see Memory_1).
Distinct footprint of the R rays: R x 45 sectors x 32 B at most (45 L1 fills per ray measured in Throughput_6), shown as KB.
"""
import json, os, statistics as st, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.join(HERE, '..', '..', 'harness', 'setup3', 'run.py')
sys.path.insert(0, os.path.join(HERE, '..', 'Memory_4'))
from prep import make_rays  # noqa: E402
RS = [1, 2, 4, 8, 16, 24, 32, 48, 64, 80, 96, 128, 192, 256, 384, 512]
SEEDS = [2026, 2027, 2028, 2029]
W = 16   # 128 registers: at most 16 warps per block
CARVE = [('l1big', []), ('l1small', ['--smem', '98304'])]


def go(binp, rays, out, extra):
    p = subprocess.run([sys.executable, HARNESS, binp, rays, 'zero', out, '--scene', 'soup:100000'] + extra, capture_output=True, text=True)
    ok = 'exit 0' in p.stdout and ' error' not in p.stdout.replace('no error', '')
    print(os.path.basename(out), 'OK' if ok else 'FAILED', flush=True)
    if not ok:
        sys.exit(p.stdout[-2000:] + p.stderr[-2000:])


def run(b0, b1, work):
    os.makedirs(work, exist_ok=True)
    for R in RS:
        for sd in SEEDS:
            f = os.path.join(work, 'rays_%d_%d.bin' % (R, sd)); make_rays(512, sd)[np.arange(32 * W) % R].tofile(f)
            for name, ex in CARVE:
                if R == RS[0] and name == 'l1big' and sd == SEEDS[0]:
                    go(b0, 'file:' + f, os.path.join(work, 's0_%d.json' % R), ['--warps', str(W), '--repeat', '2'] + ex)
                go(b1, 'file:' + f, os.path.join(work, '%s_%d_%d.json' % (name, R, sd)), ['--warps', str(W), '--repeat', '11'] + ex)
            os.remove(f)


def report(work):
    print('second trace latency per warp (cycles), median over 4 ray pools of the median over warps and launches; 16 warps, 512 lanes')
    print('%5s %8s | %9s %9s | %7s | pool spread (min..max) big / small' % ('R', 'KB<=', 'L1 ~90KB', 'L1 ~20KB', 'ratio'))
    for R in RS:
        v = []
        for name, _ in CARVE:
            m = []
            for sd in SEEDS:
                p = os.path.join(work, '%s_%d_%d.json' % (name, R, sd))
                if os.path.exists(p):
                    j = json.load(open(p)); m.append(st.median(w[1] - w[0] for l in j['launches'][1:] for w in l['warp_clk']))
            v.append(m)
        if not all(v): continue
        a, b = st.median(v[0]), st.median(v[1])
        print('%5d %8.1f | %9.0f %9.0f | %7.2f | %.0f..%.0f / %.0f..%.0f' % (R, R * 45 * 32 / 1024, a, b, b / a, min(v[0]), max(v[0]), min(v[1]), max(v[1])))


if __name__ == '__main__':
    report(sys.argv[2]) if sys.argv[1] == '--report' else run(*sys.argv[1:4])
