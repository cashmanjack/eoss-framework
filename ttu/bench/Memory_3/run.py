"""Memory_3: bytes fetched per ray vs number of nodes, leaves and triangles (slim harness, ncu).

    run.py BIN_S0_SLIM BIN_S1_SLIM WORK        BIN: build_patch --slim --sync --remap --compact-out --regs 64 --max-out 61
    /usr/bin/python3 run.py --report WORK

Scenes par:N:s (the ray t2 passes every box of N triangles and misses: the query visits every node and leaf; coherent, every lane the
same ray) and soup:N random rays (per ray, one block of 32 warps in Throughput_6; here one warp, a different random ray per lane,
so the sectors are per 32 rays). One block, one warp (or 1 block x 32 warps for the soups), 2 launches, the second profiled by ncu
(L2 warm, L1 cold): RT core L1 fills from L2 (l1tex__m_xbar2l1tex_read_sectors) of stage 1 minus stage 0 (same slim loads and stores).
"""
import json, os, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.join(HERE, '..', '..', 'harness', 'setup3', 'run.py')
sys.path.insert(0, os.path.join(HERE, '..', 'Memory_4')); sys.path.insert(0, os.path.join(HERE, '..', 'BVH_5'))
import ncu_parse  # noqa: E402
NS = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 128, 256, 512, 1024, 4096, 16384]
M = 'l1tex__m_xbar2l1tex_read_sectors.sum,lts__t_sectors_srcunit_tex_op_read.sum,lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum,dram__bytes_read.sum'


def go(binp, out, scene, ray='t2', extra=()):
    p = subprocess.run([sys.executable, HARNESS, binp, ray, 'zero', out + '.json', '--scene', scene, '--warps', '1', '--blocks', '1', '--slim', '--repeat', '3',
                        '--ncu', M, '--ncu-out', out + '.csv', '--ncu-skip', '2'] + list(extra), capture_output=True, text=True)
    ok = 'exit 0' in p.stdout and ' error' not in p.stdout.replace('no error', '')
    print(os.path.basename(out), 'OK' if ok else 'FAILED', flush=True)
    if not ok:
        sys.exit(p.stdout[-2000:] + p.stderr[-2000:])


def run(b0, b1, work):
    os.makedirs(work, exist_ok=True)
    for n in NS:
        go(b0, os.path.join(work, 'n0_par_%d' % n), 'par:%d:s' % n)
        go(b1, os.path.join(work, 'n1_par_%d' % n), 'par:%d:s' % n, extra=['--dump-gas', os.path.join(work, 'par_%d.gas' % n)])


def report(work):
    import gas
    rows = []
    print('%6s | %6s %5s %6s | %9s %9s %8s | %s' % ('N', 'nodes', 'depth', 'leaves', 'L1 fills', 'bytes', 'B/node', 'L2 hit, DRAM B'))
    for n in NS:
        try:
            r0 = ncu_parse.rows(os.path.join(work, 'n0_par_%d.csv' % n))[-1]; r1 = ncu_parse.rows(os.path.join(work, 'n1_par_%d.csv' % n))[-1]
            G = open(os.path.join(work, 'par_%d.gas' % n), 'rb').read()
        except Exception:
            continue
        W = gas.walk(G); leaves = sum(1 for off in W for _, k, _, _ in gas.slots(G, off) if k == 'leaf')
        d = lambda k: r1[k] - r0[k]
        f = d('l1tex__m_xbar2l1tex_read_sectors.sum'); l2 = d('lts__t_sectors_srcunit_tex_op_read.sum'); hit = d('lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum')
        rows.append((n, len(W), leaves, f))
        print('%6d | %6d %5d %6d | %9.0f %9.0f %8.1f | %.0f%%, %.0f' % (n, len(W), max(v['depth'] for v in W.values()), leaves, f, f * 32, f * 32 / len(W), 100 * hit / l2 if l2 else 0, d('dram__bytes_read.sum')))
    A = np.array([[1.0, r[1], r[2], r[0]] for r in rows]); y = np.array([r[3] * 32.0 for r in rows])
    co = np.linalg.lstsq(A, y, rcond=None)[0]
    print('fit bytes = %.0f + %.1f per node + %.1f per leaf + %.1f per triangle (rms %.0f)' % (*co, np.sqrt(np.mean((A @ co - y) ** 2))))


if __name__ == '__main__':
    report(sys.argv[2]) if sys.argv[1] == '--report' else run(*sys.argv[1:4])
