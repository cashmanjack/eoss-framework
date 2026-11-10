"""BVH_3: triangles vs custom primitives whose intersection always returns false (David), plain OptiX.

    run.py BUILD_DIR WORK       (BUILD_DIR holds bvh3 and bvh3_dev.ptx)
    run.py --report WORK

Same gates as the harness: driver 580.178.04, GPU 1 idle, 30 s timeout.
"""
import json, os, statistics as st, subprocess, sys

NS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 1024, 4096, 16384]
OPTIX_LIB = os.path.expanduser('~/opt/nvoptix-580.178.04')


def smi(*a):
    return subprocess.run(['nvidia-smi'] + list(a), capture_output=True, text=True).stdout.strip()


def run(bd, work):
    os.makedirs(work, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='1', LD_LIBRARY_PATH=OPTIX_LIB + ':' + os.environ.get('LD_LIBRARY_PATH', ''))
    for mode in ('s', 'o'):
        for n in NS:
            for kind, cnt in (('tri', 0), ('aabb', 0), ('aabb', 1)):
                if smi('--query-gpu=driver_version', '--format=csv,noheader').split('\n')[0] != '580.178.04':
                    sys.exit('driver changed')
                if smi('-i', '1', '--query-compute-apps=pid', '--format=csv,noheader'):
                    sys.exit('GPU 1 busy')
                out = os.path.join(work, '%s_%s_%d_c%d.json' % (kind, mode, n, cnt))
                p = subprocess.run(['timeout', '-k', '2', '30', os.path.join(bd, 'bvh3'), os.path.join(bd, 'bvh3_dev.ptx'), kind, str(n), mode, str(cnt), out],
                                   env=env, capture_output=True, text=True)
                print(os.path.basename(out), 'OK' if p.returncode == 0 else 'FAILED', flush=True)
                if p.returncode:
                    sys.exit(p.stdout + p.stderr)


def report(work):
    for mode in ('s', 'o'):
        print('\n## %s (par:N:%s geometry), cycles per optixTrace (median lane, median of launches 1..10)' % ({'s': 'stacked', 'o': 'overlapping'}[mode], mode))
        print('%6s | %9s %9s %9s | %10s %10s | %s' % ('N', 'tri', 'aabb', 'aabb+cnt', 'IS calls', 'per ray', 'hits'))
        rows = []
        for n in NS:
            r = {}
            for kind, cnt in (('tri', 0), ('aabb', 0), ('aabb', 1)):
                j = json.load(open(os.path.join(work, '%s_%s_%d_c%d.json' % (kind, mode, n, cnt))))
                L = j['launches'][1:]
                r[(kind, cnt)] = (st.median(l['cyc_med'] for l in L), st.median(l['is_calls'] for l in L), sum(l['hits'] for l in L))
            rows.append((n, r))
            print('%6d | %9.0f %9.0f %9.0f | %10.0f %10.2f | %d %d %d' % (n, r[('tri', 0)][0], r[('aabb', 0)][0], r[('aabb', 1)][0], r[('aabb', 1)][1],
                  r[('aabb', 1)][1] / 32, r[('tri', 0)][2], r[('aabb', 0)][2], r[('aabb', 1)][2]))
        big = [x for x in rows if x[0] >= 256]
        if len(big) >= 2:
            (n0, a), (n1, b) = big[0], big[-1]
            sl = lambda k: (b[k][0] - a[k][0]) / (n1 - n0)
            print('slope N %d..%d (cycles per primitive): tri %.0f, aabb %.0f, aabb+count %.0f; extra per candidate (aabb - tri) %.0f' % (
                n0, n1, sl(('tri', 0)), sl(('aabb', 0)), sl(('aabb', 1)), sl(('aabb', 0)) - sl(('tri', 0))))


if __name__ == '__main__':
    report(sys.argv[2]) if sys.argv[1] == '--report' else run(*sys.argv[1:3])
