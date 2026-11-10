"""Throughput_7 fair comparison: OptiX vs our kernel with the resume loop inside the kernel (build_patch --loop).

    run.py BIN_LOOP_S0 BIN_LOOP_S1 WORK            BIN: --slim --sync --remap --compact-out --regs 72 --max-out 61 --loop 32
    run.py --report WORK

28 warps per block (72 registers), one block per SM, G blocks, G x 896 rays, one ray per lane (blockrays).
Per scene and G: stage 0 once per scene (also dumps the triangles), then stage 1: CUDA event time per launch (6 launches,
first dropped) of the whole ray (all queries, loop in the kernel) vs OptiX closest hit on the same rays (10 launches);
final results of 64 sampled lanes checked against a CPU brute force; rounds per ray from the loop counter.
"""
import json, os, statistics as st, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.join(HERE, '..', '..', 'harness', 'setup3', 'run.py')
sys.path.insert(0, os.path.join(HERE, '..', 'Memory_4'))
from prep import make_rays  # noqa: E402
sys.path.insert(0, HERE)
from check_final import main as check  # noqa: E402

SCENES = [('tri1', 'tri1', 'coh'), ('soup1e5', 'soup:100000', 'rnd'), ('soup3e6', 'soup:3000000', 'rnd')]
GS = [1, 71, 142]
WARPS = 28
K = 32
LANES = WARPS * 32


def go(binp, rays, out, scene, extra):
    p = subprocess.run([sys.executable, HARNESS, binp, rays, 'zero', out, '--scene', scene, '--warps', str(WARPS), '--blockrays', '--slim', '--loop'] + extra,
                       capture_output=True, text=True)
    ok = 'exit 0' in p.stdout and ' error' not in p.stdout.replace('no error', '')
    print(os.path.basename(out), 'OK' if ok else 'FAILED', flush=True)
    if not ok:
        sys.exit(p.stdout[-2500:] + p.stderr[-1500:])


def run(b0, b1, work):
    os.makedirs(work, exist_ok=True)
    big = make_rays(142 * LANES)
    one = np.array([[0, 0, -1, 0, 0, 0, 1, 1e16]], np.float32)
    coh = np.repeat(one, 142 * LANES, axis=0)
    for name, scene, kind in SCENES:
        src = coh if kind == 'coh' else big
        tris = os.path.join(work, 'tris_%s.bin' % name)
        for g in GS:
            f = os.path.join(work, 'rays_%s_%d.bin' % (kind, g)); src[:g * LANES].tofile(f)
            if g == GS[0]:   # stage 0 first for each scene (smallest launch), also dumps the triangles
                go(b0, 'file:' + f, os.path.join(work, 's0_%s.json' % name), scene, ['--blocks', str(g), '--repeat', '1', '--dump-tris', tris])
            go(b1, 'file:' + f, os.path.join(work, '%s_g%d.json' % (name, g)), scene,
               ['--blocks', str(g), '--repeat', '6', '--optix-repeat', '10', '--dump-final', os.path.join(work, '%s_g%d.bin' % (name, g))])
            print(' check', name, g, end=' ')
            if os.path.exists(tris):
                check(os.path.join(work, '%s_g%d.bin' % (name, g)), f, tris, K, 64)


def report(work):
    print('%-8s %4s | %8s %8s | %6s %5s | %9s %9s | %8s' % ('scene', 'G', 'ours ms', 'optix ms', 'rnd/ray', 'max', 'ours r/us', 'optx r/us', 'optx/ours'))
    for name, scene, kind in SCENES:
        for g in GS:
            try:
                j = json.load(open(os.path.join(work, '%s_g%d.json' % (name, g))))
                a = np.fromfile(os.path.join(work, '%s_g%d.bin' % (name, g)), dtype=np.uint32).reshape(-1, 14)
            except Exception:
                continue
            n = g * LANES
            sm = st.median(l['ms'] for l in j['launches'][1:]); om = st.median(j['optix_ms'])
            rounds = K - a[:, 13].astype(int)
            print('%-8s %4d | %8.4f %8.4f | %6.2f %5d | %9.0f %9.0f | %8.2f' % (name, g, sm, om, rounds.mean(), rounds.max(), n / sm / 1000, n / om / 1000, om / sm))


if __name__ == '__main__':
    report(sys.argv[2]) if sys.argv[1] == '--report' else run(*sys.argv[1:4])
