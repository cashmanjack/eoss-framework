"""Throughput_7 with the slim harness: the same rays through OptiX and through our kernel (OptiX overhead).

    run.py BIN_SLIM_S0 BIN_SLIM_S1 BIN_SMFILTER_S1 WORK

BIN_SLIM: build_patch --slim --sync --remap --compact-out --regs 64 --max-out 61 (32 warps per block, one block per SM).
BIN_SMFILTER_S1: --clock --smfilter (non slim, used only with --sched compact to count queries per ray on the same rays).
For each scene and G in {71, 142} blocks (G x 1,024 rays, one ray per lane, blockrays): 
  slim first query: CUDA event time per launch (6 launches, first dropped);
  OptiX closest hit on the same rays: CUDA event time (10 launches after one untimed);
  queries per ray: scheduler run (compact, 16 warps x 142 blocks, 8 rounds) on the same rays.
"""
import json, os, statistics as st, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.join(HERE, '..', '..', 'harness', 'setup3', 'run.py')
sys.path.insert(0, os.path.join(HERE, '..', 'Memory_4'))
from prep import make_rays  # noqa: E402

SCENES = [('tri1', 'tri1', 'coh'), ('soup1e5', 'soup:100000', 'rnd'), ('soup3e6', 'soup:3000000', 'rnd')]
GS = [71, 142]


def go(binp, rays, out, scene, extra):
    p = subprocess.run([sys.executable, HARNESS, binp, rays, 'zero', out, '--scene', scene] + extra, capture_output=True, text=True)
    ok = 'exit 0' in p.stdout and ' error' not in p.stdout.replace('no error', '')
    print(os.path.basename(out), 'OK' if ok else 'FAILED', flush=True)
    if not ok:
        sys.exit(p.stdout[-2500:] + p.stderr[-1500:])
    return p.stdout


def run(b0, b1, bs, work):
    os.makedirs(work, exist_ok=True)
    big = make_rays(142 * 1024)
    one = np.array([[0, 0, -1, 0, 0, 0, 1, 1e16]], np.float32)
    coh = np.repeat(one, 142 * 1024, axis=0)
    for name, scene, kind in SCENES:
        src = coh if kind == 'coh' else big
        for g in GS:
            f = os.path.join(work, 'rays_%s_%d.bin' % (kind, g)); src[:g * 1024].tofile(f)
            if name == 'tri1' and g == GS[0]:   # stage 0 once, then smallest first
                go(b0, 'file:' + f, os.path.join(work, 's0.json'), scene, ['--warps', '32', '--blocks', str(g), '--blockrays', '--slim'])
            go(b1, 'file:' + f, os.path.join(work, '%s_g%d.json' % (name, g)), scene,
               ['--warps', '32', '--blocks', str(g), '--blockrays', '--slim', '--repeat', '6', '--optix-repeat', '10'])
            go(bs, 'file:' + f, os.path.join(work, '%s_g%d_q.json' % (name, g)), scene,
               ['--warps', '16', '--blocks', '142', '--sched', 'compact', '--resume', '12'])


def report(work):
    print('%-8s %4s | %8s %8s | %6s %6s | %9s %9s | %8s %8s' % ('scene', 'G', 'slim ms', 'optix ms', 'q/ray', 'maxq', 'slim r/us', 'optx r/us', 'optx/slim', 'per query'))
    for name, scene, kind in SCENES:
        for g in GS:
            try:
                j = json.load(open(os.path.join(work, '%s_g%d.json' % (name, g)))); q = json.load(open(os.path.join(work, '%s_g%d_q.json' % (name, g))))
            except Exception:
                continue
            n = g * 1024
            sm = st.median(l['ms'] for l in j['launches'][1:]); om = st.median(j['optix_ms'])
            qs = [abs(x) for x in q['ray_queries']]; mq = sum(qs) / len(qs)
            # per query: OptiX time / total queries vs our first query launch time / n
            print('%-8s %4d | %8.4f %8.4f | %6.2f %6d | %9.0f %9.0f | %8.2f %8.2f' % (name, g, sm, om, mq, max(qs), n / sm / 1000, n / om / 1000, om / sm, (om / (n * mq)) / (sm / n)))


if __name__ == '__main__':
    report(sys.argv[2]) if sys.argv[1] == '--report' else run(*sys.argv[1:5])
