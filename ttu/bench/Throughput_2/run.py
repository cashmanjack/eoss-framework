"""Throughput_2: throughput vs active lanes per warp, idle lanes masked off vs riding along with tmax 0.

    run.py BIN_S1 WORK      BIN_S1: stage 1 setup3 built with --clock --sync --smfilter, checked

One block on SM 0 (--sms 0,0). Active lanes L = 32, 16, 8, 1 of every warp; idle lanes either exit
before TTUOPEN (mask) or run the group with tmax 0 (tmax0). W = 1, 4, 16 warps; 31 launches, the
first dropped. State: warm across launches (L2 holds the BVH; L1 is empty at every kernel start).
Scenes: tri1 with one coherent ray (cheap query); soup:100000 (6.2 MB GAS) with a random ray per lane.
First query only (no resume): the question is RT core time per query, not finished rays.
"""
import os, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.join(HERE, '..', '..', 'harness', 'setup3', 'run.py')
sys.path.insert(0, os.path.join(HERE, '..', 'Memory_4'))
from prep import make_rays  # noqa: E402


def main(b1, work):
    os.makedirs(work, exist_ok=True)
    for w in (1, 4, 16):
        make_rays(32 * w).tofile(os.path.join(work, 'rays_w%d.bin' % w))
    for scene, rays in (('tri1', 'hit'), ('soup:100000', None)):
        for w in (1, 4, 16):
            for L in (32, 16, 8, 1):
                for idle in (('mask',) if L == 32 else ('mask', 'tmax0', 'tneg', 'empty')):
                    out = os.path.join(work, '%s_w%d_L%d_%s.json' % (scene.split(':')[0], w, L, idle))
                    r = rays or 'file:' + os.path.join(work, 'rays_w%d.bin' % w)
                    p = subprocess.run([sys.executable, HARNESS, b1, r, 'zero', out, '--scene', scene, '--warps', str(w), '--repeat', '31',
                                        '--sms', '0,0', '--active', str(L), '--idle', idle], capture_output=True, text=True)
                    ok = p.returncode == 0 and 'exit 0' in p.stdout and ' error' not in p.stdout.replace('no error', '')
                    print(os.path.basename(out), 'OK' if ok else 'FAILED', flush=True)
                    if not ok:
                        sys.exit(p.stdout[-2000:] + p.stderr[-2000:])


if __name__ == '__main__':
    main(*sys.argv[1:3])
