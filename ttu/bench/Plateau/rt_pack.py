"""Plateau check (a): RT only packed vs spread, with the loop slim kernel (build_patch --slim --remap --loop 32 --smfilter).

    rt_pack.py BIN_S0 BIN_S1 WORK

142 blocks x 28 warps are launched; only blocks whose SM is in the chosen set run (the others exit before any TTU code).
Spread: n TPCs 0..n-1 (SM 2t; TPC t is in GPC t mod 12, so round robin over the 12 GPCs). Packed: n TPCs taken GPC by GPC
(6 TPCs per GPC, one SM per TPC, so TPC sharing is the same in both). SM to GPC map: Plateau (b), notes/setup3/runs/plateau.
Per n and scene: CUDA events of 6 launches (first dropped), whole rays (resume loop in the kernel), every launch's rays
sampled against the OptiX / CPU check by the --dump-final of the last launch; blocks that ran counted from the host JSON.
"""
import json, os, statistics as st, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.join(HERE, '..', '..', 'harness', 'setup3', 'run.py')
sys.path.insert(0, os.path.join(HERE, '..', 'Memory_4'))
sys.path.insert(0, os.path.join(HERE, '..', 'Throughput_7b'))
from prep import make_rays  # noqa: E402
from check_final import main as check  # noqa: E402

WARPS, K, LANES = 28, 32, 28 * 32
SCENES = [('soup1e5', 'soup:100000'), ('soup3e6', 'soup:3000000')]
NS = [6, 12, 24, 36, 60]


def sets(n):
    by_gpc = [[t for t in range(71) if t % 12 == g] for g in range(12)]
    packed = [t for g in range(12) for t in by_gpc[g]][:n]
    return {'spread': [2 * t for t in range(n)], 'packed': [2 * t for t in packed]}


def go(binp, rays, out, scene, extra, quiet=True):
    p = subprocess.run([sys.executable, HARNESS, binp, rays, 'zero', out, '--scene', scene, '--warps', str(WARPS), '--blocks', '142', '--blockrays', '--slim', '--loop'] + extra,
                       capture_output=True, text=True)
    ok = 'exit 0' in p.stdout and ' error' not in p.stdout.replace('no error', '')
    print(os.path.basename(out), 'OK' if ok else 'FAILED', flush=True)
    if not ok:
        sys.exit(p.stdout[-2500:] + p.stderr[-1500:])


def run(b0, b1, work):
    os.makedirs(work, exist_ok=True)
    rays = make_rays(142 * LANES)
    f = os.path.join(work, 'rays.bin'); rays.tofile(f)
    for name, scene in SCENES:
        tris = os.path.join(work, 'tris_%s.bin' % name)
        for i, n in enumerate(NS):
            for kind, sm in sets(n).items():
                smset = ','.join(map(str, sm))
                if i == 0 and kind == 'spread':
                    go(b0, 'file:' + f, os.path.join(work, 's0_%s.json' % name), scene, ['--repeat', '1', '--sm-set', smset, '--dump-tris', tris])
                o = os.path.join(work, '%s_%s_%d' % (name, kind, n))
                go(b1, 'file:' + f, o + '.json', scene, ['--repeat', '6', '--sm-set', smset, '--dump-final', o + '.bin'])
                j = json.load(open(o + '.json'))
                ran = j['launches'][-1]['ran']
                print('  ran %d blocks (expected %d)' % (ran, n), end=' ')
                if ran != n:
                    print('MISMATCH'); continue
                # check the lanes of blocks that ran
                a = np.fromfile(o + '.bin', dtype=np.uint32).reshape(-1, 14)
                blocks = [b for b in range(142) if (2 * b if b < 71 else 2 * (b - 71) + 1) in sm]
                lanes = np.concatenate([np.arange(b * LANES, (b + 1) * LANES) for b in blocks])
                sub = a[lanes]; subr = rays[lanes]
                sub.tofile(o + '.sub.bin'); subr.tofile(o + '.rays.bin')
                check(o + '.sub.bin', o + '.rays.bin', tris, K, 32)


def report(work):
    print('%-8s %3s | %8s %8s | %9s %9s | %s' % ('scene', 'n', 'spread ms', 'packed ms', 'spread r/us', 'packed r/us', 'packed/spread rate'))
    for name, scene in SCENES:
        for n in NS:
            try:
                t = {k: st.median(l['ms'] for l in json.load(open(os.path.join(work, '%s_%s_%d.json' % (name, k, n))))['launches'][1:]) for k in ('spread', 'packed')}
            except Exception:
                continue
            r = {k: n * LANES / v / 1000 for k, v in t.items()}
            print('%-8s %3d | %8.4f %8.4f | %9.0f %9.0f | %.3f' % (name, n, t['spread'], t['packed'], r['spread'], r['packed'], r['packed'] / r['spread']))


if __name__ == '__main__':
    report(sys.argv[2]) if sys.argv[1] == '--report' else run(*sys.argv[1:4])
