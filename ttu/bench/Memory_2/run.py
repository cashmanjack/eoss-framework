"""Memory_2 (counters): how many distinct nodes fit in the RT core's node cache (the SM's L1)?

    run.py BIN_S0C BIN_S1C BIN_S0TK BIN_S1TK WORK
      S0C/S1C: build_patch --stage 0|1 --clock (one session)     S0TK/S1TK: --stage 0|1 --clock --twice --compact-out (session A warms, B timed)
    /usr/bin/python3 run.py --report WORK

Scene soup:100000, R distinct random rays (lane i -> ray i mod R), 16 warps in one block on one SM, last launch under ncu (L2 warm, L1 cold at the start).
RT core L1 fills from L2 (l1tex__m_xbar2l1tex_read_sectors) = stage 1 minus stage 0 of the same build (same input loads and stores).
  D(R)   = fills of the single session build: distinct sectors the R rays touch (the footprint).
  F2(R)  = fills of the two session build: D(R) for session A plus whatever session B has to fetch again.
  B refetch fraction = (F2 - D) / D: 0 when the second session finds everything in L1, 1 when none of it survives.
Two L1 sizes: about 90 KB (no reservation) and about 20 KB (--smem 98304).  2 ray pools (seeds) per R.
"""
import os, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.join(HERE, '..', '..', 'harness', 'setup3', 'run.py')
sys.path.insert(0, os.path.join(HERE, '..', 'Memory_4'))
from prep import make_rays  # noqa: E402
import ncu_parse  # noqa: E402
RS = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]
SEEDS = [2026, 2027]
W = 16
CARVE = [('big', []), ('small', ['--smem', '98304'])]
M = 'l1tex__m_xbar2l1tex_read_sectors.sum,l1tex__t_sectors_lookup_hit.sum,l1tex__t_sectors.sum'


def go(binp, rays, out, extra):
    p = subprocess.run([sys.executable, HARNESS, binp, 'file:' + rays, 'zero', out + '.json', '--scene', 'soup:100000', '--warps', str(W), '--repeat', '3',
                        '--ncu', M, '--ncu-out', out + '.csv', '--ncu-skip', '2'] + extra, capture_output=True, text=True)
    ok = 'exit 0' in p.stdout and ' error' not in p.stdout.replace('no error', '')
    if not ok:
        sys.exit('FAILED ' + out + p.stdout[-1500:] + p.stderr[-1500:])
    os.remove(out + '.json')


def run(s0c, s1c, s0tk, s1tk, work):
    os.makedirs(work, exist_ok=True)
    for R in RS:
        for sd in SEEDS:
            f = os.path.join(work, 'rays.bin'); make_rays(512, sd)[np.arange(32 * W) % R].tofile(f)
            for cn, ex in CARVE:
                for tag, b in (('s0c', s0c), ('s1c', s1c), ('s0t', s0tk), ('s1t', s1tk)):
                    if tag in ('s0c', 's1c') and cn == 'small': continue   # D(R) once (L1 size does not change the footprint)
                    go(b, f, os.path.join(work, '%s_%s_%d_%d' % (tag, cn, R, sd)), ex)
            print('R', R, 'seed', sd, 'done', flush=True)
    os.remove(os.path.join(work, 'rays.bin'))


def report(work):
    g = lambda tag, cn, R, sd, k='l1tex__m_xbar2l1tex_read_sectors.sum': ncu_parse.rows(os.path.join(work, '%s_%s_%d_%d.csv' % (tag, cn, R, sd)))[-1][k]
    print('R distinct rays; D = distinct sectors (footprint, single session); refetch = share of session B fills (0 = all found in L1), median of 2 pools')
    print('%5s | %9s %8s | %14s %14s' % ('R', 'D sectors', 'D KB', 'refetch ~90KB', 'refetch ~20KB'))
    for R in RS:
        D, rf = [], {'big': [], 'small': []}
        try:
            for sd in SEEDS:
                d = g('s1c', 'big', R, sd) - g('s0c', 'big', R, sd); D.append(d)
                for cn in rf:
                    f2 = g('s1t', cn, R, sd) - g('s0t', cn, R, sd); rf[cn].append((f2 - d) / d)
        except Exception:
            continue
        d = np.median(D)
        print('%5d | %9.0f %8.1f | %14.2f %14.2f' % (R, d, d * 32 / 1024, np.median(rf['big']), np.median(rf['small'])))


if __name__ == '__main__':
    report(sys.argv[2]) if sys.argv[1] == '--report' else run(*sys.argv[1:6])
