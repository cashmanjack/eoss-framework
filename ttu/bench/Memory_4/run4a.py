"""Memory_4a: no oracle grouping (regroup by pending node pointer between resume rounds), plus Memory_4
(a) and (b) redone with finished lanes masked instead of riding along with tmax 0.

    run4a.py BIN_S1 WORK      BIN_S1: stage 1 setup3 built with --clock --smfilter, checked

WORK/s<N>/ holds rays_{interleaved,grouped}.bin and starts_*.txt from the Memory_4 prep (same 8,192 rays).
Every run: --sched, --sms 0,0 (one block on SM 0), --resume 16. Warm = --passes 2 (the second pass is
reported: L2 holds what the first left, L1 empty at every launch); cold = --flush 256 before every launch.
"""
import os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.join(HERE, '..', '..', 'harness', 'setup3', 'run.py')
V = [('int_inplace', 'interleaved', 'inplace', None), ('int_compact', 'interleaved', 'compact', None), ('int_regroup', 'interleaved', 'regroup', None),
     ('grp_inplace', 'grouped', 'inplace', None), ('grp_compact', 'grouped', 'compact', None),
     ('bwarp_inplace', 'grouped', 'inplace', 'grouped_warp'), ('bwarp_compact', 'grouped', 'compact', 'grouped_warp'),
     ('bray_inplace', 'grouped', 'inplace', 'grouped_ray'), ('bray_compact', 'grouped', 'compact', 'grouped_ray')]


def main(b1, work):
    for n in ('100000', '3000000'):
        d = os.path.join(work, 's' + n); os.makedirs(os.path.join(d, 'a'), exist_ok=True)
        for w in (1, 4, 8, 16):
            for st, ex in (('warm', ['--passes', '2']), ('cold', ['--flush', '256'])):
                for name, order, sched, starts in V:
                    out = os.path.join(d, 'a', '%s_w%d_%s.json' % (name, w, st))
                    sf = ['--start-file', os.path.join(d, 'starts_%s.txt' % starts)] if starts else []
                    p = subprocess.run([sys.executable, HARNESS, b1, 'file:' + os.path.join(d, 'rays_%s.bin' % order), 'zero', out, '--scene', 'soup:' + n,
                                        '--warps', str(w), '--sms', '0,0', '--sched', sched, '--resume', '16'] + ex + sf, capture_output=True, text=True)
                    ok = p.returncode == 0 and 'exit 0' in p.stdout and ' error' not in p.stdout.replace('no error', '')
                    print(os.path.basename(out), 'OK' if ok else 'FAILED', flush=True)
                    if not ok:
                        sys.exit(p.stdout[-2000:] + p.stderr[-2000:])


if __name__ == '__main__':
    main(*sys.argv[1:3])
