"""Memory_4b: grouped vs interleaved on the 186 MB scene, all 142 SMs, 16 warps per SM (the most a 128
register ttu_one allows), 262,144 rays (prep.py, box keys: within 3% of exact keys in Memory_4).

    run4b.py BIN_S0 BIN_S1 WORK     (stage 0 and 1 built with --clock --smfilter, checked)

--sched inplace and compact, --blocks 142 (block b lands on SM 2b for b < 71, then odd SMs; a wrong guess
would make active lanes exit, which the host's store check turns into an error). Warm = --passes 2
(second pass reported); cold = --flush 256 before every launch. ncu: groups A (L1) and B (L2, DRAM),
each its own process; warm profiles only pass 1 (--ncu-skip = launches of one pass); stage 0 baseline
(no TTU, round 0 only) per state for the template's own traffic.
"""
import json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run import GA, GB  # noqa: E402

HARNESS = os.path.join(HERE, '..', '..', 'harness', 'setup3', 'run.py')
SC = 'soup:3000000'


def go(binp, rays, out, extra):
    p = subprocess.run([sys.executable, HARNESS, binp, 'file:' + rays, 'zero', out, '--scene', SC, '--warps', '16', '--blocks', '142'] + extra,
                       capture_output=True, text=True)
    ok = 'exit 0' in p.stdout or ('exit 1' in p.stdout and 'stage0' in out)   # stage 0 never finishes rays (no TTU): exit 1 by design
    ok = ok and 'did not store' not in p.stdout + p.stderr and ' error' not in p.stdout.replace('no error', '')
    print(os.path.basename(out), 'OK' if ok else 'FAILED', flush=True)
    if not ok:
        sys.exit(p.stdout[-2000:] + p.stderr[-2000:])


def main(b0, b1, work):
    d = os.path.join(work, 's3000000'); t = os.path.join(d, 't'); n = os.path.join(d, 'n')
    os.makedirs(t, exist_ok=True); os.makedirs(n, exist_ok=True)
    S = {'warm': ['--passes', '2'], 'cold': ['--flush', '256']}
    for st, ex in S.items():
        for o in ('interleaved', 'grouped'):
            for m in ('inplace', 'compact'):
                go(b1, os.path.join(d, 'rays_%s.bin' % o), os.path.join(t, '%s_%s_%s.json' % (o, m, st)), ['--sched', m, '--resume', '16'] + ex)
    for st, ex in S.items():
        for g, mm in (('A', GA), ('B', GB)):
            skip = ['--ncu-skip', '4'] if st == 'warm' else []
            go(b0, os.path.join(d, 'rays_interleaved.bin'), os.path.join(n, 'stage0_%s_%s.json' % (st, g)),
               ['--sched', 'compact', '--resume', '0'] + ex + ['--ncu', mm, '--ncu-out', os.path.join(n, 'stage0_%s_%s.csv' % (st, g))] + skip)
            for o in ('interleaved', 'grouped'):
                for m in ('inplace', 'compact'):
                    tj = json.load(open(os.path.join(t, '%s_%s_%s.json' % (o, m, st))))
                    skip = ['--ncu-skip', str(tj['launches_total'])] if st == 'warm' else []
                    base = os.path.join(n, '%s_%s_%s_%s' % (o, m, st, g))
                    go(b1, os.path.join(d, 'rays_%s.bin' % o), base + '.json', ['--sched', m, '--resume', '16'] + ex + ['--ncu', mm, '--ncu-out', base + '.csv'] + skip)


if __name__ == '__main__':
    main(*sys.argv[1:4])
