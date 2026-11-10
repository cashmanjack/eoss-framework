"""Memory_4 (a): the same incoherent rays, grouped by subtree vs randomly interleaved, root starts.

    run.py BIN_S0 BIN_S1 WORK [timing|ncu|all|b-timing|b-ncu|b]

(b): the same rays and groups, started at subtree nodes (startfiles.py): b_grouped_warp (one start node
per warp: the treelet queue model), b_grouped_ray and b_interleaved_ray (each ray at its own key node).
The (a) stage 0 baselines serve (b) too (same template, same W and state).

WORK/s<N>/ must hold prep.py output for soup:100000 (6.2 MB GAS, > L1, < L2) and soup:3000000
(186 MB, > L2). Every run goes through ttu/harness/setup3/run.py (checked sha256, idle GPU 1, 30 s).
Cache states:
  warm  --passes 2, only pass 1 is used: L2 holds what the previous pass over the same rays left
        (all of the 6.2 MB BVH; part of the 186 MB one); L1 is empty at every kernel start
  cold  --flush 256 before every launch, resume rounds included: neither L1 nor L2 holds the BVH
Timing at W = 1, 4, 8, 16 warps; ncu at W = 4, 8, 16 (W = 1 needs about 1,500 profiled launches, over 30 s).
ncu: two single pass metric groups, each its own process (L1; L2 and DRAM), caches untouched by ncu.
Stage 0 baseline (no TTU, one launch per chunk) per scene, W and state, to subtract our template's own loads.
"""
import os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.join(HERE, '..', '..', 'harness', 'setup3', 'run.py')
GA = 'l1tex__t_sectors.sum,l1tex__t_sectors_pipe_lsu.sum,l1tex__t_sectors_lookup_hit.sum,l1tex__t_sectors_pipe_lsu_lookup_hit.sum'
GB = ('l1tex__m_xbar2l1tex_read_sectors.sum,lts__t_sectors_srcunit_tex_op_read.sum,lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum,'
      'dram__bytes_read.sum,sm__cycles_elapsed.max')
SCENES = ['100000', '3000000']
ORDERS = os.environ.get('MEM4_ORDERS', 'grouped,interleaved').split(',')   # MEM4_ORDERS=grouped reruns one order
BVARS = [('grouped', 'warp'), ('grouped', 'ray'), ('interleaved', 'ray')]
STATES = {'warm': ['--passes', '2'], 'cold': ['--flush', '256']}


def go(binp, work, n, rays, out, w, extra):
    cmd = [sys.executable, HARNESS, binp, 'file:' + os.path.join(work, 's' + n, rays), 'zero', out, '--scene', 'soup:' + n,
           '--warps', str(w)] + extra
    r = subprocess.run(cmd, capture_output=True, text=True)
    ok = r.returncode == 0 and 'exit 0' in r.stdout and ' error' not in r.stdout.replace('no error', '')
    print(os.path.basename(out), 'OK' if ok else 'FAILED', flush=True)
    if not ok:
        sys.exit(r.stdout[-3000:] + r.stderr[-3000:])


def main(b0, b1, work, what='all'):
    for n in SCENES:
        d = os.path.join(work, 's' + n)
        if what in ('timing', 'all'):
            os.makedirs(os.path.join(d, 't'), exist_ok=True)
            for w in (1, 4, 8, 16):
                for st, ex in STATES.items():
                    for o in ORDERS:
                        go(b1, work, n, 'rays_%s.bin' % o, os.path.join(d, 't', '%s_w%d_%s.json' % (o, w, st)), w, ['--resume', '16'] + ex)
        if what in ('b-timing', 'b'):
            os.makedirs(os.path.join(d, 't'), exist_ok=True)
            for w in (1, 4, 8, 16):
                for st, ex in STATES.items():
                    for o, mode in BVARS:
                        sf = ['--start-file', os.path.join(d, 'starts_%s_%s.txt' % (o, mode))]
                        go(b1, work, n, 'rays_%s.bin' % o, os.path.join(d, 't', 'b_%s_%s_w%d_%s.json' % (o, mode, w, st)), w, ['--resume', '16'] + ex + sf)
        if what in ('b-ncu', 'b'):
            os.makedirs(os.path.join(d, 'n'), exist_ok=True)
            for w in (4, 8, 16):
                for st, ex in STATES.items():
                    for g, m in (('A', GA), ('B', GB)):
                        for o, mode in BVARS:
                            sf = ['--start-file', os.path.join(d, 'starts_%s_%s.txt' % (o, mode))]
                            base = os.path.join(d, 'n', 'b_%s_%s_w%d_%s_%s' % (o, mode, w, st, g))
                            go(b1, work, n, 'rays_%s.bin' % o, base + '.json', w, ['--resume', '16'] + ex + sf + ['--ncu', m, '--ncu-out', base + '.csv'])
        if what in ('ncu', 'all'):
            os.makedirs(os.path.join(d, 'n'), exist_ok=True)
            for w in (4, 8, 16):
                for st, ex in STATES.items():
                    for g, m in (('A', GA), ('B', GB)):
                        if not os.environ.get('MEM4_NOBASE'):   # MEM4_NOBASE=1: keep the existing stage 0 baselines
                          go(b0, work, n, 'rays_interleaved.bin', os.path.join(d, 'n', 'base_w%d_%s_%s.json' % (w, st, g)), w,
                             ex + ['--ncu', m, '--ncu-out', os.path.join(d, 'n', 'base_w%d_%s_%s.csv' % (w, st, g))])
                        for o in ORDERS:
                            base = os.path.join(d, 'n', '%s_w%d_%s_%s' % (o, w, st, g))
                            go(b1, work, n, 'rays_%s.bin' % o, base + '.json', w, ['--resume', '16'] + ex + ['--ncu', m, '--ncu-out', base + '.csv'])


if __name__ == '__main__':
    main(*sys.argv[1:])
