"""Throughput_6 with the slim harness (build_patch --slim): random rays only, no register image stores, no clock stores.

    run_slim.py BIN_S0 BIN_S1 WORK [timing|ncu|all]       BIN: --slim --sync --remap --compact-out --regs 64 --max-out 61
    /usr/bin/python3 run_slim.py --report WORK

Same G blocks of 32 warps, one block per SM, same scenes and rays as run.py (soup:100000, soup:3000000; a different
random ray in every lane, --blockrays), first query only. Time per launch = CUDA events (kernel only, 6 launches,
first dropped). aggregate = G x 1,024 rays / time, relative to G = 1. Also ncu (stage 1 minus stage 0 with the same
slim loads and stores): RT core L1 fills, L2 reads and hits, DRAM read bytes per ray, L2 and DRAM utilization.
"""
import json, os, statistics as st, struct, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.join(HERE, '..', '..', 'harness', 'setup3', 'run.py')
sys.path.insert(0, os.path.join(HERE, '..', 'Memory_4'))
from prep import make_rays  # noqa: E402
import ncu_parse  # noqa: E402

GS = [1, 2, 4, 8, 16, 24, 32, 48, 64, 71, 72, 80, 100, 120, 142]
NCU_GS = [1, 32, 64, 142]
SCENES = [('soup1e5', 'soup:100000'), ('soup3e6', 'soup:3000000')]
W = 32
GB = ('l1tex__m_xbar2l1tex_read_sectors.sum,lts__t_sectors_srcunit_tex_op_read.sum,lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum,'
      'dram__bytes_read.sum,sm__cycles_elapsed.max')
GC = 'gpu__time_duration.sum,lts__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,lts__t_sectors.sum,dram__bytes_write.sum'


def go(binp, rays, out, scene, g, extra=()):
    p = subprocess.run([sys.executable, HARNESS, binp, rays, 'zero', out, '--scene', scene, '--warps', str(W), '--blocks', str(g), '--blockrays', '--slim', '--repeat', '6'] + list(extra),
                       capture_output=True, text=True)
    ok = 'exit 0' in p.stdout and ' error' not in p.stdout.replace('no error', '') and ('%d block(s) ran' % g) in p.stdout
    print(os.path.basename(out), 'OK' if ok else 'FAILED', flush=True)
    if not ok:
        sys.exit(p.stdout[-2500:] + p.stderr[-1500:])


def run(b0, b1, work, what='all'):
    os.makedirs(work, exist_ok=True)
    big = make_rays(142 * W * 32)
    for g in GS:
        big[:g * W * 32].tofile(os.path.join(work, 'rays_%d.bin' % g))
    if what in ('timing', 'all'):
        go(b0, 'file:' + os.path.join(work, 'rays_142.bin'), os.path.join(work, 's0_full.json'), 'soup:100000', 142)   # stage 0: every block must run
        for name, scene in SCENES:
            for g in GS:
                go(b1, 'file:' + os.path.join(work, 'rays_%d.bin' % g), os.path.join(work, '%s_g%d.json' % (name, g)), scene, g)
    if what in ('ncu', 'all'):
        for name, scene in SCENES:
            for g in NCU_GS:
                for tag, m in (('B', GB), ('C', GC)):
                    for st_, b in (('0', b0), ('1', b1)):
                        o = os.path.join(work, 'n%s_%s_g%d_%s' % (st_, name, g, tag))
                        go(b, 'file:' + os.path.join(work, 'rays_%d.bin' % g), o + '.json', scene, g, ['--ncu', m, '--ncu-out', o + '.csv'])


def report(work):
    f = lambda u: struct.unpack('<f', struct.pack('<I', u))[0]
    print('time per launch from CUDA events (median of 5); aggregate = G x 1,024 rays / time, relative to G = 1')
    print('%-8s | %3s | %9s %12s %8s | %s' % ('scene', 'G', 'ms', 'rays/us', 'x 1 SM', 'checks'))
    for name, scene in SCENES:
        base = None
        for g in GS:
            p = os.path.join(work, '%s_g%d.json' % (name, g))
            if not os.path.exists(p):
                continue
            j = json.load(open(p)); ms = st.median(l['ms'] for l in j['launches'][1:]); rate = g * 1024 / ms / 1000
            if base is None: base = rate
            ref = j['ref']; bad = cnt = 0
            for k, wo in enumerate(j['warp_out']):
                pend = any(wo[x] for x in (56, 57, 58, 59, 60, 61, 70, 71))
                if pend: continue
                rr = ref[32 * k]; cnt += 1; hit = wo[3]
                ok = (hit == 0 and rr['hit'] == 0) or (hit == 1 and rr['hit'] == 1 and (wo[44] & 0x1fffffff) == rr['prim'] and abs(-f(wo[45]) - f(int(rr['t'], 16))) <= 1e-6 * max(1, f(int(rr['t'], 16))))
                bad += not ok
            print('%-8s | %3d | %9.4f %12.1f %8.1f | %d lanes checked, %d wrong' % (name, g, ms, rate, rate / base, cnt, bad))
    print('\nncu: RT core traffic per ray = stage 1 minus stage 0 (same slim loads and stores), last launch')
    print('%-8s | %3s | %8s | %9s %8s %8s %10s | %9s %9s' % ('scene', 'G', 'time us', 'L1 fills', 'L2 reads', 'L2 hit', 'DRAM B', 'L2 util', 'DRAM util'))
    for name, scene in SCENES:
        for g in NCU_GS:
            try:
                r = {}
                for tag in ('B', 'C'):
                    for s_ in ('0', '1'):
                        r[(tag, s_)] = ncu_parse.rows(os.path.join(work, 'n%s_%s_g%d_%s.csv' % (s_, name, g, tag)))[-1]
            except Exception as e:
                continue
            n = g * 1024
            sub = lambda t, k: r[(t, '1')][k] - r[(t, '0')][k]
            l2r = sub('B', 'lts__t_sectors_srcunit_tex_op_read.sum')
            print('%-8s | %3d | %8.1f | %9.1f %8.1f %7.1f%% %10.0f | %8.1f%% %8.1f%%' % (
                name, g, r[('C', '1')]['gpu__time_duration.sum'] / 1000, sub('B', 'l1tex__m_xbar2l1tex_read_sectors.sum') / n, l2r / n,
                100 * sub('B', 'lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum') / l2r if l2r else float('nan'), sub('B', 'dram__bytes_read.sum') / n,
                r[('C', '1')]['lts__throughput.avg.pct_of_peak_sustained_elapsed'], r[('C', '1')]['dram__throughput.avg.pct_of_peak_sustained_elapsed']))


if __name__ == '__main__':
    report(sys.argv[2]) if sys.argv[1] == '--report' else run(*sys.argv[1:5])
