"""Plateau check (b): Memory_4 grouped vs interleaved at plateau points, with the loop slim kernel (RT traffic only).

    plateau_check.py BIN_S0_PLAIN BIN_LOOP_S0 BIN_LOOP_S1 WORK

BIN_S0_PLAIN: any checked stage 0 binary (prep.py's OptiX only reference run).
BIN_LOOP_*: build_patch --slim --sync --remap --compact-out --regs 72 --max-out 61 --loop 32.
Rays: prep.py (seed 2026) N = G x 896 rays, written in two orders (grouped by the hit subtree key = an oracle, interleaved = random).
Points: soup:100000 (6 MB, L2 resident) and soup:3000000 (186 MB) at G = 48 and 140 blocks of 28 warps, whole rays.
Per point and order: CUDA events (6 launches, first dropped) and ncu (stage 1 minus stage 0 scaled by the LSU sector ratio: stage 0 always runs all K rounds of reloads):
RT L1 sectors, L1 fills from L2, L2 hit, DRAM bytes per ray, whole kernel L2 sectors per ray and L2 / DRAM utilization.
"""
import csv, gzip, io, json, os, statistics as st, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.join(HERE, '..', '..', 'harness', 'setup3', 'run.py')
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'Throughput_7b'))
import prep  # noqa: E402
from check_final import main as check  # noqa: E402

WARPS, K, LANES = 28, 32, 28 * 32
SCENES = [('soup1e5', 'soup:100000'), ('soup3e6', 'soup:3000000')]
GS = [48, 140]   # G x 896 rays must be a multiple of 512 (prep.py's reference run uses 16 warps)
ORDERS = ['interleaved', 'grouped']
GB = ('l1tex__t_sectors.sum,l1tex__t_sectors_pipe_lsu.sum,l1tex__m_xbar2l1tex_read_sectors.sum,lts__t_sectors_srcunit_tex_op_read.sum,'
      'lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum,dram__bytes_read.sum,lts__t_sectors.sum,sm__cycles_elapsed.max')
GC = 'gpu__time_duration.sum,lts__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed'


def go(binp, rays, out, scene, g, extra):
    p = subprocess.run([sys.executable, HARNESS, binp, rays, 'zero', out, '--scene', scene, '--warps', str(WARPS), '--blocks', str(g), '--blockrays', '--slim', '--loop'] + extra,
                       capture_output=True, text=True)
    ok = 'exit 0' in p.stdout and ' error' not in p.stdout.replace('no error', '')
    print(os.path.basename(out), 'OK' if ok else 'FAILED', flush=True)
    if not ok:
        sys.exit(p.stdout[-2500:] + p.stderr[-1500:])


def run(b_plain, b0, b1, work):
    os.makedirs(work, exist_ok=True)
    for name, scene in SCENES:
        for g in GS:
            pd = os.path.join(work, 'prep_%s_%d' % (name, g))
            if not os.path.exists(os.path.join(pd, 'prep.json')):
                prep.main(b_plain, pd, scene, str(g * LANES))
            tris = os.path.join(pd, 'ref.tris')
            for order in ORDERS:
                f = os.path.join(work, 'rays_%s_%s_%d.bin' % (name, order, g))
                np.fromfile(os.path.join(pd, 'rays_%s.bin' % order), np.float32).reshape(-1, 8)[:g * LANES].tofile(f)
                if g == GS[0] and order == ORDERS[0]:
                    go(b0, 'file:' + f, os.path.join(work, 's0_%s.json' % name), scene, g, ['--repeat', '1'])
                o = os.path.join(work, '%s_%s_g%d' % (name, order, g))
                go(b1, 'file:' + f, o + '.json', scene, g, ['--repeat', '6', '--dump-final', o + '.bin'])
                check(o + '.bin', f, tris, K, 32)
                for tag, m in (('B', GB), ('C', GC)):
                    for stage, b in ((0, b0), (1, b1)):
                        go(b, 'file:' + f, '%s_n%d%s.json' % (o, stage, tag), scene, g, ['--repeat', '2', '--ncu', m, '--ncu-out', '%s_n%d%s.csv' % (o, stage, tag)])


def ncu_last(path):
    """metric -> value of the last kernel row of an ncu --csv --page raw file."""
    lines = [l for l in open(path) if l.startswith('"') and 'ID' not in l.split(',')[0]]
    hdr = None; rows = []
    for l in open(path):
        if l.startswith('"ID"'):
            hdr = next(csv.reader([l]))
        elif hdr and l.startswith('"'):
            rows.append(dict(zip(hdr, next(csv.reader([l])))))
    r = rows[-1]
    num = lambda s: float(s.replace(',', '')) if s.replace(',', '').replace('.', '', 1).replace('e+', '').replace('-', '').isdigit() else float('nan')
    return {k: num(v) for k, v in r.items() if k and not k.startswith(('ID', 'Process', 'Kernel', 'Context', 'Stream', 'Block', 'Grid', 'Device', 'CC', 'Section', 'Metric'))}


def report(work):
    print('%-8s %3s %-11s | %9s %9s | %8s %8s %6s %8s | %9s %8s %6s %6s | %5s' % ('scene', 'G', 'order', 'ms', 'rays/us', 'L1 sect', 'L2 rd', 'L2hit', 'DRAM B', 'L2 sect', 'RT share', 'L2 %', 'DRAM %', 'sect/cyc'))
    for name, scene in SCENES:
        for g in GS:
            for order in ORDERS:
                o = os.path.join(work, '%s_%s_g%d' % (name, order, g))
                try:
                    j = json.load(open(o + '.json')); ms = st.median(l['ms'] for l in j['launches'][1:])
                    n = g * LANES
                    b1, b0 = ncu_last(o + '_n1B.csv'), ncu_last(o + '_n0B.csv'); c1 = ncu_last(o + '_n1C.csv')
                except Exception as e:
                    print(name, g, order, 'missing', e); continue
                kk = b1['l1tex__t_sectors_pipe_lsu.sum'] / b0['l1tex__t_sectors_pipe_lsu.sum']   # stage 0 runs all K rounds of loads: scale its traffic by the LSU sector ratio
                d = lambda k: (b1[k] - kk * b0[k]) / n
                l2 = d('lts__t_sectors_srcunit_tex_op_read.sum'); hit = d('lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum')
                tot = b1['lts__t_sectors.sum'] / n
                print('%-8s %3d %-11s | %9.4f %9.0f | %8.1f %8.1f %5.0f%% %8.0f | %9.1f %7.0f%% %6.0f %6.0f | %5.1f' % (
                    name, g, order, ms, n / ms / 1000, (b1['l1tex__t_sectors.sum'] - b1['l1tex__t_sectors_pipe_lsu.sum']) / n, l2, 100 * hit / l2 if l2 else float('nan'),
                    d('dram__bytes_read.sum'), tot, 100 * l2 / tot, c1['lts__throughput.avg.pct_of_peak_sustained_elapsed'], c1['dram__throughput.avg.pct_of_peak_sustained_elapsed'], b1['lts__t_sectors.sum'] / b1['sm__cycles_elapsed.max']))


if __name__ == '__main__':
    report(sys.argv[2]) if sys.argv[1] == '--report' else run(*sys.argv[1:5])
