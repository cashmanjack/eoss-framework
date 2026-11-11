"""ncu comparison of our persistent loop kernel and OptiX's launch kernel on the same rays and scene (soup, G = 142 blocks, R rounds per lane).

    ncu_compare.py WORK SCENE R [OUT_PREFIX]

Ours: build_patch --outer R (see run.py), stage 0 run first (no TTU), then stage 1 under ncu through setup3/run.py (--ncu, -k ttu_one, one CSV per metric group).
OptiX: the same process family with --no-ttu (OptiX reference launches only, no TTU code of ours) under ncu, kernels chosen by name.
Raw metrics, caches and clocks left alone; the last profiled launch of each kernel is used (the first one is cold).
"""
import csv, os, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import long_run as T  # noqa: E402

NCU = os.path.expanduser('~/opt/cuda-12.8/bin/ncu')
LAUNCH = ('launch__registers_per_thread,launch__block_size,launch__grid_size,launch__waves_per_multiprocessor,launch__occupancy_limit_registers,launch__occupancy_limit_warps,'
          'launch__occupancy_limit_shared_mem,launch__occupancy_limit_blocks,launch__shared_mem_per_block_static,launch__shared_mem_per_block_dynamic,launch__stack_size,'
          'sm__maximum_warps_per_active_cycle_pct,sm__warps_active.avg.per_cycle_active,sm__warps_active.avg.pct_of_peak_sustained_active,sm__cycles_active.avg,sm__cycles_active.sum,'
          'sm__cycles_elapsed.avg,gpu__time_duration.sum,smsp__warps_launched.sum,smsp__inst_executed.sum,smsp__issue_active.avg.pct_of_peak_sustained_active')
STALL1 = ','.join('smsp__warps_issue_stalled_%s' % s for s in ('long_scoreboard', 'long_scoreboard_pipe_l1tex', 'short_scoreboard', 'wait', 'barrier', 'branch_resolving',
                                                               'no_instruction', 'selected', 'not_selected'))
STALL2 = ','.join('smsp__warps_issue_stalled_%s' % s for s in ('math_pipe_throttle', 'lg_throttle', 'mio_throttle', 'misc', 'drain', 'imc_miss', 'dispatch_stall', 'sleeping',
                                                               'tex_throttle', 'membar', 'mio_throttle_pipe_mio'))
MEM = ('l1tex__t_sectors.sum,l1tex__t_sectors_pipe_lsu.sum,l1tex__m_xbar2l1tex_read_sectors.sum,lts__t_sectors.sum,dram__bytes_read.sum,dram__bytes_write.sum,'
       'l1tex__t_sector_hit_rate.pct,lts__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__lsu_writeback_active.avg.pct_of_peak_sustained_active')
GROUPS = [('launch', LAUNCH), ('stall1', STALL1), ('stall2', STALL2), ('mem', MEM)]


def parse(path):
    hdr = None; rows = []
    for l in open(path):
        if l.startswith('"ID"'): hdr = next(csv.reader([l]))
        elif hdr and l.startswith('"'): rows.append(dict(zip(hdr, next(csv.reader([l])))))
    return rows


def ncu_optix(binp, rays, scene, metrics, csv_out, name_re):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='1', LD_LIBRARY_PATH=os.path.expanduser('~/opt/nvoptix-580.178.04') + ':' + os.environ.get('LD_LIBRARY_PATH', ''))
    ptx = os.path.join(os.path.dirname(binp), 'setup3_dev.ptx')
    cmd = ['timeout', '-k', '2', '60', NCU, '--target-processes', 'all', '-k', 'regex:' + name_re, '--cache-control', 'none', '--clock-control', 'none', '--metrics', metrics,
           '--csv', '--page', 'raw', '--log-file', csv_out, binp, ptx, os.path.join(os.path.dirname(binp), 'words.txt'), 'file:' + rays, 'zero', csv_out + '.json',
           '--scene', scene, '--warps', '28', '--blockrays', '--slim', '--loop', '--blocks', str(T_G), '--outer', str(T_R), '--no-ttu', '--quick', '--optix-repeat', '1']
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    return r


T_G = 142
T_R = 10


def num(s):
    try: return float(s.replace(',', ''))
    except Exception: return float('nan')


def metrics_of(path, pick):
    """metric -> value of the row chosen by pick(rows)."""
    rows = parse(path)
    r = pick(rows)
    return {k: num(v) for k, v in r.items() if k and (k.startswith(('launch__', 'sm__', 'smsp__', 'gpu__', 'l1tex__', 'lts__', 'dram__')))}, r.get('Kernel Name', '')


def ours(work, scene, R, lean=False, prefix='ours'):
    G = 142
    n = G * T.LANES * R
    rays = '/dev/shm/t8/nc_rays.bin'; T.make_rays(n).tofile(rays)
    ex = ['--lean'] if lean else []
    b0, b1 = '/dev/shm/t8/nc_s0', '/dev/shm/t8/nc_s1'
    T.build(0, G, R, b0, ex); T.build(1, G, R, b1, ex)
    base = ['--scene', scene, '--warps', '28', '--blockrays', '--slim', '--loop', '--blocks', str(G), '--outer', str(R)] + (['--lean'] if lean else [])
    T.go(b0, rays, '/dev/shm/t8/nc_s0.json', scene, G, R, ['--repeat', '1', '--quick'] + (['--lean'] if lean else []))
    res = {}
    for name, m in GROUPS:
        csvf = os.path.join(work, '%s_%s.csv' % (prefix, name))
        cmd = [sys.executable, os.path.join(T.S3, 'run.py'), b1, 'file:' + rays, 'zero', '/dev/shm/t8/nc_s1.json'] + base + ['--repeat', '2', '--quick', '--ncu', m, '--ncu-out', csvf]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if 'exit 0' not in p.stdout: sys.exit(p.stdout[-1500:] + p.stderr[-1500:])
        v, kn = metrics_of(csvf, lambda rows: rows[-1]); res.update(v)
        print('  ours', name, 'ok', kn[:30], flush=True)
    for f in (rays, b0, b1, b0 + '.checked', b1 + '.checked', b0 + '.layout.json', b1 + '.layout.json'):
        if os.path.exists(f): os.remove(f)
    return res, n


def optix(work, scene, R, name_re, prefix='optix'):
    G = 142
    n = G * T.LANES * R
    rays = '/dev/shm/t8/nc_rays.bin'; T.make_rays(n).tofile(rays)
    b0 = '/dev/shm/t8/nc_s0'; T.build(0, G, R, b0)     # a checked binary; --no-ttu stops after the OptiX reference, no TTU code runs
    res = {}
    for name, m in GROUPS:
        csvf = os.path.join(work, '%s_%s.csv' % (prefix, name))
        r = ncu_optix(b0, rays, scene, m, csvf, name_re)
        if not os.path.exists(csvf) or not parse(csvf): sys.exit('no ncu rows for OptiX: ' + (r.stdout[-1500:] + r.stderr[-800:]))
        v, kn = metrics_of(csvf, lambda rows: rows[-1]); res.update(v)
        print('  optix', name, 'ok', kn[:40], len(parse(csvf)), 'rows', flush=True)
    for f in (rays, b0, b0 + '.checked', b0 + '.layout.json'):
        if os.path.exists(f): os.remove(f)
    return res, n


if __name__ == '__main__':
    import json
    work, scene, R = sys.argv[1], sys.argv[2], int(sys.argv[3])
    T_R = R
    what = sys.argv[4] if len(sys.argv) > 4 else 'both'
    os.makedirs(work, exist_ok=True)
    out = {}
    if what in ('both', 'ours'): out['ours'] = ours(work, scene, R)[0]
    if what == 'lean': out['ours_lean'] = ours(work, scene, R, lean=True, prefix='lean')[0]
    if what in ('both', 'optix'): out['optix'] = optix(work, scene, R, sys.argv[5] if len(sys.argv) > 5 else 'optix')[0]
    json.dump(out, open(os.path.join(work, 'ncu_%s_%s.json' % (scene.replace(':', ''), what)), 'w'), indent=1)
