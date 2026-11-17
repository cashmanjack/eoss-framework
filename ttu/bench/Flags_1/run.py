"""Flags_1 runner (predictions: model/predictions_flags.md). OptiX launches only, no TTU code.

    run.py WORK STEP     STEP: t1 | t2 | t3 | t4 | t5 | t6 | all     (WORK = private raw data directory, e.g. notes/flags1)

Per configuration: (a) the app with probe launches (32 rays, 100 launches, 3 traces each) and 20 timed launches of 262,144 rays (/dev/shm/flags1,
/dev/shm/flags1_dev.ptx), (b) the same configuration with bigonly=1 under ncu (kernel optixLaunch, 2 launches, the second is read).
Raw files get a timestamp in the name and are never overwritten (the app opens with mode wx, ncu logs are checked here).
Needs: LD_LIBRARY_PATH to the extracted OptiX library when the system one is missing, TMPDIR an own directory (ncu), CUDA_VISIBLE_DEVICES set here to the idle GPU.
"""
import csv, json, os, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'config')); import gpucfg  # noqa: E402
CFG = gpucfg.load(); CFG.check_idle()
ENV = CFG.env(dict(os.environ, TMPDIR=os.environ.get('TMPDIR', '/dev/shm/ncu_tmp_' + os.environ.get('USER', 'u'))))
os.makedirs(ENV['TMPDIR'], exist_ok=True)
EXE, PTX = '/dev/shm/flags1', '/dev/shm/flags1_dev.ptx'
NCU = os.path.expanduser('~/opt/cuda-12.8/bin/ncu')
METRICS = 'l1tex__t_sectors.sum,l1tex__t_sectors_pipe_lsu.sum,l1tex__m_xbar2l1tex_read_sectors.sum,smsp__thread_inst_executed.sum,smsp__inst_executed.sum,gpu__time_duration.sum'
TS = time.strftime('%Y%m%dT%H%M%S')

DISABLE_ANYHIT, ENFORCE_ANYHIT, TERMINATE, CULL_BACK, CULL_FRONT = 1, 2, 4, 16, 32

CONFIGS = {
    't1': [('t1_none', 'scene=layers:64 gopaque=1 rflags=0'), ('t1_term', 'scene=layers:64 gopaque=1 rflags=4')],
    't1s': [('t1s_none', 'scene=soup:100000 gopaque=1 rflags=0'), ('t1s_term', 'scene=soup:100000 gopaque=1 rflags=4')],
    't2': [('t2_noah', 'scene=layers:16 ah=0 gopaque=1 rflags=0'), ('t2_ah_none', 'scene=layers:16 ah=1 gopaque=0 rflags=0'), ('t2_ah_disable', 'scene=layers:16 ah=1 gopaque=0 rflags=1'),
           ('t2_ah_enforce', 'scene=layers:16 ah=1 gopaque=0 rflags=2'), ('t2_geom_opaque', 'scene=layers:16 ah=1 gopaque=1 rflags=0')],
    't3': [('t3_none', 'scene=layers:16:alt gopaque=1 rflags=0'), ('t3_cull_back', 'scene=layers:16:alt gopaque=1 rflags=16'), ('t3_cull_front', 'scene=layers:16:alt gopaque=1 rflags=32'),
           ('t3_cull_both', 'scene=layers:16:alt gopaque=1 rflags=48')],
    't4': [('t4_C_one_instance', 'scene=ias:64:C gopaque=1 rmask=2'), ('t4_D_none_masked', 'scene=ias:64:D gopaque=1 rmask=2'), ('t4_A_63_masked', 'scene=ias:64:A gopaque=1 rmask=2'),
           ('t4_B_nearest_visible', 'scene=ias:64:A gopaque=1 rmask=255')],
    't6': [('t6_%s' % m, 'scene=ias:256:%s raymiss=1 gopaque=1 rmask=1' % m) for m in ('V', 'R16', 'L16', 'E', 'Q')],
    't5': [('t5_tmax_%g' % r, 'scene=soup:100000 gopaque=1 tmax=%g' % r) for r in (0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 1e16)],
}


def run_cfg(work, name, args):
    base = os.path.join(work, '%s_%s' % (name, TS))
    if any(os.path.exists(base + e) for e in ('.json', '.ncu.csv', '.ncu32.csv')): sys.exit('refuse to overwrite ' + base)
    CFG.check_idle()
    p = subprocess.run(['timeout', '-k', '2', '60', EXE, PTX, base + '.json'] + args.split() + ['nrep=3'], env=ENV, capture_output=True, text=True)
    if p.returncode: sys.exit('STOP %s exit %d\n%s%s' % (name, p.returncode, p.stdout[-1500:], p.stderr[-1500:]))
    q = subprocess.run(['timeout', '-k', '2', '120', NCU, '--target-processes', 'all', '--cache-control', 'none', '--clock-control', 'none', '-k', 'regex:optixLaunch', '--metrics', METRICS,
                        '--csv', '--page', 'raw', '--log-file', base + '.ncu.csv', EXE, PTX, base + '.bigonly.json'] + args.split() + ['bigonly=1'], env=ENV, capture_output=True, text=True)
    if q.returncode: sys.exit('STOP ncu %s exit %d\n%s%s' % (name, q.returncode, q.stdout[-1000:], q.stderr[-1000:]))
    # one warp launch under ncu (32 rays, one SM, L1 cold at the start): fills = sectors the launch fetched from L2 (ray words 32 and the BVH lines)
    q = subprocess.run(['timeout', '-k', '2', '120', NCU, '--target-processes', 'all', '--cache-control', 'none', '--clock-control', 'none', '-k', 'regex:optixLaunch', '--metrics', METRICS,
                        '--csv', '--page', 'raw', '--log-file', base + '.ncu32.csv', EXE, PTX, base + '.big32.json'] + args.split() + ['bigonly=1', 'big=32'], env=ENV, capture_output=True, text=True)
    if q.returncode: sys.exit('STOP ncu32 %s exit %d\n%s%s' % (name, q.returncode, q.stdout[-1000:], q.stderr[-1000:]))
    return base


def summarize(base):
    j = json.load(open(base + '.json'))
    c = np.array(j['probe_clk'], dtype=np.int64).reshape(j['probe_launches'], 32, 8)
    tr = [np.median(c[:, :, 2 + 2 * k].max(axis=1) - c[:, :, 1 + 2 * k].min(axis=1)) if j['nrep'] > k else 0 for k in range(3)]
    rows = [r for r in csv.reader(l for l in open(base + '.ncu.csv') if l.startswith('"'))]
    h, last = rows[0], rows[-1]
    g = lambda m: float(last[h.index(m)].replace(',', ''))
    n = j['big_n']
    rows32 = [r for r in csv.reader(l for l in open(base + '.ncu32.csv') if l.startswith('"'))]
    h32, last32 = rows32[0], rows32[-1]
    g32 = lambda m: float(last32[h32.index(m)].replace(',', ''))
    wo = np.array(j['warp_out'])
    return dict(trace0=tr[0], trace1=tr[1], trace2=tr[2], hit=float(wo[:, 0].mean()), prim=int(wo[0, 2]), ah=float(wo[:, 3].mean()), front=int(wo[0, 4]),
                big_ns_per_ray=1e6 * float(np.median(j['big_ms'])) / n, hit_frac=j['big_hit_frac'], ah_big=j['big_ah_mean'], nearest_frac=j['big_nearest_frac'],
                rt_sectors=(g('l1tex__t_sectors.sum') - g('l1tex__t_sectors_pipe_lsu.sum')) / n, fills=g('l1tex__m_xbar2l1tex_read_sectors.sum') / n,
                fills32=g32('l1tex__m_xbar2l1tex_read_sectors.sum'), thread_inst32=g32('smsp__thread_inst_executed.sum'), thread_inst=g('smsp__thread_inst_executed.sum') / n, warp_inst=g('smsp__inst_executed.sum') / n, ncu_us=g('gpu__time_duration.sum') / 1000.0)


if __name__ == '__main__':
    work, step = sys.argv[1], sys.argv[2]
    os.makedirs(work, exist_ok=True)
    for t in (['t1', 't1s', 't2', 't3', 't4', 't5'] if step == 'all' else [step]):
        for name, args in CONFIGS[t]:
            base = run_cfg(work, name, args)
            s = summarize(base); s['name'] = name
            print(' ', {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()}, flush=True)
            open(base + '.summary.json', 'x').write(json.dumps(s))
