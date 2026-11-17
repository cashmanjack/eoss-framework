"""Omm_1 runner (OptiX only, no TTU code). run.py WORK TAG [ncu] key=value ...   needs /dev/shm/om1/{omm, omm_dev.ptx}.
Raw files WORK/TAG_<TS>.json (+ .ncu.csv), never overwritten. GPU from gpucfg (we use TTU_GPU=1), idle check first."""
import os, subprocess, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'config'))
import gpucfg
CFG = gpucfg.load(); ENV = CFG.env()
W, TAG = sys.argv[1], sys.argv[2]; rest = sys.argv[3:]; DO_NCU = bool(rest) and rest[0] == 'ncu'
if DO_NCU: rest = rest[1:]
TS = time.strftime('%Y%m%dT%H%M%S'); D = '/dev/shm/om1'; os.makedirs(W, exist_ok=True)
METRICS = 'smsp__inst_executed.sum,smsp__thread_inst_executed.sum,l1tex__t_sectors.sum,gpu__time_duration.sum'
base = os.path.join(W, '%s_%s' % (TAG, TS))
if any(os.path.exists(base + e) for e in ('.json', '.ncu.csv')): sys.exit('refuse to overwrite ' + base)
CFG.check_idle()
args = [a.replace('@BASE', base).replace('@RES', '/dev/shm/om1/res/' + os.path.basename(base) + '.res') for a in rest]
p = subprocess.run(['timeout', '-k', '2', '180', D + '/omm', D + '/omm_dev.ptx', base + '.json'] + args, env=ENV, capture_output=True, text=True)
print(TAG, 'rc', p.returncode, p.stdout.strip()[-80:], p.stderr[-300:], flush=True)
if p.returncode: sys.exit('STOP')
if DO_NCU:
    a2 = [x for x in args if not x.startswith(('dumpres=', 'count=', 'reps='))]
    tmp = '/dev/shm/om1/ncutmp'; os.makedirs(tmp, exist_ok=True)
    q = subprocess.run(['timeout', '-k', '2', '300', 'ncu', '--target-processes', 'all', '--cache-control', 'none', '--clock-control', 'none', '-k', 'regex:optixLaunch', '-s', '1', '-c', '1', '--metrics', METRICS, '--csv', '--page', 'raw',
                        '--log-file', base + '.ncu.csv', D + '/omm', D + '/omm_dev.ptx', base + '.ncurun.json'] + a2 + ['count=0', 'reps=1'], env=dict(ENV, TMPDIR=tmp), capture_output=True, text=True)
    print(TAG, 'ncu rc', q.returncode, q.stderr[-200:], flush=True)
    if q.returncode: sys.exit('STOP ncu')
