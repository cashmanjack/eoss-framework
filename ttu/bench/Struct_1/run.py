"""Struct_1 runner (OptiX only, no TTU code). run.py WORK TAG [ncu] key=value ...   (host args; scene, bf, upd, comp, inst, rays, ...)
Needs /dev/shm/st1/{struct, struct_dev.ptx}. GPU from TTU_GPU (we use 1), idle check first. Raw files WORK/TAG_<TS>.json (+ .ncu.csv with 'ncu'), never overwritten."""
import os, subprocess, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'config'))
import gpucfg
CFG = gpucfg.load(); ENV = CFG.env()
W, TAG = sys.argv[1], sys.argv[2]; rest = sys.argv[3:]; DO_NCU = bool(rest) and rest[0] == 'ncu'
if DO_NCU: rest = rest[1:]
TS = time.strftime('%Y%m%dT%H%M%S'); D = '/dev/shm/st1'; os.makedirs(W, exist_ok=True); os.makedirs(D + '/dumps', exist_ok=True)
METRICS = 'l1tex__t_sectors.sum,l1tex__t_sectors_pipe_lsu.sum,l1tex__m_xbar2l1tex_read_sectors.sum,smsp__thread_inst_executed.sum,smsp__inst_executed.sum,gpu__time_duration.sum'
base = os.path.join(W, '%s_%s' % (TAG, TS))
if any(os.path.exists(base + e) for e in ('.json', '.ncu.csv')): sys.exit('refuse to overwrite ' + base)
CFG.check_idle()
args = []
for a in rest: args.append(a.replace('@BASE', base).replace('@DUMP', '/dev/shm/st1/dumps/%s_%s' % (TAG, TS)))
p = subprocess.run(['timeout', '-k', '2', '120', D + '/struct', D + '/struct_dev.ptx', base + '.json'] + args, env=ENV, capture_output=True, text=True)
print(TAG, 'rc', p.returncode, p.stdout.strip()[-80:], p.stderr[-300:], flush=True)
if p.returncode: sys.exit('STOP')
if DO_NCU:
    a2 = [x for x in args if not x.startswith(('dump=', 'dumpres=', 'probe='))]
    tmp = '/dev/shm/st1/ncutmp'; os.makedirs(tmp, exist_ok=True)
    q = subprocess.run(['timeout', '-k', '2', '300', 'ncu', '--target-processes', 'all', '--cache-control', 'none', '--clock-control', 'none', '-k', 'regex:optixLaunch', '--metrics', METRICS, '--csv', '--page', 'raw',
                        '--log-file', base + '.ncu.csv', D + '/struct', D + '/struct_dev.ptx', base + '.bigonly.json'] + a2 + ['bigonly=1'], env=dict(ENV, TMPDIR=tmp), capture_output=True, text=True)
    print(TAG, 'ncu rc', q.returncode, q.stderr[-200:], flush=True)
    if q.returncode: sys.exit('STOP ncu')
