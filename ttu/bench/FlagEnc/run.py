"""Flag / mask / tmax encoding study: one memdump process per variant (own OptiX launch, no TTU code of ours).
Each variant: compile PTX with -D constants, memdump read --fn --fnrel -0x10000 --len 0x10000 (kernel method),
raw window to /dev/shm/flagenc (NVIDIA code, private, never git). GPU from gpucfg (set TTU_GPU)."""
import os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
OD = os.path.join(HERE, '..', '..', 'optix_dump')
sys.path.insert(0, os.path.join(HERE, '..', '..', 'config'))
import gpucfg
CFG = gpucfg.load(); ENV = CFG.env()
OUT = '/dev/shm/flagenc'; os.makedirs(OUT, exist_ok=True)
F = dict(NONE=0, DISABLE_ANYHIT=1, ENFORCE_ANYHIT=2, TERMINATE=4, DISABLE_CLOSESTHIT=8, CULL_BACK=16, CULL_FRONT=32,
         CULL_OPAQUE=64, CULL_NON_OPAQUE=128)
VARS = {'base': {}}
for k, v in F.items():
    if v: VARS['f_' + k] = dict(FLAGS=v)
VARS['f_TERM_DISCH'] = dict(FLAGS=12)
for m in (15, 1, 128): VARS['m_%d' % m] = dict(MASK=m)
VARS['t_0p05'] = dict(TMAX='0.05f'); VARS['t_2'] = dict(TMAX='2.0f'); VARS['tmin_0p1'] = dict(TMIN='0.1f')
sel = sys.argv[1:] or list(VARS)
stamp = time.strftime('%Y%m%dT%H%M%S')
for name in sel:
    d = VARS[name]
    ptx = os.path.join(OUT, name + '.ptx')
    defs = ['-D%s=%s' % kv for kv in d.items()]
    r = subprocess.run(['nvcc', '-O2', '-arch=sm_89', '-ptx', '-I' + os.path.expanduser('~/optix_sdk/include'), '-I' + OD,
                        '-o', ptx, os.path.join(HERE, 'memdump_var_dev.cu')] + defs, capture_output=True, text=True)
    if r.returncode: sys.exit(r.stderr)
    # memdump loads memdump_dev.ptx from its own directory: copy the exe next to the variant ptx
    exe = os.path.join(OUT, 'md_' + name); os.makedirs(exe, exist_ok=True)
    subprocess.run(['cp', os.path.join(OD, 'memdump'), exe + '/memdump']); subprocess.run(['cp', ptx, exe + '/memdump_dev.ptx'])
    raw = os.path.join(OUT, '%s_%s.bin' % (name, stamp))
    if os.path.exists(raw): sys.exit('exists ' + raw)
    r = subprocess.run(['nvidia-smi', '-i', str(CFG.index), '--query-compute-apps=pid', '--format=csv,noheader'], capture_output=True, text=True)
    if r.stdout.strip(): sys.exit('GPU busy')
    cmd = ['timeout', '-k', '2', '30', exe + '/memdump', 'read', '--fn', '--fnrel', '-0x10000', '--method', 'kernel', '--len', '0x10000', '--out', raw]
    r = subprocess.run(cmd, env=ENV, capture_output=True, text=True)
    print(name, 'exit', r.returncode, r.stdout.strip().splitlines()[-1:] , r.stderr.strip()[-200:])
    if r.returncode: sys.exit('stopping at first failure')
