"""Baseline measurement helpers (Base_1). OptiX programs only, no TTU code, no code changed. Never overwrites a raw file (mode 'x').
Idle rule: no compute process on any GPU (nvidia-smi PIDs, never names) and under 10 percent utilisation on the used GPU.
Processes are started and waited on through their own subprocess handle."""
import os, sys, json, time, hashlib, subprocess, statistics, datetime

HOME = os.path.expanduser('~')
OUT_DIR = HOME + '/eoss/notes/base1'
GPU = os.environ.get('BASE_GPU', '1')          # tethys: GPU 1 (both idle, no display clients)
OPTIX_LIB = HOME + '/opt/nvoptix-580.178.04'    # tethys has no system libnvoptix; the extracted copy matches driver 580.178.04
CUDA_LIB = HOME + '/opt/cuda-12.8/lib64'        # NVRTC builtins (RTNN)

def smi(*a):
    return subprocess.run(['nvidia-smi'] + list(a), capture_output=True, text=True, timeout=30).stdout.strip()

def gpu_info(gpu=GPU):
    q = smi('-i', gpu, '--query-gpu=name,driver_version,clocks.max.sm,memory.total,pci.bus_id', '--format=csv,noheader').split(', ')
    return dict(index=int(gpu), name=q[0], driver=q[1], max_sm_mhz=q[2], mem=q[3], bus=q[4], host=os.uname().nodename)

def idle_check():
    """Raise unless no compute process is on any GPU and the used GPU is under 10 percent utilisation.
    Utilisation is a ~1 s window, so right after our own previous run it can still read high: poll up to 20 s, then refuse."""
    for i in range(10):
        apps = smi('--query-compute-apps=pid,gpu_uuid,used_memory', '--format=csv,noheader')
        util = int(smi('-i', GPU, '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'))
        if not apps and util <= 10: return
        if apps: break                              # another compute process: no waiting
        time.sleep(2)
    raise SystemExit('GPU not idle (compute apps: %r, util %d): stop' % (apps, util))

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()[:16]

def file_info(path):
    return dict(path=path.replace(HOME, '~'), bytes=os.path.getsize(path), sha256_16=sha(path))

def env(cuda_visible=GPU, extra=None):
    e = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=cuda_visible)
    e['LD_LIBRARY_PATH'] = OPTIX_LIB + ':' + CUDA_LIB + ':' + e.get('LD_LIBRARY_PATH', '')
    e['PATH'] = HOME + '/opt/cuda-12.8/bin:' + e['PATH']
    for k in ('LHOUT', 'LHSAMPLE', 'LHNOTIME', 'CB_OFF', 'RTNN_REC', 'RTNN_DUMP', 'RTNN_QCAP', 'ARK_REC', 'ARK_DUMP', 'ARK_OUT', 'DEVOUT'):
        e.pop(k, None)                          # no instrumentation switch is on unless the config says so
    e.update(extra or {})
    return e

WARM_BIN = '/dev/shm/base1_warm'   # built from warm.cu (nvcc -O2 -arch=sm_89 -o /dev/shm/base1_warm warm.cu)
WARM_MS = 2000

def clock_summary(sm, max_mhz):
    """Clock record of one run: all 50 ms samples are kept; summary excludes the first 3 samples (process start)."""
    body = sm[3:] if len(sm) > 6 else sm
    return dict(samples_mhz=sm, n=len(sm), max_sm_mhz_gpu=max_mhz, min_after_start=min(body) if body else None, median=statistics.median(sm) if sm else None,
                max=max(sm) if sm else None, frac_at_least_2400=round(sum(1 for x in sm if x >= 2400) / len(sm), 3) if sm else None,
                full_clock=bool(body) and min(body) >= 2400)

def run(cmd, cwd, environ, timeout=300, warm=True):
    """One timed run. Idle check, then a warmup process (all SMs busy WARM_MS ms, run to completion) so the clock is at boost,
    then the program; SM clock sampled every 50 ms by our own nvidia-smi child (killed through its handle). The warmup is not part of the timing."""
    idle_check()
    wenv = dict(environ, CUDA_VISIBLE_DEVICES=GPU)
    warm_out = None
    if warm:
        w = subprocess.run([WARM_BIN, str(WARM_MS)], env=wenv, capture_output=True, text=True, timeout=60)
        warm_out = w.stdout.strip() if w.returncode == 0 else 'WARM FAILED rc %d %s' % (w.returncode, w.stderr[-200:])
        if w.returncode != 0: raise SystemExit(warm_out)
    clk = subprocess.Popen(['nvidia-smi', '-i', GPU, '--query-gpu=clocks.sm', '--format=csv,noheader,nounits', '-lms', '50'],
                           stdout=subprocess.PIPE, text=True)
    t0 = time.perf_counter()
    try:
        p = subprocess.run(['timeout', '-k', '2', str(timeout)] + cmd, cwd=cwd, env=environ, capture_output=True, text=True)
        wall = time.perf_counter() - t0
    finally:
        clk.terminate()
        try: samples = clk.communicate(timeout=5)[0]
        except Exception: clk.kill(); samples = ''
    sm = [int(float(l)) for l in samples.split() if l.strip()]
    return dict(rc=p.returncode, wall_s=round(wall, 4), stdout=p.stdout, stderr=p.stderr[-2000:], warmup=warm_out,
                clock=clock_summary(sm, int(gpu_info()['max_sm_mhz'].split()[0])))

def stats(xs):
    xs = [x for x in xs if x is not None]
    if not xs: return None
    return dict(n=len(xs), median=statistics.median(xs), min=min(xs), max=max(xs), mean=statistics.fmean(xs))

def write(name, doc):
    """Raw result file: timestamped, never overwritten."""
    os.makedirs(OUT_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    path = '%s/%s_%s.json' % (OUT_DIR, name, ts)
    with open(path, 'x') as f: json.dump(doc, f, indent=1)
    return path

def header(mapping, binary, inputs, notes):
    return dict(mapping=mapping, date=datetime.datetime.now().isoformat(timespec='seconds'), gpu=gpu_info(),
                binary=file_info(binary), inputs=[file_info(i) for i in inputs], notes=notes,
                code_changed=False, tool='plain program timers (+ the OptiX recorder fields the program prints); no ncu, no NVBit',
                bittu='not applicable: OptiX kernels are invisible to NVBit (F12, F60); no TTU code runs here')

def clocks_all(runs):
    return dict(runs=len(runs), runs_at_full_clock=sum(1 for r in runs if r['clock']['full_clock']), lowest_after_start=min((r['clock']['min_after_start'] or 0) for r in runs))

def summarize(runs, keys):
    return {k: stats([r['parsed'].get(k) for r in runs if r['rc'] == 0]) for k in keys}

def check_run(cmd, cwd, environ, timeout=600):
    """Untimed run for a correctness check (no warmup, clocks not recorded)."""
    idle_check()
    p = subprocess.run(['timeout', '-k', '2', str(timeout)] + cmd, cwd=cwd, env=environ, capture_output=True, text=True)
    return dict(rc=p.returncode, stdout_tail=p.stdout[-600:], stderr_tail=p.stderr[-400:])

def summ(runs, keys):
    return summarize(runs, keys) | {'wall_s': stats([x['wall_s'] for x in runs if x['rc'] == 0]), 'clock': clocks_all(runs)}
