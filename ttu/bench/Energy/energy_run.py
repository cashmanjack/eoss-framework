"""Energy per RT query (first test): NVML total energy counter around back to back launches of the checked traversal group.

    energy_run.py WORK [SCENES] [REPS]        run with /usr/bin/python3 (numpy not needed); builds use ~/eoss/.venv/bin/python

Method. setup3 host (built to /dev/shm/t9e with the --energy S option): after its normal checked launches it repeats the same launch (and an
optional read flush) for S seconds, NVML energy counter (mJ) before and after (device synchronized), SM clock sampled during the loop, launch count.
One block per SM on every SM, ONE warp per block (32 queries per SM per session), so the first session of a launch is not served by another warp's fetch.
States (Latency_2's scenes, ray as in core_set):
  L2   = single session, no flush: BVH warm in L2, L1 empty at kernel start         E(s1 single) - E(s0 single)
  DRAM = single session after a read flush of FLUSH MB (2 x L2)                       E(s1 single, flush) - E(s0 single, flush)
  L1   = second session in the same kernel (nodes in L1 after the first)              [E(s1 twice) - E(s1 single)] - [E(s0 twice) - E(s0 single)]
s0 = the same build with no TTU instructions (stage 0): it carries the harness cost (loads, clock reads, stores, launch). Idle power: the counter over
IDLE s with nothing running, before and after each scene; 'net' numbers subtract idle power x time. Clocks are recorded per run.
Safety as every TTU run: stage 0 first, smallest stage 1 (1 block, 1 warp, 1 launch) first, run.py (timeout 30, idle GPU, check_seq ALL PASS refused otherwise).
"""
import ctypes as C, json, os, subprocess, sys, time, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'config')); import gpucfg  # noqa: E402
CFG = gpucfg.load(); os.environ['TTU_GPU'] = str(CFG.index)
S3 = os.path.join(HERE, '..', '..', 'harness', 'setup3')
VENV = os.path.expanduser('~/eoss/.venv/bin/python')
BASE, D = '/dev/shm/t9e/setup3', '/dev/shm/t9ec'
NSM = CFG.sm_count
FLUSH = 2 * CFG.l2_bytes // (1 << 20)
SEC, IDLE = 3.0, 4.0
WT = 16   # 128 registers per thread: at most 512 threads per block
SCENES = [('tri1', 'tri1', 'hit'), ('grid:12', 'grid:12', 'tri:6'), ('grid:144', 'grid:144', 'tri:72'), ('grid:200', 'grid:200', 'tri:100'),
          ('grid:2500', 'grid:2500', 'tri:1250'), ('grid:4096', 'grid:4096', 'tri:2048'), ('grid:20736', 'grid:20736', 'tri:10368')]


def sh(c, **kw): return subprocess.run(c, capture_output=True, text=True, **kw)


def build(name, stage, flags):
    os.makedirs(D, exist_ok=True)
    out = '%s/%s_s%d' % (D, name, stage)
    p = sh([VENV, os.path.join(S3, 'build_patch.py'), '--stage', str(stage), '--bin', BASE, '--out', out] + flags)
    if p.returncode: sys.exit(p.stdout + p.stderr)
    c = sh([VENV, os.path.join(S3, 'check_seq.py'), out])
    if 'ALL PASS' not in c.stdout.strip().splitlines()[-1]: sys.exit('check_seq FAILED %s\n%s' % (out, c.stdout[-2500:]))
    return out


def go(binp, rays, out, extra):
    p = sh([sys.executable, os.path.join(S3, 'run.py'), binp, rays, 'zero', out] + extra)
    ok = 'exit 0' in p.stdout and 'ttu_one: no error' in p.stdout
    if not ok: sys.exit('STOP (fault, hang or bad exit):\n' + p.stdout[-2500:] + p.stderr[-1500:])
    return p.stdout


NV = C.CDLL('libnvidia-ml.so.1'); NV.nvmlInit_v2()
_h = C.c_void_p(); NV.nvmlDeviceGetHandleByIndex_v2(CFG.index, C.byref(_h))   # CUDA_DEVICE_ORDER=PCI_BUS_ID: same order as nvidia-smi


def energy_mj():
    v = C.c_ulonglong()
    if NV.nvmlDeviceGetTotalEnergyConsumption(_h, C.byref(v)): sys.exit('NVML total energy counter unavailable')
    return v.value


def idle_power(sec=IDLE):
    comp, _ = gpucfg.users(CFG.index)
    if comp: sys.exit('GPU busy')
    e0, t0 = energy_mj(), time.time(); time.sleep(sec); e1, t1 = energy_mj(), time.time()
    return (e1 - e0) / 1000.0 / (t1 - t0)    # W


def run_one(b, ray, scene, flush, warps, tag, work, blocks):
    out = '%s/%s.json' % (work, tag)
    o = go(b, ray, out, ['--scene', scene, '--warps', str(warps), '--blocks', str(blocks), '--repeat', '2', '--energy', str(SEC)] + (['--flush', str(flush)] if flush else []))
    j = [json.loads(l[7:]) for l in o.splitlines() if l.startswith('ENERGY ')]
    if len(j) != 1: sys.exit('no ENERGY line\n' + o[-1500:])
    return j[0]


def main(work, scenes=None, reps=2):
    os.makedirs(work, exist_ok=True)
    sc = [s for s in SCENES if scenes is None or s[0] in scenes]
    fl = ['--clock', '--compact-out']
    b = {(k, s): build('e%s' % ('t' if k == 'twice' else 's'), s, fl + (['--twice'] if k == 'twice' else [])) for k in ('single', 'twice') for s in (0, 1)}
    print('stage 0 first, then smallest stage 1 (1 block, 1 warp, 1 launch)', flush=True)
    go(b[('single', 0)], 'hit', work + '/s0_first.json', ['--warps', '1', '--repeat', '1']); go(b[('twice', 0)], 'hit', work + '/t0_first.json', ['--warps', '1', '--repeat', '1'])
    go(b[('single', 0)], 'hit', work + '/s0_all.json', ['--warps', '1', '--blocks', str(NSM), '--repeat', '1'])
    go(b[('single', 1)], 'hit', work + '/s1_first.json', ['--warps', '1', '--repeat', '1']); go(b[('twice', 1)], 'hit', work + '/t1_first.json', ['--warps', '1', '--repeat', '1'])
    go(b[('single', 1)], 'hit', work + '/s1_all.json', ['--warps', '1', '--blocks', str(NSM), '--repeat', '1'])
    go(b[('single', 0)], 'hit', work + '/s0_w20.json', ['--warps', str(WT), '--blocks', str(NSM), '--repeat', '1']); go(b[('twice', 0)], 'hit', work + '/t0_w20.json', ['--warps', str(WT), '--blocks', str(NSM), '--repeat', '1'])
    go(b[('single', 1)], 'hit', work + '/s1_w4.json', ['--warps', '4', '--blocks', str(NSM), '--repeat', '1'])
    go(b[('single', 1)], 'hit', work + '/s1_w20.json', ['--warps', str(WT), '--blocks', str(NSM), '--repeat', '1'])
    res = []
    for name, scene, ray in sc:
        row = dict(scene=name)
        for W in (1, WT):
            idle = [idle_power()]
            runs = {}
            run_one(b[('single', 0)], ray, scene, 0, W, 'warmup_w%d' % W, work, NSM)   # clock warm up, discarded
            for rep in range(int(reps)):
                for key, (kind, stage, flush) in {'s0': ('single', 0, 0), 's1': ('single', 1, 0), 't0': ('twice', 0, 0), 't1': ('twice', 1, 0), 's0f': ('single', 0, FLUSH), 's1f': ('single', 1, FLUSH)}.items():
                    r = run_one(b[(kind, stage)], ray, scene, flush, W, '%s_w%d_%s_r%d' % (name.replace(':', ''), W, key, rep), work, NSM)
                    runs.setdefault(key, []).append(r)
                    print('  ', name, 'w%d' % W, key, rep, 'mJ/launch %.3f' % (r['mJ'] / r['launches']), 'launches/s %.0f' % (r['launches'] / r['s']), 'W %.1f' % (r['mJ'] / 1000 / r['s']), 'SM MHz %.0f' % r['sm_mhz_mean'], flush=True)
            idle.append(idle_power())
            pi = st.mean(idle)
            q = NSM * W * 32   # queries per session per launch
            def per_launch(k): return st.median([r['mJ'] / r['launches'] for r in runs[k]])
            def net(k): return st.median([(r['mJ'] / 1000 - pi * r['s']) / r['launches'] * 1000 for r in runs[k]])   # mJ per launch above idle
            row['w%d' % W] = dict(idle_W=pi, idle_runs=idle, q_per_launch=q,
                       uJ_per_query=dict(L2=(per_launch('s1') - per_launch('s0')) / q * 1000, DRAM=(per_launch('s1f') - per_launch('s0f')) / q * 1000,
                                         L1=((per_launch('t1') - per_launch('s1')) - (per_launch('t0') - per_launch('s0'))) / q * 1000),
                       net_uJ_per_query_incl_harness={k: net(k) / q * 1000 for k in runs},
                       mJ_per_launch={k: per_launch(k) for k in runs}, launches_per_s={k: st.median([r['launches'] / r['s'] for r in runs[k]]) for k in runs},
                       sm_mhz={k: st.median([r['sm_mhz_mean'] for r in runs[k]]) for k in runs}, mem_mhz={k: runs[k][0]['mem_mhz'] for k in runs},
                       spread_pct={k: 100 * (max(r['mJ'] / r['launches'] for r in runs[k]) - min(r['mJ'] / r['launches'] for r in runs[k])) / per_launch(k) for k in runs})
        res.append(row); print(json.dumps(row), flush=True)
    json.dump(dict(gpu=CFG.name, sm_count=NSM, flush_mb=FLUSH, sec=SEC, rows=res), open(work + '/energy.json', 'w'), indent=1)


if __name__ == '__main__':
    a = sys.argv
    main(a[1], a[2].split(',') if len(a) > 2 and a[2] != 'all' else None, a[3] if len(a) > 3 else 2)
