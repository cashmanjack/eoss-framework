"""Ablations of the persistent loop kernel (why is it 2.4 to 3.6 times slower than OptiX?), one change at a time, then combined.

    ablate.py CONFIG [time|ncu|both]      CONFIGS: base28 hoist28 coal28 base13 comb13   (scene soup:100000, G = 142 blocks, random rays)

 base28   the kernel of the long launch runs (28 warps per SM, 72 registers, 13 scattered result words per ray)
 hoist28  --hoist: the constant input words are loaded once before the outer loop (see build_patch.py)
 coal28   --coal : one 16 byte STG.128 record per ray, consecutive lanes write consecutive records
 base13   base kernel, blocks of 13 warps (OptiX's achieved occupancy), R scaled so the ray count stays about 12.7 M
 comb13   --hoist --coal, 13 warps, 96 registers (20 extra registers)
 (a per warp work counter fetch does not exist in this kernel: records are assigned statically, lane g of round k reads record k G L + g)

time: stage 0 (group as NOPs) first, then stage 1 under timeout 30 on GPU 1 alone (run.py), 6 launches (first = cold, median of the other 5),
      every ray of the last launch compared with OptiX's result (hit, t bits, primitive) and 200 rays with a CPU brute force.
ncu : stage 1 under ncu (one launch profiled after one skipped), 1.27 M rays, MIO stall warps and the other counters.
Binaries live in /dev/shm/t8/abl and are deleted after the run.
"""
import json, os, statistics as st, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
S3 = os.path.join(HERE, '..', '..', 'harness', 'setup3')
sys.path.insert(0, os.path.join(HERE, '..', 'Memory_4')); sys.path.insert(0, S3)
from prep import make_rays  # noqa: E402
from resume_check import closest  # noqa: E402

D = '/dev/shm/t8/abl'
BASE = D + '/setup3'
VENV = os.path.expanduser('~/eoss/.venv/bin/python')
NCU = os.path.expanduser('~/opt/cuda-12.8/bin/ncu')
G = 142
K = 32
SCENE = 'soup:100000'
SAMPLE = 200
WORK = os.path.expanduser('~/eoss/notes/setup3/runs/ablate')
CFG = {   # name: (warps, regs, build flags, host flags)
    'base28': (28, 72, [], []),
    'hoist28': (28, 72, ['--hoist'], []),
    'coal28': (28, 72, ['--coal'], ['--coal']),
    'base13': (13, 72, [], []),
    'comb13': (13, 96, ['--hoist', '--coal'], ['--coal']),
}
NCU_METRICS = ','.join([
    'gpu__time_duration.sum', 'sm__cycles_active.avg', 'sm__warps_active.avg.per_cycle_active', 'smsp__inst_executed.sum', 'smsp__thread_inst_executed.sum',
    'smsp__issue_active.avg.pct_of_peak_sustained_active', 'launch__registers_per_thread',
    'smsp__warps_issue_stalled_mio_throttle', 'smsp__warps_issue_stalled_mio_throttle_pipe_mio', 'smsp__warps_issue_stalled_lg_throttle',
    'smsp__warps_issue_stalled_long_scoreboard', 'smsp__warps_issue_stalled_long_scoreboard_pipe_l1tex', 'smsp__warps_issue_stalled_short_scoreboard',
    'smsp__warps_issue_stalled_wait', 'smsp__warps_issue_stalled_barrier', 'smsp__warps_issue_stalled_branch_resolving',
    'l1tex__t_sectors.sum', 'l1tex__t_sectors_pipe_lsu.sum', 'lts__t_sectors.sum', 'dram__bytes_read.sum', 'lts__throughput.avg.pct_of_peak_sustained_elapsed'])


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def rounds_for(W, n_target):
    return max(1, round(n_target / (G * 32 * W)))


def build(stage, W, R, regs, flags, out):
    stride = G * 32 * W * 96
    cmd = [VENV, os.path.join(S3, 'build_patch.py'), '--stage', str(stage), '--bin', BASE, '--slim', '--remap', '--compact-out', '--regs', str(regs), '--max-out', '61',
           '--loop', str(K), '--outer', str(R), '--stride', str(stride), '--out', out] + flags
    p = sh(cmd)
    if p.returncode: sys.exit(p.stdout + p.stderr)
    c = sh([VENV, os.path.join(S3, 'check_seq.py'), out])
    if 'ALL PASS' not in c.stdout.splitlines()[-1]: sys.exit(c.stdout[-2500:])
    return c.stdout.splitlines()[-1]


def go(binp, rays, out, W, R, extra, ncu=None):
    cmd = [sys.executable, os.path.join(S3, 'run.py'), binp, 'file:' + rays, 'zero', out, '--scene', SCENE, '--warps', str(W), '--blockrays', '--slim', '--loop',
           '--blocks', str(G), '--outer', str(R)] + extra
    if ncu: cmd += ['--ncu', NCU_METRICS, '--ncu-out', ncu, '--ncu-skip', '1']
    p = sh(cmd)
    ok = 'exit 0' in p.stdout and ('no error' in p.stdout or ncu)
    print(' ', os.path.basename(out), 'OK' if ok else 'FAILED', flush=True)
    if not ok: sys.exit(p.stdout[-2500:] + p.stderr[-1500:])
    return p.stdout


def run_time(name):
    W, regs, bflags, hflags = CFG[name]
    R = rounds_for(W, 12.7e6); n = G * 32 * W * R
    rays = '%s/rays_%s.bin' % (D, name); make_rays(n).tofile(rays)
    b0, b1 = '%s/s0_%s' % (D, name), '%s/s1_%s' % (D, name)
    sha0 = build(0, W, R, regs, bflags, b0); sha1 = build(1, W, R, regs, bflags, b1)
    tris, fin, ref = ['%s/%s_%s.bin' % (D, k, name) for k in ('tris', 'fin', 'ref')]
    go(b0, rays, '%s/s0_%s.json' % (WORK, name), W, R, ['--repeat', '1', '--quick', '--dump-tris', tris] + hflags)
    go(b1, rays, '%s/%s.json' % (WORK, name), W, R, ['--repeat', '6', '--quick', '--optix-repeat', '6', '--dump-final', fin, '--dump-ref', ref] + hflags)
    j = json.load(open('%s/%s.json' % (WORK, name)))
    ms = [l['ms'] for l in j['launches']][1:]; oms = j['optix_ms']
    a = np.fromfile(fin, dtype=np.uint32).reshape(-1, 14); r = np.fromfile(ref, dtype=np.uint32).reshape(-1, 8)
    assert len(a) == n == len(r), (len(a), n, len(r))
    hit = a[:, 0] != 0; rhit = r[:, 0] != 0
    t = np.abs(a[:, 4].view(np.float32)); rt = r[:, 1].view(np.float32)
    prim = (a[:, 3] & 0x1fffffff).astype(np.int64); both = hit & rhit
    unstored = int((a[:, 13] == 0x77777777).sum()); unfin = int((a[:, 13] == 0).sum()); rounds = K - a[:, 13].astype(np.int64)
    bad_hit = int((hit != rhit).sum()); bad_t = int((both & (t != rt)).sum()); bad_p = int((both & (prim != r[:, 2].astype(np.int64))).sum())
    T = np.fromfile(tris, dtype=np.float32).reshape(-1, 3, 3).astype(np.float64); Rr = np.fromfile(rays, dtype=np.float32, count=n * 8).reshape(-1, 8)
    idx = np.unique(np.linspace(0, n - 1, SAMPLE).astype(np.int64)); cpu_bad = 0
    for i in idx:
        tc, pc = closest(T, Rr[i, :3].astype(np.float64), Rr[i, 4:7].astype(np.float64))
        good = (not hit[i]) if tc is None else (hit[i] and abs(t[i] - tc) <= 1e-5 * max(1.0, tc) and prim[i] == pc)
        cpu_bad += not good
    res = dict(name=name, warps=W, regs=regs, R=R, rays=n, sha0=sha0, sha1=sha1, ours_ms=st.median(ms), ours_min=min(ms), ours_max=max(ms), launches=ms, optix_ms=st.median(oms),
               unstored=unstored, round_limit=unfin, rounds_mean=float(rounds.mean()), rounds_max=int(rounds.max()),
               vs_optix_bad_hit=bad_hit, vs_optix_bad_t=bad_t, vs_optix_bad_prim=bad_p, cpu_checked=len(idx), cpu_bad=int(cpu_bad))
    res['ours_rays_per_us'] = n / res['ours_ms'] / 1000; res['optix_rays_per_us'] = n / res['optix_ms'] / 1000
    json.dump(res, open('%s/res_%s.json' % (WORK, name), 'w'), indent=1)
    print('%-8s W %2d R %3d rays %9d | ours %8.3f ms (%.0f r/us, min %.3f max %.3f) | optix %8.3f ms (%.0f r/us) | rounds %.2f max %d | unstored %d limit %d | wrong vs optix hit %d t %d prim %d | cpu %d/%d wrong'
          % (name, W, R, n, res['ours_ms'], res['ours_rays_per_us'], res['ours_min'], res['ours_max'], res['optix_ms'], res['optix_rays_per_us'], res['rounds_mean'], res['rounds_max'],
             unstored, unfin, bad_hit, bad_t, bad_p, len(idx), cpu_bad), flush=True)
    for f in (rays, tris, fin, ref, b0, b1, b0 + '.checked', b1 + '.checked', b0 + '.layout.json', b1 + '.layout.json'):
        if os.path.exists(f): os.remove(f)


def run_ncu(name):
    W, regs, bflags, hflags = CFG[name]
    R = rounds_for(W, 1.27e6); n = G * 32 * W * R
    rays = '%s/nrays_%s.bin' % (D, name); make_rays(n).tofile(rays)
    b0, b1 = '%s/n0_%s' % (D, name), '%s/n1_%s' % (D, name)
    build(0, W, R, regs, bflags, b0); build(1, W, R, regs, bflags, b1)
    go(b0, rays, '%s/n0_%s.json' % (D, name), W, R, ['--repeat', '1', '--quick'] + hflags)
    csvf = '%s/ncu_%s.csv' % (WORK, name)
    go(b1, rays, '%s/n1_%s.json' % (D, name), W, R, ['--repeat', '3', '--quick'] + hflags, ncu=csvf)
    sys.path.insert(0, HERE)
    import ncu_compare as NC
    v, kn = NC.metrics_of(csvf, lambda rows: rows[-1])
    v['rays'] = n; v['warps'] = W
    json.dump(v, open('%s/ncu_%s.json' % (WORK, name), 'w'), indent=1)
    print('ncu', name, kn[:20], 'rays', n)
    for f in (rays, b0, b1, b0 + '.checked', b1 + '.checked', b0 + '.layout.json', b1 + '.layout.json'):
        if os.path.exists(f): os.remove(f)


def run_ncu_optix():
    """OptiX's launch kernel on the same family of rays (1.27 M, R = 10 at 28 warps), the same ncu counters."""
    sys.path.insert(0, HERE)
    import ncu_compare as NC, long_run as T
    R = 10; n = G * T.LANES * R
    rays = D + '/nrays_optix.bin'; make_rays(n).tofile(rays)
    b0 = D + '/n0_optix'; build(0, 28, R, 72, [], b0)
    csvf = '%s/ncu_optix.csv' % WORK
    cmd = ['timeout', '-k', '2', '60', NCU, '--target-processes', 'all', '-k', 'regex:optix', '--cache-control', 'none', '--clock-control', 'none', '--metrics', NCU_METRICS, '--csv',
           '--page', 'raw', '--log-file', csvf, b0, D + '/setup3_dev.ptx', D + '/words.txt', 'file:' + rays, 'zero', csvf + '.json', '--scene', SCENE, '--warps', '28', '--blockrays',
           '--slim', '--loop', '--blocks', str(G), '--outer', str(R), '--no-ttu', '--quick', '--optix-repeat', '1']
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='1', LD_LIBRARY_PATH=os.path.expanduser('~/opt/nvoptix-580.178.04') + ':' + os.environ.get('LD_LIBRARY_PATH', ''))
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    v, kn = NC.metrics_of(csvf, lambda rows: rows[-1])
    v['rays'] = n
    json.dump(v, open('%s/ncu_optix.json' % WORK, 'w'), indent=1)
    print('ncu optix', kn[:40], 'rays', n, p.returncode)
    for f in (rays, b0, b0 + '.checked', b0 + '.layout.json'):
        if os.path.exists(f): os.remove(f)


if __name__ == '__main__':
    os.makedirs(WORK, exist_ok=True)
    name = sys.argv[1]; what = sys.argv[2] if len(sys.argv) > 2 else 'time'
    if name == 'optix': run_ncu_optix()
    else:
        if what in ('time', 'both'): run_time(name)
        if what in ('ncu', 'both'): run_ncu(name)
