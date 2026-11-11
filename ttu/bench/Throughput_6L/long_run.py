"""Throughput_6 / plateau with long launches: a persistent loop kernel (build_patch --loop 32 --outer R), G blocks of 28 warps (896 lanes),
each lane traces R rays (ray k x G x 896 + gid in round k), so one launch is G x 896 x R rays (millions) and runs for milliseconds.

    long_run.py WORK SCENE KIND G R [REPEAT] [SMS]        KIND: rnd (random rays, make_rays) | coh (one ray)
    SMS: with an --smfilter build only these SMs run (not used here: G blocks = G SMs, one block per SM)

Per config: build stage 0 and stage 1 (same flags, outer R, stride G x 896 x 96 words), check_seq both, run stage 0 first (no TTU code),
then stage 1 (--quick: 6 timed launches, last one copied back), OptiX on the same rays (--optix-repeat 6, results of every ray dumped).
Check: every record (ray) of the last launch vs OptiX (hit flag, t bits, primitive) and a CPU brute force closest hit on SAMPLE rays.
Binaries live in /dev/shm/t8 and are deleted after the run.
"""
import json, os, statistics as st, subprocess, sys, shutil
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
S3 = os.path.join(HERE, '..', '..', 'harness', 'setup3')
sys.path.insert(0, os.path.join(HERE, '..', 'Memory_4')); sys.path.insert(0, S3)
from prep import make_rays  # noqa: E402
from resume_check import closest  # noqa: E402

BASE = '/dev/shm/t8/setup3'
VENV = os.path.expanduser('~/eoss/.venv/bin/python')   # build_patch and check_seq import the eoss package (no numpy there); the driver runs under /usr/bin/python3
LANES = 28 * 32
K = 32
SAMPLE = 200


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def build(stage, G, R, out, extra=()):
    stride = G * LANES * 96
    cmd = [VENV, os.path.join(S3, 'build_patch.py'), '--stage', str(stage), '--bin', BASE, '--slim', '--remap', '--compact-out', '--regs', '72',
           '--max-out', '61', '--loop', str(K), '--outer', str(R), '--stride', str(stride), '--out', out] + list(extra)
    p = sh(cmd)
    if p.returncode: sys.exit(p.stdout + p.stderr)
    c = sh([VENV, os.path.join(S3, 'check_seq.py'), out])
    if 'ALL PASS' not in c.stdout.splitlines()[-1]: sys.exit(c.stdout[-2000:])


def go(binp, rays, out, scene, G, R, extra):
    cmd = [sys.executable, os.path.join(S3, 'run.py'), binp, 'file:' + rays, 'zero', out, '--scene', scene, '--warps', '28', '--blockrays', '--slim', '--loop',
           '--blocks', str(G), '--outer', str(R)] + extra
    p = sh(cmd)
    ok = 'exit 0' in p.stdout and 'no error' in p.stdout
    print(' ', os.path.basename(out), 'OK' if ok else 'FAILED', flush=True)
    if not ok: sys.exit(p.stdout[-2500:] + p.stderr[-1500:])
    return p.stdout


def run(work, scene, kind, G, R, repeat=6, tag=None, smset=None, rays_file=None):
    os.makedirs(work, exist_ok=True)
    n = G * LANES * R
    tag = tag or '%s_g%d_r%d' % (scene.replace(':', ''), G, R)
    rays = '/dev/shm/t8/rays_%s.bin' % tag
    if rays_file:    # a prepared ray file of exactly n rays (e.g. a grouped order)
        rays = rays_file
    elif kind == 'coh':
        np.repeat(np.array([[0, 0, -1, 0, 0, 0, 1, 1e16]], np.float32), n, axis=0).tofile(rays)
    else:
        make_rays(n).tofile(rays)
    b0, b1 = '/dev/shm/t8/s0_' + tag, '/dev/shm/t8/s1_' + tag
    bx = ['--smfilter'] if smset else []
    build(0, G, R, b0, bx); build(1, G, R, b1, bx)
    sx = ['--sm-set', ','.join(map(str, smset))] if smset else []
    tris = '/dev/shm/t8/tris_' + tag + '.bin'; fin = '/dev/shm/t8/fin_' + tag + '.bin'; ref = '/dev/shm/t8/ref_' + tag + '.bin'
    go(b0, rays, os.path.join(work, 's0_%s.json' % tag), scene, G, R, ['--repeat', '1', '--quick', '--dump-tris', tris] + sx)
    out = go(b1, rays, os.path.join(work, '%s.json' % tag), scene, G, R,
             ['--repeat', str(repeat), '--quick', '--optix-repeat', '6', '--dump-final', fin, '--dump-ref', ref] + sx)
    j = json.load(open(os.path.join(work, '%s.json' % tag)))
    ms = [l['ms'] for l in j['launches']][1:]; oms = j['optix_ms']
    a = np.fromfile(fin, dtype=np.uint32).reshape(-1, 14)
    r = np.fromfile(ref, dtype=np.uint32).reshape(-1, 8)
    assert len(a) == n == len(r), (len(a), n, len(r))
    if smset:    # only the blocks on the chosen SMs ran: keep their records (block b of round k = records k G 896 + b 896 ..)
        blocks = [b for b in range(G) if (2 * b if b < 71 else 2 * (b - 71) + 1) in smset]
        ids = (np.arange(R)[:, None, None] * G * LANES + np.array(blocks)[None, :, None] * LANES + np.arange(LANES)[None, None, :]).ravel()
        nran = int(j['launches'][-1]['ran']); assert nran == len(blocks), (nran, len(blocks))
        a = a[ids]; r = r[ids]; n_proc = len(ids)
    else:
        ids = None; n_proc = n
    hit = a[:, 0] != 0; rhit = r[:, 0] != 0
    t = np.abs(a[:, 4].view(np.float32)); rt = r[:, 1].view(np.float32)
    prim = (a[:, 3] & 0x1fffffff).astype(np.int64); both = hit & rhit
    unstored = int((a[:, 13] == 0x77777777).sum()); unfin = int((a[:, 13] == 0).sum())
    rounds = K - a[:, 13].astype(np.int64)
    bad_hit = int((hit != rhit).sum()); bad_t = int((both & (t != rt)).sum()); bad_p = int((both & (prim != r[:, 2].astype(np.int64))).sum())
    # CPU brute force on SAMPLE records spread over the launch
    T = np.fromfile(tris, dtype=np.float32).reshape(-1, 3, 3).astype(np.float64); Rr = np.fromfile(rays, dtype=np.float32, count=n * 8).reshape(-1, 8)
    if ids is not None: Rr = Rr[ids]
    idx = np.unique(np.linspace(0, n_proc - 1, SAMPLE).astype(np.int64)); cpu_bad = 0
    for i in idx:
        tc, pc = closest(T, Rr[i, :3].astype(np.float64), Rr[i, 4:7].astype(np.float64))
        good = (not hit[i]) if tc is None else (hit[i] and abs(t[i] - tc) <= 1e-5 * max(1.0, tc) and prim[i] == pc)
        cpu_bad += not good
    res = dict(tag=tag, scene=scene, G=G, R=R, rays=n_proc, ours_ms=st.median(ms), ours_min=min(ms), ours_max=max(ms), optix_ms=st.median(oms), launches=ms, optix_launches=oms,
               unstored=unstored, round_limit=unfin, rounds_mean=float(rounds.mean()), rounds_max=int(rounds.max()), hits=int(hit.sum()),
               vs_optix_bad_hit=bad_hit, vs_optix_bad_t=bad_t, vs_optix_bad_prim=bad_p, cpu_checked=len(idx), cpu_bad=int(cpu_bad))
    res['ours_rays_per_us'] = n_proc / res['ours_ms'] / 1000; res['optix_rays_per_us'] = n / res['optix_ms'] / 1000   # OptiX always traces all n rays
    json.dump(res, open(os.path.join(work, 'res_%s.json' % tag), 'w'), indent=1)
    print('%-22s rays %9d | ours %8.3f ms (%.0f r/us) | optix %8.3f ms (%.0f r/us) | optix/ours %.2f | rounds %.2f max %d | unstored %d limit %d | vs optix wrong: hit %d t %d prim %d | cpu %d/%d wrong'
          % (tag, n_proc, res['ours_ms'], res['ours_rays_per_us'], res['optix_ms'], res['optix_rays_per_us'], res['optix_ms'] / res['ours_ms'], res['rounds_mean'], res['rounds_max'], unstored, unfin, bad_hit, bad_t, bad_p, len(idx), cpu_bad), flush=True)
    for f in ((rays,) if not rays_file else ()) + (tris, fin, ref, b0, b1, b0 + '.checked', b1 + '.checked', b0 + '.layout.json', b1 + '.layout.json'):
        if os.path.exists(f): os.remove(f)
    return res


if __name__ == '__main__':
    work, scene, kind, G, R = sys.argv[1:6]
    run(work, scene, kind, int(G), int(R), int(sys.argv[6]) if len(sys.argv) > 6 else 6)
