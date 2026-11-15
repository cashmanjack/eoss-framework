"""Throughput_10: payload and attribute size (OptiX only, no TTU code).
    payload_run.py WORK            run with /usr/bin/python3; GPU from gpucfg (TTU_GPU)
Per config (payload values NPAY, attribute values NATTR, triangles or custom AABB primitives from the same 100,000 triangle soup) and ray set (random, coherent camera):
rays per us (3 runs x 10 launches, median of launches), and ncu on one optixLaunch: registers per thread, stack size, mean active warps per SM, local memory bytes (stores, loads) per ray, kernel time.
Every result's hit / t / primitive (where the payload carries them) is checked against the NPAY 8 triangle run.
"""
import json, os, subprocess, sys, statistics as st, re
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..', '..', 'config')); import gpucfg  # noqa: E402
CFG = gpucfg.load(); ENV = CFG.env(); D = '/dev/shm/pl'; NVCC = os.path.expanduser('~/opt/cuda-12.8/bin/nvcc'); NCU = os.path.expanduser('~/opt/cuda-12.8/bin/ncu')
N_TRI = 100000; NR = 1 << 20
def sh(c, **kw): return subprocess.run(c, capture_output=True, text=True, env=ENV, **kw)
def rays():
    rng = np.random.default_rng(2026); n = NR
    R = np.zeros((n, 8), np.float32); R[:, 0:3] = rng.uniform(-1, 1, (n, 3)); d = rng.normal(size=(n, 3)); R[:, 4:7] = d / np.linalg.norm(d, axis=1, keepdims=True); R[:, 7] = 1e16; R.tofile(D + '/rand.bin')
    W = 1024; H = n // W; x, y = np.meshgrid((np.arange(W) + .5) / W * 2 - 1, (np.arange(H) + .5) / H * 2 - 1)
    dd = np.stack([0.4 * x.ravel(), 0.4 * y.ravel(), np.ones(W * H)], 1); dd /= np.linalg.norm(dd, axis=1, keepdims=True)
    C = np.zeros((n, 8), np.float32); C[:, 0:3] = (0, 0, -4); C[:, 4:7] = dd; C[:, 7] = 1e16; C.tofile(D + '/coh.bin')
def ptx(npay, nattr, custom):
    p = '%s/dev_p%d_a%d_c%d.ptx' % (D, npay, nattr, custom)
    if not os.path.exists(p):
        r = sh([NVCC, '-O2', '-std=c++17', '-arch=sm_89', '-ptx', '-I' + os.path.expanduser('~/optix_sdk/include'), '-DNPAY=%d' % npay, '-DNATTR=%d' % nattr, '-DCUSTOM=%d' % custom, os.path.join(HERE, 'payload_dev.cu'), '-o', p])
        if r.returncode: sys.exit(r.stderr)
    return p
METRICS = 'launch__registers_per_thread,launch__stack_size,sm__warps_active.avg.per_cycle_active,gpu__time_duration.sum,l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum,l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum,launch__occupancy_limit_registers,sm__maximum_warps_per_active_cycle_pct'
def num(s):
    s = s.replace(',', ''); return float(s)
def ncu(args):
    r = sh([NCU, '--metrics', METRICS, '-k', 'regex:optixLaunch', '-s', '1', '-c', '1'] + args)
    if r.returncode: sys.exit('ncu failed\n' + r.stdout[-1500:] + r.stderr[-800:])
    m = {}; unit = {}
    for l in r.stdout.splitlines():
        f = l.split()
        if len(f) >= 3 and ('__' in f[0] or '.' in f[0]) and re.match(r'^[\d.,]+$', f[-1]):
            u = f[-2] if len(f) > 2 else ''; v = num(f[-1])
            if u == 'Mbyte': v *= 1e6
            elif u == 'Kbyte': v *= 1e3
            elif u == 'Gbyte': v *= 1e9
            elif u == 'usecond': pass
            m[f[0]] = (v, u)
    return {k: v[0] for k, v in m.items()}
def main(work):
    os.makedirs(work, exist_ok=True); os.makedirs(D, exist_ok=True); rays()
    cfgs = [(p, a, 0) for a in (2, 8) for p in (0, 2, 8, 16, 32)] + [(p, a, 1) for p in (2, 8, 32) for a in (0, 1, 2, 4, 8)]
    ref = {}; rows = []
    for (npay, nattr, custom) in cfgs:
        P = ptx(npay, nattr, custom)
        for rs in ('rand', 'coh'):
            args = [D + '/payload', P, str(N_TRI), D + '/%s.bin' % rs, D + '/out.bin', '10', str(nattr), str(custom), str(npay)]
            ms = []
            for _ in range(3):
                r = sh(['timeout', '-k', '2', '60'] + args)
                if r.returncode: sys.exit('host failed\n' + r.stderr[-800:])
                ms.append(st.median(json.loads(r.stdout.strip().splitlines()[-1])['ms']))
            o = np.fromfile(D + '/out.bin', np.uint32).reshape(-1, 4)
            if npay >= 1 and not custom and npay == 8 and nattr == 2: ref[rs] = o.copy()
            met = ncu(args[:5] + ['2'] + args[6:])
            rows.append(dict(npay=npay, nattr=nattr, custom=custom, rays=rs, ms=st.median(ms), ms_runs=ms, ncu=met, hit_sum=int(o[:, 0].sum()) if npay >= 1 else None, chk=o.copy()))
            r0 = rows[-1]; m = r0['ncu']
            print('payload %2d attr %d %-8s %-4s %6.3f ms %6.0f rays/us  regs %3.0f stack %5.0f warps/SM %5.2f  local st %6.1f B/ray ld %6.1f B/ray (sectors %.2f/%.2f)  ncu time %.0f us' % (
                npay, nattr, 'custom' if custom else 'triangle', rs, r0['ms'], NR / r0['ms'] / 1000, m.get('launch__registers_per_thread', -1), m.get('launch__stack_size', -1), m.get('sm__warps_active.avg.per_cycle_active', -1),
                m.get('l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum', 0) / NR, m.get('l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum', 0) / NR, m.get('l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum', 0) / NR, m.get('l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum', 0) / NR, m.get('gpu__time_duration.sum', 0)), flush=True)
    # results check: fields present in the payload (hit, t, prim = values 0..2) against the triangle NPAY 8 run, triangles only
    bad = 0; checked = 0
    for r in rows:
        if r['custom'] or r['npay'] < 3: continue
        o = r['chk']; rr = ref[r['rays']]; checked += 1
        if (o[:, :3] != rr[:, :3]).any(): bad += 1; print('MISMATCH', r['npay'], r['nattr'], r['rays'])
    print('result check (triangle runs with at least 3 payload values vs payload 8): %d runs, %d mismatching' % (checked, bad))
    for r in rows: r.pop('chk')
    json.dump(rows, open(os.path.join(work, 'payload.json'), 'w'), indent=1)
if __name__ == '__main__': main(sys.argv[1])
