"""Energy per bounce ray (OptiX only, no TTU code): NVML total energy over windows of back to back short launches.

    energy_bounce.py WORK SCENE_N LOG2_PRIMARY WINDOW_S REPS

Bounce rays as bounce_oracle.py (origin = camera hit, random direction). Configs, each = one host process with an idle window before, the loop window, an idle window after:
  pixel   natural order trace only            morton  Morton of origin (5 bits per axis, CPU sort), trace only
  oracle  grouped by the ray's own hit subtree (full key, a bound), trace only
  net     natural file through the GPU sort pipeline (key generation, radix sort, gather, trace, scatter), all of it inside the window
Per window: energy (mJ) and time, launches, mean SM clock (sampled), end clocks and power. net energy = window energy minus mean idle power x window time.
Reps are interleaved (rep outer, config inner). Every result checked against the natural order result of the same rays. Env TTU_GPU / MORTON_DIR as morton_run.
"""
import json, os, subprocess, sys, statistics as st
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'Memory_4'))
import morton_run as mr, netsort_run as ns  # noqa: E402
M = mr.M


def host(scene_n, rays, out, modes, timeout=400):
    mr.gpu_free()
    r = subprocess.run(['timeout', '-k', '2', str(timeout), M + '/morton', M + '/morton_dev.ptx', str(scene_n), rays, out, '1'] + modes, env=mr.ENV, capture_output=True, text=True)
    if r.returncode: sys.exit('host failed %d\n%s\n%s' % (r.returncode, r.stdout[-1500:], r.stderr[-1500:]))
    return [json.loads(l) for l in r.stdout.splitlines() if l.startswith('{')]


def main(work, scene_n, lg, win, reps):
    os.makedirs(work, exist_ok=True); scene_n = int(scene_n); n = 1 << int(lg); win = float(win); reps = int(reps)
    R = ns.make(scene_n, n, 'bounce'); N = len(R)
    base = M + '/e_base.bin'; R.tofile(base); ref = M + '/e_ref'
    mr.host(scene_n, base, ref, ['plain'], ['--dump-gas', ref + '.gas', '--dump-tris', ref + '.tris'])
    canon = np.fromfile(ref + '.0.bin', np.uint32).reshape(-1, 8)
    tris = np.fromfile(ref + '.tris', np.float32).reshape(-1, 3, 3)
    W, kids, root = mr.prep.tree(open(ref + '.gas', 'rb').read())
    codes = mr.oracle_codes(W, kids, root, R, canon, tris)
    print('scene %d: %d bounce rays, %.1f%% hit' % (scene_n, N, 100 * (canon[:, 0] != 0).mean()), flush=True)
    perms = {'pixel': np.arange(N), 'morton': mr.order_by(mr.code_o(R, 5)), 'oracle': mr.order_by(codes['full'][0])}
    for k, p in perms.items(): R[p].tofile(M + '/e_%s.bin' % k)
    cfgs = ['pixel', 'morton', 'oracle', 'net']; rows = []
    for rep in range(reps):
        order = cfgs[rep % 4:] + cfgs[:rep % 4]   # rotate the first config each rep
        for c in order:
            f = M + '/e_%s.bin' % ('pixel' if c == 'net' else c); perm = perms['pixel' if c == 'net' else c]
            mid = 'egysort:%g:5:0:o' % win if c == 'net' else 'egy:%g' % win
            rs = host(scene_n, f, M + '/e_out', ['egyidle:%g' % win, mid, 'egyidle:%g' % win], timeout=int(3 * win + 300))
            o = np.fromfile(M + '/e_out.1.bin', np.uint32).reshape(-1, 8); os.remove(M + '/e_out.1.bin')
            bad = int((o != canon[perm]).any(1).sum()) if c != 'net' else int((o != canon).any(1).sum())
            i0, w, i1 = rs
            pw = (i0['mJ'] + i1['mJ']) / (i0['s'] + i1['s'])      # idle power, mW
            net_mJ = w['mJ'] - pw * w['s']
            rays = w['launches'] * N
            row = dict(scene=scene_n, rep=rep, cfg=c, rays_per_launch=N, launches=w['launches'], s=w['s'], mJ=w['mJ'], idle_mW=pw, idle_mW_before=i0['mJ'] / i0['s'], idle_mW_after=i1['mJ'] / i1['s'],
                       net_mJ=net_mJ, rays_per_us=rays / w['s'] / 1e6, us_per_launch=w['s'] / w['launches'] * 1e6, nJ_per_ray=w['mJ'] * 1e6 / rays, nJ_per_ray_net=net_mJ * 1e6 / rays,
                       W_total=w['mJ'] / w['s'] / 1e3, sm_mhz_mean=w['sm_mhz_mean'], sm_mhz_end=w['sm_mhz_end'], mem_mhz=w['mem_mhz'], wrong=bad)
            rows.append(row)
            print('rep %d %-7s %5d launches  %6.1f W  idle %4.1f W  %6.0f Mray/s  %6.1f nJ/ray (net of idle %6.1f)  SM %4.0f MHz  wrong %d' % (rep, c, w['launches'], row['W_total'], pw / 1e3, row['rays_per_us'], row['nJ_per_ray'], row['nJ_per_ray_net'], row['sm_mhz_mean'], bad), flush=True)
            json.dump(dict(scene=scene_n, rays=N, window_s=win, rows=rows), open(os.path.join(work, 'energy_bounce_s%d.json' % scene_n), 'w'), indent=1)
    for f in os.listdir(M):
        if f.startswith('e_'): os.remove(os.path.join(M, f))
    print('median of %d reps (rays per us | nJ per ray total | nJ per ray net of idle | speed and energy per ray relative to pixel):' % reps)
    med = {c: {k: st.median(r[k] for r in rows if r['cfg'] == c) for k in ('rays_per_us', 'nJ_per_ray', 'nJ_per_ray_net', 'W_total', 'sm_mhz_mean')} for c in cfgs}
    for c in cfgs:
        m = med[c]; print('%-7s %7.1f  %7.2f  %7.2f   time %.2fx  energy %.2fx (net %.2fx)   %5.1f W  %4.0f MHz' % (c, m['rays_per_us'], m['nJ_per_ray'], m['nJ_per_ray_net'], m['rays_per_us'] / med['pixel']['rays_per_us'], med['pixel']['nJ_per_ray'] / m['nJ_per_ray'], med['pixel']['nJ_per_ray_net'] / m['nJ_per_ray_net'], m['W_total'], m['sm_mhz_mean']))


if __name__ == '__main__': main(*sys.argv[1:6])
