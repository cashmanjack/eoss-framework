"""Prepare step for Unit_3 style runs and the lane group runs (predictions: model/predictions_lane_groups.md). CPU only (+ g++), no GPU.
    /usr/bin/python3 prepare_groups.py DATA        DATA = /dev/shm/t9u holding gas_<N>.bin and tris_<N>.bin (stage 0 launches with --dump-gas / --dump-tris), N = 1000 ... 1000000
Per size: replay table (assign mode 1, exact), 40,000 random rays (origin uniform in [-1,1]^3, uniform direction, seed 2026 + N), keeps the rays that hit nothing (exact replay), first 1,024 of them = miss<N>.bin (32 chunks of 32 lanes:
every warp runs the chunk's 32 rays, lane i = ray i), coh<N>.bin (the ray with counts nearest the mean in all 1,024 lanes), grp<g>_<N>.bin (chunk c, lane l = ray c*32 + l // g, g = 2 4 8 16 32: groups of g consecutive lanes share a ray),
and the 32 ray versions miss32_<N>.bin, coh32_<N>.bin, grp<g>_32_<N>.bin (the first chunk). Writes counts_<N>.json (mean nodes, slots, triangles of the selected rays and of every group file's distinct rays)."""
import json, os, subprocess, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); RP = os.path.join(HERE, '..', '..', '..', 'model', 'replay')
D = sys.argv[1]; SIZES = (1000, 10000, 30000, 100000, 300000, 1000000); GS = (2, 4, 8, 16, 32)
for exe in ('assign', 'replay'):
    if not os.path.exists('%s/%s' % (D, exe)): subprocess.run(['g++', '-O2', '-o', '%s/%s' % (D, exe), '%s/%s.cpp' % (RP, exe)], check=True)
for n in SIZES:
    tab = '%s/table_%d.bin' % (D, n)
    if not os.path.exists(tab):
        subprocess.run([sys.executable, RP + '/make_table.py', '%s/gas_%d.bin' % (D, n), '%s/tris_%d.bin' % (D, n), tab], check=True, env=dict(os.environ, ASSIGN=D + '/assign', ASSIGN_MODE='1'))
    g = np.random.default_rng(2026 + n); NC = 40000
    o = g.uniform(-1, 1, (NC, 3)); d = g.normal(size=(NC, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
    R = np.zeros((NC, 8), np.float32); R[:, 0:3] = o; R[:, 3] = 0; R[:, 4:7] = d; R[:, 7] = 1e16
    R.tofile('%s/cand_%d.bin' % (D, n)); out = '%s/cand_%d.out' % (D, n)
    subprocess.run(['%s/replay' % D, tab, '%s/cand_%d.bin' % (D, n), str(NC), '0', out], check=True)
    r = np.fromfile(out, np.int32).reshape(NC, 8)
    nohit = np.where(r[:, 5] < 0)[0]; assert len(nohit) >= 1024, (n, len(nohit))
    sel = nohit[:1024]; S = R[sel]; cnt = r[sel][:, :3].astype(float); mean = cnt.mean(0)
    rep = int(np.argmin((((cnt - mean) / mean) ** 2).sum(1)))
    S.tofile('%s/miss%d.bin' % (D, n)); S[:32].tofile('%s/miss32_%d.bin' % (D, n))
    np.repeat(S[rep:rep + 1], 1024, 0).tofile('%s/coh%d.bin' % (D, n)); np.repeat(S[rep:rep + 1], 32, 0).tofile('%s/coh32_%d.bin' % (D, n))
    info = dict(n=n, hit_free_candidates=int(len(nohit)), mean_counts=[float(x) for x in mean], rep=rep, rep_counts=[int(x) for x in r[sel][rep][:3]], groups={})
    for gg in GS:
        idx = np.array([c * 32 + l // gg for c in range(32) for l in range(32)]); S[idx].tofile('%s/grp%d_%d.bin' % (D, gg, n)); S[idx[:32]].tofile('%s/grp%d_32_%d.bin' % (D, gg, n))
        info['groups'][gg] = [float(x) for x in cnt[np.unique(idx)].mean(0)]      # mean counts of the distinct rays used
    json.dump(info, open('%s/counts_%d.json' % (D, n), 'w')); print(n, info)
