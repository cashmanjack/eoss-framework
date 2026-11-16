"""Mochi (collision detection, github MDurgaKeerthi/Mochi-DCD-on-RT, s23-3Dparticlesim, 100,000 particles, patched copy /dev/shm/mochi/src_copy) predictions BEFORE the measured run.
    /usr/bin/python3 model/predict_mochi.py DUMP_DIR [--md OUT.md]     (replay of every frame's refitted GAS dump; CPU only)
Input per frame f: gas_f%02d.bin (GAS after the refit of the last substep) and particles_f%02d.bin (Particle: center 3, velocity 3, radius, mass; launch index = row).
Boxes: center -/+ 2 radius (boundsProg). Rays: origin = center, direction (0,0,1), tmin 0, tmax 1e-16. Replay: model/replay/replay_aabb.cpp (ASSIGN=/dev/shm/rp/assign, binary /dev/shm/rp/replay_aabb).
Hardware candidates = 1.006 x primitives in entered leaf slots (F125). Time: F127 fixed part, c0 110 and cq 4.9 cycles per candidate (+1) per SM, 142 SMs at 2.5 GHz (assumed clock)."""
import os, subprocess, sys, glob
import numpy as np
D = sys.argv[1]; md = sys.argv[sys.argv.index('--md') + 1] if '--md' in sys.argv else None
HERE = os.path.dirname(os.path.abspath(__file__)); W = '/dev/shm/mochi/pred_%s' % os.path.basename(D.rstrip('/')); os.makedirs(W, exist_ok=True)
SM, CLK = 142, 2.5e9
rows = []
for gf in sorted(glob.glob(D + '/gas_f*.bin')):
    f = int(gf[-6:-4]); P = np.fromfile(D + '/particles_f%02d.bin' % f, np.float32).reshape(-1, 8); N = len(P)
    c, r = P[:, :3], P[:, 6:7]; lo, hi = c - 2 * r, c + 2 * r
    np.stack([lo, hi, lo], 1).astype(np.float32).tofile(W + '/f%02d.tris' % f)
    subprocess.run([sys.executable, HERE + '/replay/make_table.py', gf, W + '/f%02d.tris' % f, W + '/f%02d.table' % f], env=dict(os.environ, ASSIGN='/dev/shm/rp/assign'), check=True, capture_output=True)
    R = np.zeros((N, 8), np.float32); R[:, :3] = c; R[:, 5] = 1; R[:, 7] = 1e-16; R.tofile(W + '/f%02d.rays' % f)
    subprocess.run(['/dev/shm/rp/replay_aabb', W + '/f%02d.table' % f, W + '/f%02d.rays' % f, str(N), '0', W + '/f%02d.rep' % f], check=True)
    rep = np.fromfile(W + '/f%02d.rep' % f, np.int32).reshape(-1, 8); cand = rep[:, 2] * 1.006
    w = cand[: N // 32 * 32].reshape(-1, 32); eff = w.sum() / (32 * w.max(1).sum()); q = rep[:, 2].mean() + 1
    ms = (110 + 4.9 * q) * N / (SM * CLK) * 1e3
    rows.append((f, rep[:, 0].mean(), rep[:, 1].mean(), rep[:, 2].mean(), cand.mean(), rep[:, 7].mean() + 1, eff, ms))
out = ['| frame | nodes / query | leaf slots / query | primitives in slots / query | hardware candidates / query | exact box hits (incl. self) / query | lane efficiency | RT part of the launch, ms (F127, 142 SMs at 2.5 GHz) |', '|---|---|---|---|---|---|---|---|']
out += ['| %d | %.2f | %.2f | %.2f | %.2f | %.2f | %.3f | %.4f |' % r for r in rows]
t = '\n'.join(out); print(t)
if md: open(md, 'x').write(t + '\n')
