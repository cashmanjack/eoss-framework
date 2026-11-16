"""kNN growing radius: active set per round from the replay of the program's own rule (queue 13). OptiX build only (predict mode of the patched program, no search launch), GPU 0.
The heap holds every candidate the hardware hands to the intersection program (heap starts at the largest float), so a query stops when its candidates (primitives in entered leaf slots of the
replay, F125 ratio 1.006) reach K. Round r+1 active set = queries with fewer than K in round r. Compare with the measured active sets (200000, 200000, 200000, 197660, 83).
    /usr/bin/python3 model/knn_rounds.py   (needs /dev/shm/knn/knn_optix, rand.csv, pts_rand.bin, /dev/shm/rp/replay_aabb)"""
import os, subprocess, sys, shutil, numpy as np
K = '/dev/shm/knn/'; W = K + 'rounds_rule'; shutil.rmtree(W, ignore_errors=True); os.makedirs(W + '/act'); os.makedirs(W + '/dump')
env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', CUDA_DEVICE_ORDER='PCI_BUS_ID', KNN_DUMP=W + '/dump', KNN_ACTIVE_DIR=W + '/act')
np.arange(200000, dtype=np.int32).tofile(W + '/act/active_r1.bin')
meas = [200000, 200000, 200000, 197660, 83]
for rd in range(1, 8):
    if not os.path.exists(W + '/act/active_r%d.bin' % rd): break
    subprocess.run(['timeout', '-k', '2', '120', K + 'knn_optix', 'rand.csv', 'rand.csv', '16', '0.004'], cwd=K, env=env, capture_output=True)
    D = W + '/r%d' % rd; os.makedirs(D, exist_ok=True)
    for a, b in (('gas0.bin', 'dump/gas_r%d.bin' % rd), ('points.bin', '../pts_rand.bin'), ('aabb_radius0.txt', 'dump/radius_r%d.txt' % rd), ('rays0.bin', 'dump/rays_r%d.bin' % rd)):
        if os.path.lexists(D + '/' + a): os.remove(D + '/' + a)
        os.symlink(W + '/' + b, D + '/' + a)
    subprocess.run(['/usr/bin/python3', 'model/replay/rtnn_prep.py', D, D + '/rp'], check=True, capture_output=True)
    N = os.path.getsize(D + '/rays0.bin') // 32
    subprocess.run(['/dev/shm/rp/replay_aabb', D + '/rp.table', D + '/rays0.bin', str(N), '0', D + '/rep.bin'], check=True)
    r = np.fromfile(D + '/rep.bin', np.int32).reshape(-1, 8); t = r[:, 2]; act = np.fromfile(W + '/act/active_r%d.bin' % rd, np.int32)
    nxt = act[t < 16]; nxt2 = act[t * 1.006 < 16]
    print('round %d active %d (measured %s) replay candidates mean %.3f stop %d -> next active %d (x1.006 rule: %d)' % (rd, N, meas[rd - 1] if rd <= 5 else '-', t.mean(), (t >= 16).sum(), len(nxt), len(nxt2)), flush=True)
    nxt.tofile(W + '/act/active_r%d.bin' % (rd + 1)) if len(nxt) else None
    if not len(nxt): break
