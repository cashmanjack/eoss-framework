"""BVH_4 host check (OptiX only; CPU reference with real instance transforms).
    ias_check.py WORK SOUP_N K MODE N_RAYS       (needs /dev/shm/b4/ias, ias_dev.ptx; GPU 1 idle; no TTU code)
Rays: prep.make_rays seed 2026 (random origins in [-1,1]^3, random directions). The CPU reference transforms each ray into every instance's object space
(inverse of the 3x4 object to world matrix), tests all triangles in float64 (Moller Trumbore, direction not normalized so t is the same in both spaces), and keeps the nearest hit.
Compared with OptiX: hit flag, instance id (100 + k), primitive index, t (rel 1e-4), barycentrics (abs 1e-3).
"""
import os, subprocess, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..', 'Memory_4'))
import prep  # noqa: E402
work, soup, K, mode, nr = sys.argv[1:6]; K = int(K); nr = int(nr); os.makedirs(work, exist_ok=True)
R = prep.make_rays(nr); rf = work + '/rays.bin'; R.tofile(rf); pre = work + '/ias_%s_%d_%s' % (soup, K, mode)
env = dict(os.environ, CUDA_VISIBLE_DEVICES='1', LD_LIBRARY_PATH=os.path.expanduser('~/opt/nvoptix-580.178.04') + ':' + os.environ.get('LD_LIBRARY_PATH', ''))
if subprocess.run(['nvidia-smi', '-i', '1', '--query-compute-apps=pid', '--format=csv,noheader'], capture_output=True, text=True).stdout.strip(): sys.exit('GPU 1 busy')
r = subprocess.run(['timeout', '-k', '2', '60', '/dev/shm/b4/ias', '/dev/shm/b4/ias_dev.ptx', soup, str(K), mode, rf, pre], env=env, capture_output=True, text=True)
if r.returncode: sys.exit('host failed %d\n%s\n%s' % (r.returncode, r.stdout, r.stderr[-1500:]))
print(r.stdout.strip())
out = np.fromfile(pre + '.out.bin', np.uint32).reshape(-1, 8); tris = np.fromfile(pre + '.tris.bin', np.float32).reshape(-1, 3, 3).astype(np.float64)
xf = np.fromfile(pre + '.xf.bin', np.float32).reshape(K, 3, 4).astype(np.float64)
best = np.full(nr, np.inf); bprim = -np.ones(nr, np.int64); binst = -np.ones(nr, np.int64); bu = np.zeros(nr); bv = np.zeros(nr)
A = tris[:, 0]; E1 = tris[:, 1] - A; E2 = tris[:, 2] - A
for k in range(K):
    M = xf[k, :, :3]; T = xf[k, :, 3]; Mi = np.linalg.inv(M)
    o = (R[:, 0:3].astype(np.float64) - T) @ Mi.T; d = R[:, 4:7].astype(np.float64) @ Mi.T
    for s in range(0, nr, 64):
        oo = o[s:s + 64, None, :]; dd = d[s:s + 64, None, :]
        pv = np.cross(dd, E2[None]); det = (E1[None] * pv).sum(2)
        with np.errstate(all='ignore'):
            inv = 1.0 / det; sv = oo - A[None]; u = (sv * pv).sum(2) * inv; qv = np.cross(sv, E1[None]); vv = (dd * qv).sum(2) * inv; t = (E2[None] * qv).sum(2) * inv
        ok = (np.abs(det) > 1e-12) & (u >= 0) & (vv >= 0) & (u + vv <= 1) & (t > 0)
        t = np.where(ok, t, np.inf); j = t.argmin(1); tm = t[np.arange(len(j)), j]
        sl = slice(s, s + 64); upd = tm < best[sl]
        best[sl] = np.where(upd, tm, best[sl]); bprim[sl] = np.where(upd, j, bprim[sl]); binst[sl] = np.where(upd, 100 + k, binst[sl])
        bu[sl] = np.where(upd, u[np.arange(len(j)), j], bu[sl]); bv[sl] = np.where(upd, vv[np.arange(len(j)), j], bv[sl])
hit = out[:, 0] != 0; chit = np.isfinite(best); ot = out[:, 1].view(np.float32).astype(np.float64)
bad_hit = int((hit != chit).sum()); both = hit & chit
bad_inst = int((out[both, 7 - 0] != binst[both]).sum()) if False else int((out[both, 6] != binst[both]).sum())
bad_prim = int((out[both, 2] != bprim[both]).sum()); bad_t = int((np.abs(ot[both] - best[both]) > 1e-4 * best[both]).sum())
ou = out[:, 3].view(np.float32).astype(np.float64); ov = out[:, 4].view(np.float32).astype(np.float64)
bad_uv = int(((np.abs(ou[both] - bu[both]) > 1e-3) | (np.abs(ov[both] - bv[both]) > 1e-3)).sum())
ids = np.unique(binst[both], return_counts=True)
print('rays %d: OptiX hits %d, CPU hits %d, hit flag mismatches %d; on %d common hits: instance id mismatches %d, primitive %d, t %d, barycentrics %d; hits per instance %s' % (nr, hit.sum(), chit.sum(), bad_hit, both.sum(), bad_inst, bad_prim, bad_t, bad_uv, dict(zip(ids[0].tolist(), ids[1].tolist()))))
c3 = np.fromfile(pre + '.c3.bin', np.uint64)
print('launch constant words (offset: value), nonzero:', {hex(8 * i): hex(int(x)) for i, x in enumerate(c3) if x})
