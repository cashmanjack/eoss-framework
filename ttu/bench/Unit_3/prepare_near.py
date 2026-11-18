"""Phase 10i test 2 (model/predictions_phase10i.md): 'near' ray files. CPU only (+ the exact replay built by prepare_groups.py).
    /usr/bin/python3 prepare_near.py DATA      DATA = /dev/shm/t9u after prepare_groups.py and prepare_class.py
Per size: base = ray 0 of the class file cls1_32_<N>.bin (the 32 distinct class rays, same counts). near ray k (k = 0..31) = base with the origin moved by k x EPS across the direction, so the 32 rays are different float values with the same path.
The exact replay must give every near ray the counts of the class (nodes, slots, triangles); if not, EPS is divided by 3 (down to the float ulp) and the check repeated; a size whose near rays cannot be made equal is skipped (reported).
near<D>_<N>.bin = 1,024 lanes (the D ray pattern, 32 / D lanes per ray, tiled 32 times), near<D>_32_<N>.bin = the 32 lane pattern; D = 2 4 8 16 32. Writes near_<N>.json."""
import json, os, subprocess, sys
import numpy as np
D = sys.argv[1]
for n in (100000, 1000000):
    cls = json.load(open('%s/class_%d.json' % (D, n)))['cls']; S = np.fromfile('%s/cls1_32_%d.bin' % (D, n), np.float32).reshape(32, 8)
    base = S[0].copy(); d = base[4:7].astype(np.float64); ax = np.argmin(np.abs(d)); e = np.zeros(3); e[ax] = 1
    u = np.cross(d, e); u /= np.linalg.norm(u); eps = 1e-6; ok = False
    while eps >= 5e-8:
        R = np.tile(base, (32, 1)); R[:, 0:3] = (base[0:3].astype(np.float64)[None, :] + np.arange(32)[:, None] * eps * u[None, :]).astype(np.float32)
        assert len({tuple(r[0:3]) for r in R}) == 32, 'origins not distinct at eps %g' % eps
        R.tofile('%s/nearchk_%d.bin' % (D, n)); out = '%s/nearchk_%d.out' % (D, n)
        subprocess.run(['%s/replay' % D, '%s/table_%d.bin' % (D, n), '%s/nearchk_%d.bin' % (D, n), '32', '0', out], check=True)
        r = np.fromfile(out, np.int32).reshape(32, 8)
        if (r[:, :3] == np.array(cls)).all() and (r[:, 5] < 0).all(): ok = True; break
        eps /= 3
    if not ok: print(n, 'SKIPPED: near rays cannot be made equal in counts (last eps %g)' % eps); continue
    for dd in (2, 4, 8, 16, 32):
        pat = R[np.arange(32) // (32 // dd)]; pat.tofile('%s/near%d_32_%d.bin' % (D, dd, n)); np.tile(pat, (32, 1)).tofile('%s/near%d_%d.bin' % (D, dd, n))
    json.dump(dict(n=n, cls=cls, eps=eps, base_origin=[float(x) for x in base[:3]], u=[float(x) for x in u]), open('%s/near_%d.json' % (D, n), 'w')); print(n, 'near files written, eps', eps, 'counts', cls)
