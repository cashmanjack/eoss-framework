"""Second prepare step for the lane group runs (design correction after the first run, see model/predictions_lane_groups.md addendum). CPU only.
    /usr/bin/python3 prepare_class.py DATA     (after prepare_groups.py: tables and cand_<N>.out exist; 400,000 more candidates for the two large sizes)
Per size: among the rays that hit nothing, find the (nodes, slots, triangles) class closest to the F197 mean counts that holds at least 32 rays; take 32 rays of that class (same counts by construction).
cls<g>_<N>.bin = 1,024 rays (the 32 ray pattern tiled 32 times: lane l = class ray l // g), cls<g>_32_<N>.bin = the 32 ray pattern; g = 1 2 4 8 16 32. Writes class_<N>.json."""
import collections, json, os, subprocess, sys
import numpy as np
D = sys.argv[1]; F197 = {1000: (5.95, 2.69, 5.2), 10000: (9.3, 3.8, 7.7), 30000: (10.4, 3.5, 7.95), 100000: (11.7, 4.4, 9.0), 300000: (13.4, 4.5, 9.5), 1000000: (13.6, 4.7, 9.7)}   # 30,000 and 300,000: counts interpolated in log size between the neighbouring classes (blind test sizes)
for n, tgt in F197.items():
    tab = '%s/table_%d.bin' % (D, n); NC = 400000 if n >= 30000 else 40000
    g = np.random.default_rng(31337 + n); o = g.uniform(-1, 1, (NC, 3)); d = g.normal(size=(NC, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
    R = np.zeros((NC, 8), np.float32); R[:, 0:3] = o; R[:, 4:7] = d; R[:, 7] = 1e16
    R.tofile('%s/cand2_%d.bin' % (D, n)); out = '%s/cand2_%d.out' % (D, n)
    subprocess.run(['%s/replay' % D, tab, '%s/cand2_%d.bin' % (D, n), str(NC), '0', out], check=True)
    r = np.fromfile(out, np.int32).reshape(NC, 8); nh = np.where(r[:, 5] < 0)[0]
    cls = collections.defaultdict(list)
    for i in nh: cls[tuple(r[i, :3])].append(i)
    best = min((k for k, v in cls.items() if len(v) >= 32), key=lambda k: sum(((a - b) / b) ** 2 for a, b in zip(k, tgt)))
    ids = cls[best][:32]; S = R[ids]
    for gg in (1, 2, 4, 8, 16, 32):
        pat = S[np.arange(32) // gg]; pat.tofile('%s/cls%d_32_%d.bin' % (D, gg, n)); np.tile(pat, (32, 1)).tofile('%s/cls%d_%d.bin' % (D, gg, n))
    json.dump(dict(n=n, cls=[int(x) for x in best], size=len(cls[best]), nohit=int(len(nh)), target=tgt), open('%s/class_%d.json' % (D, n), 'w')); print(n, 'class', best, 'has', len(cls[best]), 'rays; no hit rays', len(nh), 'target', tgt)
