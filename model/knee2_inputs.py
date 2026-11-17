"""Knee blind test inputs (2026-11-17). CPU only.  /usr/bin/python3 model/knee2_inputs.py DATA OUT.json
Per size: node fetch sequences of the 32 class rays from the root (cls1_32_<N>.bin) and from the nearest internal child of the root (STARTCHILD, the 'shifted' lane), D(g) = distinct node lines per step per warp
(mean over steps and max over steps) for the plain g ladder and for the shifted variant (per group of g lanes sharing a ray, the first g/2 lanes start at the root, the other g/2 at the root's child on the ray's path:
the same distinct rays, one step apart); writes start files sh<g>_<N>.txt (1,024 lines, lane pattern tiled; 'root' or the root slot of the child) for g = 2 4 8 16 32."""
import json, os, struct, subprocess, sys
import numpy as np
D, OUT = sys.argv[1], sys.argv[2]; res = {}; G = (1, 2, 4, 8, 16, 32)
for n in (1000, 10000, 30000, 100000, 300000, 1000000):
    tab = '%s/table_%d.bin' % (D, n); ray = '%s/cls1_32_%d.bin' % (D, n)
    def seqs(path, env):
        subprocess.run(['%s/replay' % D, tab, ray, '32', '0', path + '.out'], check=True, env=dict(os.environ, NODESEQ=path, **env))
        a = np.fromfile(path, np.int32); i = 0; out = []
        for _ in range(32):
            c = a[i]; ev = a[i + 1:i + 1 + c]; i += 1 + c; out.append([int(x) for x in ev if x >= 0])
        return out, np.fromfile(path + '.out', np.int32).reshape(32, 8)
    sp, op = seqs('%s/seq_%d.bin' % (D, n), {}); sc, oc = seqs('%s/seqc_%d.bin' % (D, n), {'STARTCHILD': '1'})
    nn = struct.unpack('<3i', open(tab, 'rb').read(12))[0]
    rec = np.frombuffer(open(tab, 'rb').read(12 + 32 * 12), dtype=[('t', '<i4'), ('c', '<i4'), ('b', '<f4', 6)], offset=12, count=12)   # root slots
    slot_of = {int(c): int(s) for s, (t, c) in enumerate(zip(rec['t'], rec['c'])) if t == 1}
    first = [s[0] for s in sc]; assert all(f in slot_of for f in first), n
    assert (oc[:, 5] < 0).all()                     # still no hit from the child
    info = dict(plain={}, shifted={}, counts_root=[float(x) for x in op[:, :3].mean(0)], counts_child=[float(x) for x in oc[:, :3].mean(0)])
    for g in G:
        lanes = [sp[l // g] for l in range(32)]; K = max(map(len, lanes)); per = [len({x[k] for x in lanes if len(x) > k}) for k in range(K)]
        info['plain'][g] = dict(mean=float(np.mean(per)), max=max(per))
        if g >= 2:
            lanes = [(sc if (l % g) >= g // 2 else sp)[l // g] for l in range(32)]; K = max(map(len, lanes)); per = [len({x[k] for x in lanes if len(x) > k}) for k in range(K)]
            info['shifted'][g] = dict(mean=float(np.mean(per)), max=max(per))
            pat = ['%d' % slot_of[first[l // g]] if (l % g) >= g // 2 else 'root' for l in range(32)]
            open('%s/sh%d_%d.txt' % (D, g, n), 'w').write(''.join(p + '\n' for p in pat * 32))
    res[n] = info
    print(n, 'counts root %s child %s' % (np.round(info['counts_root'], 1), np.round(info['counts_child'], 1)), 'plain mean', {g: round(v['mean'], 1) for g, v in info['plain'].items()}, 'shifted mean', {g: round(v['mean'], 1) for g, v in info['shifted'].items()})
json.dump(res, open(OUT, 'x'))
