"""Knee map inputs (2026-11-17): distinct node lines per warp for each configuration, from the replay's node fetch sequences. CPU only.
    /usr/bin/python3 model/knee_inputs.py DATA OUT.json     DATA = /dev/shm/t9u (cls1_32_<N>.bin = the 32 class rays, table_<N>.bin, replay)
For g lanes per ray (lane l has ray l // g), a warp's node fetches by step k (k-th internal node fetched by each lane, lanes with fewer nodes drop out):
  step_max  = max over k of the distinct node ids at step k      (primary statistic of H)
  step_mean = mean over k of the distinct node ids at step k
  total     = distinct node ids over the warp's whole traversal"""
import json, os, subprocess, sys
import numpy as np
D, OUT = sys.argv[1], sys.argv[2]; res = {}
for n in (1000, 10000, 100000, 1000000):
    sq = '%s/seq_%d.bin' % (D, n)
    subprocess.run(['%s/replay' % D, '%s/table_%d.bin' % (D, n), '%s/cls1_32_%d.bin' % (D, n), '32', '0', '%s/seq_%d.out' % (D, n)], check=True, env=dict(os.environ, NODESEQ=sq))
    a = np.fromfile(sq, np.int32); i = 0; seqs = []
    for _ in range(32):
        c = a[i]; ev = a[i + 1:i + 1 + c]; i += 1 + c; seqs.append([int(x) for x in ev if x >= 0])
    res[n] = {}
    for g in (1, 2, 4, 8, 16, 32):
        lanes = [seqs[l // g] for l in range(32)]; K = max(len(x) for x in lanes)
        per = [len({x[k] for x in lanes if len(x) > k}) for k in range(K)]
        res[n][g] = dict(step_max=max(per), step_mean=float(np.mean(per)), total=len({v for x in lanes for v in x}), steps=K)
    print(n, {g: (v['step_max'], round(v['step_mean'], 1), v['total']) for g, v in res[n].items()})
json.dump(res, open(OUT, 'x'))
