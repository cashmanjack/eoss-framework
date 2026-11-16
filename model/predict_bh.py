"""RT-BarnesHut (OWL, third party code in ~/OWLRayTracing, patched copy in /dev/shm/bh, diff in notes/q12) on the 4070 Ti: predictions BEFORE measuring.
Inputs: the CPU walk of the program's own octree (BH_PREDICT: per point in launch order visits, approximated nodes, leaves entered, leaf interactions), the opening rule of the host code
(approximate when node size < distance x 0.5). A traversal (one optixTraverse + invoke) per visited node is assumed (a visited node is one hit or miss event of the rope loop).
Time: total traversals x c / (60 SMs x 2.5 GHz), c cycles per traversal per SM in [7.4, 64] (best unit rate of Finding 3 ... per random launch rate of the 4070 Ti, Finding 117 data: 122 cycles per ray at 1.9 queries); point c = 20.
Lane efficiency: sum of visits over lanes / (32 x warp maximum), warps of 32 consecutive launch positions.
    /usr/bin/python3 model/predict_bh.py > model/predictions_bh_4070ti.md"""
import numpy as np
print('# RT-BarnesHut predictions (written before any GPU measurement), 2026-11-15, RTX 4070 Ti (io GPU 0)\n')
print('Workload: Gaussian disc bodies (sigma 100, 100, 20), masses 1 to 100, theta 0.5 (the code\'s THRESHOLD), bucket 32. Launch order = DFS order of the bucket tree (orderPointsDFS).\n')
print('| bodies | BH nodes | visits per point (mean / max) | approximated per point | leaves per point | leaf interactions | lane efficiency | traversals total | predicted launch ms (c 7.4 / 20 / 64) |')
print('|---|---|---|---|---|---|---|---|---|')
for tag, n, nodes in (('g100k', 100000, 4531), ('g1M', 1000000, 44214)):
    r = np.fromfile('/dev/shm/bh/pred_%s.bin' % tag, np.int32).reshape(-1, 4); v = r[:, 0].astype(float)
    w = v[: len(v) // 32 * 32].reshape(-1, 32); eff = w.sum() / (32 * w.max(1).sum()); tot = v.sum()
    t = lambda c: tot * c / (60 * 2.5e9) * 1e3
    print('| %d | %d | %.1f / %d | %.1f | %.1f | %.1f | %.3f | %.3g | %.2f / %.2f / %.2f |' % (n, nodes, v.mean(), v.max(), r[:, 1].mean(), r[:, 2].mean(), r[:, 3].mean(), eff, tot, t(7.4), t(20), t(64)))
print('\nWhat can fail: (1) iterations per point differ from visits (a hit can skip nodes, a miss may cover several; test: recorder count within 25 percent); (2) lane efficiency (test: within 0.1);')
print('(3) time outside the bracket (the per iteration shading work, 150 byte node loads, is not in c); (4) the acceleration structure (one triangle per BH node) fits the L1 for 100,000 bodies (4,531 triangles), not for 1,000,000.')
print('Shading work not in the bracket: leaf interactions x ~40 instructions over 60 SMs x 128 lanes x 2.5 GHz = %.2f ms (100,000) and %.2f ms (1,000,000) if fully overlapped; the launch is predicted at the larger of traversal and shading plus a quarter of the smaller.' % (100000 * 851.4 * 40 / (60 * 128 * 2.5e9) * 1e3, 1000000 * 1024.0 * 40 / (60 * 128 * 2.5e9) * 1e3))
print('Not predicted: nodes and primitives per traversal (needs a first-hit replay mode with the real ray sequence; not built).')
