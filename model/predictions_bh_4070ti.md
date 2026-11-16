# RT-BarnesHut predictions (written before any GPU measurement), 2026-11-15, RTX 4070 Ti (io GPU 0)

Workload: Gaussian disc bodies (sigma 100, 100, 20), masses 1 to 100, theta 0.5 (the code's THRESHOLD), bucket 32. Launch order = DFS order of the bucket tree (orderPointsDFS).

| bodies | BH nodes | visits per point (mean / max) | approximated per point | leaves per point | leaf interactions | lane efficiency | traversals total | predicted launch ms (c 7.4 / 20 / 64) |
|---|---|---|---|---|---|---|---|---|
| 100000 | 4531 | 158.3 / 257 | 93.6 | 26.6 | 851.4 | 0.877 | 1.58e+07 | 0.78 / 2.11 / 6.76 |
| 1000000 | 44214 | 223.0 / 352 | 146.7 | 32.0 | 1024.0 | 0.909 | 2.23e+08 | 11.00 / 29.73 / 95.14 |

What can fail: (1) iterations per point differ from visits (a hit can skip nodes, a miss may cover several; test: recorder count within 25 percent); (2) lane efficiency (test: within 0.1);
(3) time outside the bracket (the per iteration shading work, 150 byte node loads, is not in c); (4) the acceleration structure (one triangle per BH node) fits the L1 for 100,000 bodies (4,531 triangles), not for 1,000,000.
Shading work not in the bracket: leaf interactions x ~40 instructions over 60 SMs x 128 lanes x 2.5 GHz = 0.18 ms (100,000) and 2.13 ms (1,000,000) if fully overlapped; the launch is predicted at the larger of traversal and shading plus a quarter of the smaller.
Not predicted: nodes and primitives per traversal (needs a first-hit replay mode with the real ray sequence; not built).
