# Scene only prediction of grouping speedups on the RTX 4070 Ti, 2026-11-15
Code model/predict_bounce.py. Inputs per launch order come from the CPU replay over the decoded OptiX BVH (queries per ray q, lines requested and L1 misses per ray from the reuse distance histogram of one SM's fetch stream with a 60 KB L1, 60 SMs); the performance model parameters were fitted on the RTX 6000 Ada (Memory_4) and k (sectors per line request, 7.46) was set on triton's random 1e5 ray set. Nothing was fitted or tuned on the 4070 Ti; the code ran once per scene, no setting changed after seeing the result. The target GPU enters only through its SM count (60) and L2 size (48 MB, for the DRAM share of misses). Measured: bounce ray orders of Finding 107 (trace only, pixel order = natural).
Caveat: the script was written after the Finding 107 measurements existed (the earlier written prediction file covers the oracle gap only), so this is a no fit transfer test, not a pre-registered one.

| scene | order | predicted speedup over pixel order | measured |
|---|---|---|---|
| 100,000 tris (BVH 6.5 MB, q 1.43) | shuffled | 0.70 | 0.64 |
| | Morton origin k5 | 1.18 | 1.10 |
| | oracle depth 1 / 2 / 3 / full | 1.14 / 1.25 / 1.10 / 1.12 | 1.14 / 1.23 / 1.26 / 1.26 |
| 3 million tris (BVH 186 MB, q 1.55) | shuffled | 0.64 | 0.52 |
| | Morton origin k5 | 1.21 | 1.15 |
| | oracle depth 1 / 2 / 3 / full | 1.04 / 1.17 / 1.18 / 1.18 | 1.06 / 1.19 / 1.29 / 1.30 |

rms log error over the 6 orders: 0.087 (100,000), 0.103 (3 million), 0.095 over both. Against it: the same model on the random ray set on triton (Finding 100, set up with the same recipe and a different ray set) had 0.36. Misses: depth 3 and full oracle are predicted 8 to 14% low on both scenes (the model sees almost no gain between depth 2 and full, the chip shows 5 to 10% more), shuffled is predicted 9 to 19% too fast (too little penalty), Morton 5 to 8% too fast. The replay gives q = 1.43 and 1.55 queries per ray for these bounce rays (no measured q on the 4070 Ti for these rays to compare with; Finding 81 compared q on triton: 1.85 and 1.98 against 1.88 and 2.00).
