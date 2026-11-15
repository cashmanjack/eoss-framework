# Scene only predictions for RANDOM rays on the 4070 Ti (io GPU 0), written 2026-11-15 BEFORE any ordered launch was timed
Pre registered check of the performance model (code: ttu/bench/Morton/random_check.py predict; same path as model/predict_bounce.py, Finding 113). Two soups not used before (300,000 and 1,500,000 triangles; BVH 18.6 and 93.1 MiB against the 48 MiB L2) and one real scene (Bistro Exterior, McGuire archive OBJ, 2,833,329 triangles, BVH 175.9 MiB; re downloaded to /dev/shm, not in git). 8,388,608 random rays per scene: origin uniform in the scene box (the soup box [-1,1]^3), direction uniform on the sphere, tmax 1e16. Orders: natural (random), Morton of the origin (5 bits per axis), oracle grouping by the hit subtree at depth 1, 2, 3, 4 and in full. Predicted quantity: speedup over the natural order of the trace time.
Inputs: performance model parameters fitted on the RTX 6000 Ada (Memory_4 one SM data), k set on triton random 1e5 rays, CPU replay for queries per ray and reuse distance histograms (60 SMs, 60 KB L1), L2 48 MiB. Nothing fitted on the 4070 Ti. The GPU was used once per scene to build and dump the GAS (one plain trace of the natural order, its time not read) before this file was written.
| scene | hit fraction | queries per ray (replay) | Morton origin k5 | oracle d1 | d2 | d3 | d4 | full |
|---|---|---|---|---|---|---|---|---|
| soup 300,000 (BVH 18.6 MiB) | 0.878 | 1.90 | 1.40 | 1.13 | 1.27 | 1.35 | 1.39 | 1.39 |
| soup 1,500,000 (BVH 93.1 MiB) | 0.927 | 1.96 | 1.51 | 1.09 | 1.24 | 1.40 | 1.46 | 1.51 |
| Bistro Exterior (BVH 175.9 MiB) | 0.351 | 1.31 | 0.95 | 1.06 | 1.25 | 1.26 | 1.27 | 1.27 |

Groups per depth (d1/d2/d3/d4/full): s3e5 13/157/1885/17349/22080; s15e5 13/157/1885/22402/111582; bistro 13/155/1411/6997/39564

Measured with the same script, step `measure`, after this file is committed. Error reported as the rms of log(predicted / measured) over the 6 orders per scene and over all 18, next to the rms of the measured log speedups (the spread to be explained).
