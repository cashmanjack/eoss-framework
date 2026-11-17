# Held out scene predictions with the F123 form, no refit (2026-11-17, tethys RTX 4090 GPU 1, written BEFORE any ordered launch was timed)

Question: does the traversal model in its F123 form (reuse feature plus warp level sharing, 'share' with kappa 1, parameters from notes/q11/reuse_feature.json, fitted once on soup 300,000 random rays and bounce rays on the 100,000 soup, 4070 Ti OptiX data) predict the speedup of ray orders on graphics scenes that were never in any fit, never in any earlier test, and never measured by us before?

Scenes (McGuire archive OBJ, downloaded 2026-11-17 to /dev/shm/hs, not in git; converted with ttu/bench/RealScene/scene2tris; degenerate triangles dropped):
- Crytek Sponza (sponza.obj): 262,266 triangles, BVH 16.3 MiB (below the 72 MiB L2 of the 4090), 93.4 percent of random rays hit.
- Conference Room (conference.obj): 124,619 triangles, BVH 7.7 MiB, 95.3 percent hit.
- Rungholt (rungholt.obj): 6,704,264 triangles, BVH 416.3 MiB (5.8 times the L2), 59.4 percent hit.
Never used before: not in the F123 fit sets (soups 300,000 and 1.5 million, Bistro random rays, bounce rays on the 100,000 and 3 million soups), not in F102 (Bistro, San Miguel, Power Plant, Lucy).

Method (the same code path as F117 and F123, only the GPU constants differ): ttu/bench/Morton/random_check.py predict (RC_NSM=128, RC_L2_MB=72; one plain trace per scene to build and dump the GAS, its time not read), 8,388,608 random rays per scene (origin uniform in the scene box, direction uniform on the sphere, tmax 1e16, seed 2026), CPU replay over the decoded OptiX BVH for q, node fetches, and the reuse distance histogram of one SM's fetch stream (128 SMs, 60 KB L1 = 480 lines), then model/predict_heldout.py (F123 'share' form, kappa 1, parameters unchanged). The 'share' form has no DRAM term (F123: its fitted weight was zero), so Rungholt, whose BVH is far above the L2, is a test of that omission. k (sectors per line request) was set on triton's random 1e5 ray set; nothing is fitted on tethys or on these scenes. Orders: natural (random order), Morton of the origin 5 bits, oracle depth 1, 2, 3, 4, full (needs the answer: a bound). Predicted quantity: speedup of the order over the natural order. Measured with random_check.py measure (10 launches per order, median, every result compared with the natural order, 'wrong' must be 0).

Predictions (F123 form; the Memory_4 form of the predict step in brackets, for comparison, also scored):
| scene | Morton k5 | oracle d1 | d2 | d3 | d4 | full |
|---|---|---|---|---|---|---|
| Sponza | 1.175 (1.187) | 1.115 (1.073) | 1.153 (1.131) | 1.159 (1.146) | 1.160 (1.149) | 1.160 (1.149) |
| Conference | 1.056 (1.203) | 1.053 (1.106) | 1.053 (1.106) | 1.053 (1.106) | 1.053 (1.106) | 1.053 (1.106) |
| Rungholt | 1.327 (1.299) | 1.143 (1.080) | 1.267 (1.157) | 1.347 (1.250) | 1.422 (1.361) | 1.504 (1.516) |
Replay inputs: Sponza q 1.41, lane to warp fetch ratio 1.5 to 2.2; Conference q 1.16, 1.9 to 2.7; Rungholt q 1.56, 1.5 to 3.0. Oracle d2 to full are identical for Sponza (d3 to full) and Conference (d1 to full) in the prediction because the replay's histograms are identical there (few distinct groups at depth).
Written expectations (before measuring): Sponza within 0.10 rms log error; Conference the one I expect to fail (small BVH, the model sees almost no gain, the measured gain of Morton may be larger, the Memory_4 form says 1.2); Rungholt: the F123 form under predicts the gain because it has no DRAM term, and its full oracle gain is the least certain; I expect rms log error 0.10 to 0.20 there.

Reading rules (written now):
- Error per scene = rms of log(predicted / measured) over the 6 orders; spread = rms of the measured log speedups. F123's own held out error was 0.08 (18 points). Pass for a scene: error at most 0.12 (1.5 times F123's). The pooled error over the 18 points is reported next to the 0.08.
- If the F123 form beats the Memory_4 form on pooled error it is reported as the better of the two on new scenes; if not, that is reported.
- No refit, no retuning after the measurement. A refit on these scenes would be a separate, labeled step.
- Random rays only (like the F117 Bistro row). Camera and bounce rays on these scenes are NOT tested here (the bounce ray code builds only soups); the result must not be read as covering them.
