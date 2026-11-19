# RTX 5090 (pc14), session 9: spheres (F204 repeated) and the triangle rate against Ada, 2026-11-19

OptiX only (SDK 8.1 headers, driver 580.126.09), no TTU code of ours. Predictions model/predictions_5090_batch9.md (e613ab7, before any run). Programs ttu/bench/Sphere_1 (F204's, one `stack` scene and one `ENTRY_ONLY` variant added), runner ttu/bench/pc14/run_sphere_pc14.py (pc14_gpucheck.sh before every process, timeout 30, ncu 120). Analysis model/analyze_spheres.py with NSM 170, model/analyze_5090_batch9_ncu.py, model/analyze_5090_batch9_stack.py. Raw (private, timestamped) notes/pc14_sphere. All runs rc 0, no fault. SM clock 2.54 to 2.60 GHz (calibration launches). One run per cell, no cell repeated.

## A. Spheres
Ncu warp instructions per candidate (smsp__inst_executed over the 1,048,576 ray launch, N = 16, 64, 256, per warp, slope over N):
| arm | D1 | D1b | reading |
|---|---|---|---|
| built in sphere S | 0.0 (723 per warp at every N) | 0.0 | no SM instruction per candidate: the RT core |
| custom C | 211 | 202 | SM program (4090: 190 to 230) |
| T0, T1 | 0.0 | 0.0 | unit |
(S and the triangle arms have the same 723 per warp for every N: the raygen and the call itself. The L2 sectors per warp are also flat for S, 33 for every N, against 16 per candidate for C: the sphere data is not fetched by the SM. The 4090 S: 215 to 245.)

One warp, slope per candidate (cycles, identical lanes, N = 1 to 256) and saturation (cycles per ray per SM, 4,194,304 rays, NSM 170):
| arm | D1 one warp | D1b one warp | D1 saturation | D1b saturation |
|---|---|---|---|---|
| S | 171 (R2 0.993) | 170 | 6.03 (1.000) | 6.03 |
| C | 1,309 (1.000) | 1,258 | 3.05 | 2.87 |
| T0 | 3 (0.86, no work) | 414 (0.86) | 0.03 | 2.74 (0.88) |
| T1 | 97 (0.75) | 769 (0.94) | 0.33 | 5.40 (0.99) |
4090 (F204): S 1,341 and 1,312 one warp, 3.45 and 3.07 saturation; C 1,226 and 1,179, 3.00 and 2.78; T0 D1b 541 and 4.46; T1 D1b 849 and 7.24. So the built in sphere is 7.7 times cheaper than a custom primitive at one warp (4090: it was 0.9 times) and about 2.0 times MORE expensive at saturation when the 32 lanes are identical (4090: 1.15 times), because the unit tests every lane's ray itself (6 cycles per ray per SM per sphere) while the SM program serves 32 converged lanes with one call.

D2 (20,000 random spheres, random rays), saturation cycles per ray per SM, r = 0.02 / 0.04 / 0.08:
| arm | cycles | 4090 (F204) |
|---|---|---|
| S | 39.8 / 44.4 / 92.8 | 83.7 / 97.6 / 221.6 |
| C (both roots, as F204) | 93.9 / 107.5 / 254.8 | 86.2 / 102.7 / 262.7 |
| C entry root only (post hoc, below) | 96.9 / 123.3 / 279.6 | n m |
| T0 | 76.2 / 106.6 / 297.8 | 84.3 / 110.2 / 282.2 |
| T1 | 91.3 / 109.4 / 244.1 | 99.8 / 118.4 / 249.4 |
S / C(entry only) 0.41 / 0.36 / 0.33 (4090: S / C 0.97 / 0.95 / 0.84). GAS MB: S 1.4, C 0.7, T0 22.2, T1 88.8 (4090: 1.0, 0.7, 24.8, 99.4). The custom and tessellated arms cost the same per ray as on the 4090 (within 10 percent, 5 to 9 percent for C); only S moved, by a factor 2.1 to 2.4 faster than on the 4090 and 2.4 to 3.0 faster than a custom program here.

Correctness finding (unplanned): S and my custom program (F204's, which reports the exit root when the ray starts inside a sphere) agree on hit and sphere index for only 92.4, 64.2 and 49.6 percent of the rays at r 0.02, 0.04, 0.08 (4090: 99.998 to 100). Reason: for rays whose origin lies outside every sphere (checked on the first 20,000 rays, centres regenerated on the CPU) S and C agree 100 percent; for origins inside a sphere (7.9, 47.5, 99.2 percent of the rays) C reports the exit hit (hit 1.000) and S does not (hit 0.79, 0.93, 0.97: only entry hits of other spheres). The hardware sphere test returns the entry root only. Post hoc (after seeing this, disclosed) a custom arm with the entry root only (ENTRY_ONLY, sphere_entry_dev.ptx) was run on D2: agreement with S 100.000 percent at all three radii, t within 1e-4 relative for 99.99, 99.97, 99.79 percent of the rays that hit; its ncu count 211 per candidate (the same as C). So the 2.4 to 3.0 times above is like for like.

Reading: the built in sphere runs in the RT core on the 5090 (0 SM instructions per candidate, flat over N, no L2 traffic from the SM, 7.7 times lower one warp latency per candidate), as the whitepaper says. At saturation the picture depends on lane convergence: 2.0 times worse than a custom program for 32 identical lanes (6.0 against 3.05 cycles per ray per SM per sphere), 2.4 to 3.0 times better on random rays. Per sphere test the unit costs about 6 cycles per ray per SM, against 0.94 per stacked triangle (B), so a sphere candidate is worth about 6 stacked triangles of unit time at saturation but only 2 triangles at one warp (171 against 82). Not tested: the linear swept sphere (not in the 8.1 API), spheres with any hit, rays that start inside (the hardware does not report the exit), a Blackwell with the OptiX 9 API (the driver ran the sphere in hardware through the 8.1 call), kNN style workloads.

## B. Triangle rate
New scene `stack`: N triangles with the same xy at different z, the axis ray inside every box and outside every triangle, identical lanes, zero hits (checked). One warp and saturation:
| N | 1 | 4 | 16 | 64 | 256 | 1,024 | 4,096 |
|---|---|---|---|---|---|---|---|
| one warp cycles | 5,015 | 5,581 | 7,194 | 9,662 | 30,935 | 74,235 | 341,919 |
| saturation cycles per ray per SM | 14.52 | 16.22 | 34.37 | 65.42 | 263.2 | 841.3 | 3,877 |
Slope per stacked triangle by least squares: one warp 81.8 (all N, R2 0.997), 82.3 (N >= 64), 82.7 (N >= 256); saturation 0.941 (0.999), 0.948, 0.954; between consecutive N at saturation 0.57 to 1.51, 0.65 to 1.03 above N = 16. Saturation / one warp slope ratio 87. Intercept at saturation N = 1: 14.52 (Ada OptiX, F186: 14.4).
Ada comparison. (a) Same scene, same method (OptiX), Ada values from F204 and F186: one triangle 14.52 against 14.4 (1.01 times; overhead dominated); T0 and T1 near miss (D1b) per sphere 2.74 and 5.40 against 4.46 and 7.24 (1.63 and 1.34 times cheaper on the 5090, fits R2 0.88 and 0.99); D2 T0 0.90, 0.97, 1.06 times the 4090 cost per ray, T1 0.91, 0.92, 0.98, C 1.09, 1.05, 0.97 (equal within 10 percent: these runs are traversal and divergence bound). (b) Mixed method (OptiX here, the TTU run F194 on Ada, scenes par:N:s similar to the stack but not identical in tree shape): Ada 56.0, 192.8, 761, 3,122 cycles per ray per SM at N = 64, 256, 1,024, 4,096; the 5090 stack 65.4, 263, 841, 3,877 (1.17, 1.37, 1.11, 1.24 times SLOWER). Not a valid ratio (OptiX raygen on the 5090 side, the TTU loop on Ada; on the 5090 itself F407 gives 36.1 cycles per ray for par:64:s with the TTU loop against 65.4 for the OptiX stack, so the method gap is about 29 cycles at N = 64). (c) The 5090's own TTU result, existing (F407): par:64:s 0.0277 rays per cycle per SM against 0.0178 on Ada (1.56 times), short coherent tri1 gap rate 1.07 to 1.14 times. One warp: F67 52 cycles per triangle is a latency step; the stack's 82 per triangle includes node steps and cannot be set against it (second trace on the 5090 equals Ada, 790 against 782, F405).
Reading: the largest per triangle gains seen are 1.56 (TTU long coherent query, F407) and 1.34 to 1.63 (OptiX, near miss tessellated spheres); the one triangle query, the random ray workloads and the custom primitive show 1.0 times; no measurement here reaches the whitepaper's 2.0. The whitepaper figure is presumably the triangle test rate of the intersection hardware alone; in these scenes the unit also does box tests, node fetches and leaf slots (F194: node 1.3 cycles, triangle 0.67 on Ada), and the OptiX stack scene cannot separate them (the tree is the builder's). Not measured: the same stack scene on an Ada GPU (needs the same binary built for sm_89, ttu/bench/pc14/run_sphere_pc14.py stack); pure triangle term without nodes (needs our own sequence, not allowed here); hits.

## Predictions scored (model/predictions_5090_batch9.md)
Held: 1 (hardware branch, 45 percent: S 0 instructions, flat); 2 (C 211 and 202 instructions inside 150 to 280, one warp 1,309 and 1,258 inside 1,150 to 1,600, saturation 3.05 and 2.87 inside 2.3 to 3.5); 4 partly (one warp under 600: 171 held, central 350 missed by 51 percent; no instruction growth held; D2 S / C 0.42, 0.41, 0.36 inside 0.3 to 0.8 held); 6 (T hit fractions equal F204's); 7 (C D2 within 1.4 of the 4090: 1.09, 1.05, 0.97); 8 (C, T0, T1 GAS sizes within 30 percent; S 1.4 against 1.0 MB missed); 11 (one triangle saturation intercept 14.52 inside 50 percent of 14.4; 2x at 1.8 to 2.2 not seen, the 20 percent branch is the one that failed); 12 (saturation to one warp ratio 87 inside 80 to 160); 9 (not below 0.6 of F67: 1.57 of F67).
Missed: 4 saturation (S under 1.5, central 0.8: 6.03); 5 (S against C agreement 99.9 percent: 92.4, 64.2, 49.6; cause the exit root, entry only custom 100 percent); 6 (T D1b saturation slopes within 30 percent of Ada: T0 0.61 times, T1 0.75 times; T0 / C within 0.6 to 1.15: 1.17 at r 0.08); 9 (one warp slope 40 to 70: 82, ratio to F67 0.8 to 1.3: 1.57); 10 (saturation slope 0.35 to 0.55 and 0.35 to 0.65: 0.94, the "no gain" branch (25 percent) was closer; ratio central 1.5 not measured on a like for like scene).
