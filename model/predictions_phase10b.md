# Predictions, 2026-11-17 (tethys, RTX 4090 GPU 1): divergence profile, opacity micromaps, concurrent kernels, transform sequence

Committed before any run of these four items. Smoke runs (tiny launches to check that code works) are allowed; any number read from them is disclosed in the results file. Rule for all timings: enough rays that the run to run spread (max minus min over repeats, over the median) is under 3 percent; if not, the cell is marked noisy and not scored.

## 1. Divergence profile (Div_1)

Probe: in every intersection or any hit callback the leader lane adds to per warp bins by popcount(__activemask()): callbacks, callback cycles, and the interval cycles from the previous callback exit (or raygen start) to this callback exit, attributed to this callback's lane count. A handoff is a callback whose entry is at least 300 cycles after the previous exit of the same warp (the unit round trip is above 800 cycles, callbacks inside one handoff follow within tens). Handoff lanes = the lane count of the first callback of the handoff. "Few lanes" = 8 or fewer. Warp time = raygen start to raygen end of the warp, so the share of warp time with few lanes = interval cycles in bins 1 to 8 over total warp time. Datasets: only what exists on tethys (synthetic; the triton real datasets are not here): RTNN bunny style point sets are not here, so RTNN on a uniform 100,000 point set and a clustered one, kNN (own), Arkade fixed on synthetic 3D, TrueKNN on uniform 100,000 and 400,000, Mochi frame 0 (100,000 particles).

Predictions (mean active lanes per callback, share of warp time with few lanes):
- RTNN: mean 22 to 27 of 32 (lane efficiency 0.7 to 0.85, as F125 and F126 measured from counts); few lane time 3 to 15 percent.
- kNN own growing radius: round 1 mean 24 to 30; later rounds (small active sets) mean below 16; whole run few lane time 5 to 25 percent.
- Arkade fixed: mean 14 to 24; few lane time 10 to 35 percent (callbacks are 73 to 95 percent of its time, F220, so lane count matters most here).
- TrueKNN: mean 8 to 18; few lane time 20 to 50 percent (late rounds have a few lanes with long candidate lists, F216).
- Mochi frame 0: mean 26 to 31 (1.075 calls per query, most queries have only the self candidate); few lane time under 5 percent.
- Ranking of few lane time: TrueKNN > Arkade > kNN > RTNN > Mochi (at least the extremes TrueKNN highest and Mochi lowest).
- Handoff lanes are within 2 of the callback lanes for Mochi and RTNN (one callback per handoff), and clearly below callback lanes for Arkade (several callbacks per handoff).
- Probe perturbation: kernel time with probes within 15 percent of the unprobed build (earlier probes: F133).

## 2. Opacity micromaps (Omm_1)

Scene: soup of 100,000 triangles (Struct_1 soup), random rays, closest hit. Alpha pattern: per triangle a disc in barycentric space is transparent (hole), the rest opaque; four patterns by primitive id mod 4, hole radii 0.15, 0.25, 0.35, 0.45 around the point (1/3, 1/3) (the transparent fraction of each pattern is computed exactly in the host and reported). Micro triangle states are classified exactly (opaque if the nearest point of the micro triangle to the disc centre is outside the disc, transparent if the farthest vertex is inside, else unknown), so OMM plus any hit must reproduce plain any hit exactly. Variants: D opaque (any hit disabled, no alpha), A plain any hit (alpha tested in the program), B(L) OMM 4 state at subdivision L = 2, 4, 6 with the same any hit program for unknown states, C(L) OMM with unknown encoded as opaque (any hit disabled, approximate).
- B(L) hit, primitive and t equal A on all rays (0 mismatches). C(L) differs from A on 1 to 10 percent of rays at L 2 falling to under 3 percent at L 6 (unknown area is the only error).
- Any hit invocations per ray, as a fraction of A: B(2) 0.5 to 0.9, B(4) 0.12 to 0.3, B(6) 0.03 to 0.09 (the unknown fraction falls about 2 times per level). C(L): 0.
- A costs more than D: time A over D 2 to 6 (each invocation is an SM round trip, F49 and F133 scale, about 850 cycles).
- Time over A: B(2) 0.6 to 0.95, B(4) 0.35 to 0.65, B(6) 0.25 to 0.5; C(6) within 1.0 to 1.5 times D (the micromap fetch costs something, not zero). If OMM is handled in software on this GPU the invocation ratio would be 1.0: that is a separate outcome I would report.
- SM warp instructions per ray (ncu) fall about with the invocations (within 20 percent of the invocation ratio plus a fixed part).
- GAS size grows by under 10 percent for the shared micromap array (4 arrays of 4^L x 2 bits each).

## 3. Concurrent kernels (Conc_1)

Our own checked TTU kernel (Throughput_1 / TPC_1 build, G blocks of 32 warps, random rays in soup:100,000 and coherent tri1) beside a plain persistent kernel (setup3 --compute), as in F92 to F101, on the 4090. Both directions measured: RT rate against RT alone, compute iterations per second against compute alone. New placements for the compute kernel: "high" (SMs 64 to 127, other TPCs than the 32 RT blocks on SMs 0, 2, ..., 62) and "sib" (the odd SMs 1, 3, ..., 63, the TPC siblings of the RT SMs). Compute types: alu, l1 (32 KB window, 4 blocks), mem (1 GB stream). Existing placements "odd" and "all" kept for continuity. RT G = 32 for the new placements, so the RT work is the same in both.
- mem, random rays: RT rate 0.02 to 0.2 of alone in both high and sib, the two within 25 percent of each other (L2 and DRAM are chip wide, F94 and F101).
- mem, coherent rays: RT 0.3 to 0.8 of alone, high and sib within 25 percent.
- l1 at 4 blocks: RT 0.7 to 1.0 on sib (shares the SM pair), 0.95 to 1.05 on high (nothing shared but the L2): this is the discriminating test between siblings and other TPCs.
- alu: RT 0.95 to 1.05 in both placements.
- Compute side (mem, high or sib): 0.85 to 1.0 of its alone rate (RT with 32 blocks asks for little bandwidth against 1 GB streaming); RT random rays slow the compute side less than RT slows under it.
- If the compute kernel does not run beside the TTU kernel on this GPU (as in F211) I stop and report that instead.

## 4. Transform TTU sequence, prepare only (Xform_1)

The IAS group of F139 and F195 is the sequence; the only inputs that change with a non identity instance are in the data (the instance leaf in the IAS holds the object to world and world to object matrices, F85). Prepare: setup3 --ias gets an --imat option (rotation about z, uniform scale, translation, the OptiX build input matrix), the register image words are recomputed from the live IAS, build_patch --ias builds the sequence, check_seq passes in both stages, stage 0 (no TTU instruction) runs with OptiX through the transformed IAS and a CPU float64 reference agreeing (hit, t, primitive, barycentrics), and the traversal group words in the register image are compared with the identity case. Stage 1 is NOT run.
- check_seq ALL PASS for the transformed build, bytes of the TTU sequence identical to the identity IAS build (same group, same fuse count 15, close bit 74 on the last load only).
- The words that change between identity and transform images: none of the ray words; the IAS root offset word may differ if the IAS size differs (it should not).
- OptiX (through the transformed IAS) equals the CPU reference on all hits of a small ray set (as F85: all hits, t within 1e-4 relative).
- Plan to commit with the results: what stage 1 would run (one warp, one ray, then rays that hit and miss the rotated triangle, 90 degree and 37 degree rotations, translation, scale 2), the success criterion (hit, t, primitive, barycentrics equal OptiX; instance pair equals the instance id), and the approval it needs (new instance data, same sequence class as F139).
