# Model audit: which constants came from identical lanes, and does the lane sharing term predict F197 (2026-11-17, no GPU, no refit)

Script model/audit_lane_sharing.py. Sources checked in the repo (runner arguments, README, findings); "identical" = one ray in every lane of the warp (no ray file or --blockrays).

## 1. Provenance of the constants
| Constant | Value | Source | Lanes |
|---|---|---|---|
| a_S1, b_q, b_F, b_D32, c_qW (model A, notes/performance_model.md) | 1.094, 20.6, 5.15, 14.2, 31.2 | Memory_4 one SM, soup 1e5 and 3e6, six ray orders, 4/8/16 warps (rays from files) | distinct |
| A, b1, C, g, kappa (F123 sharing form, notes/q11/reuse_feature.json) | 0.92, 1.21, 475, 2.1, 1 | OptiX 4070 Ti, random and bounce rays, 12 training points (speedups over natural order) | distinct |
| c0, cq (F127 custom primitive fixed part) | 50 to 420, 1.8 to 8.8 | RTNN OptiX launches, 12 configurations | distinct |
| payload and attribute registers (F112) | | OptiX random and coherent rays | both |
| F67 line: 934 + 343 node + 113 slot + 52 triangle | | Latency_34, one warp, par:N:s scenes, no ray file | identical |
| fixed cost 740, 43 per level (F25) | | same ray traced twice | identical |
| Throughput_1 / F23, F24 (rate per warp count, about 12 cycles per ray at 16 warps) | | same ray in every lane | identical |
| unit capacity U (F105: 0.14 / 0.153 rays per cycle per TPC), sibling gain 4 percent (F62) | | tri1, par64 | identical |
| F194 fit (4.7 + 1.3 node + 0.06 slot + 0.67 triangle) | | Unit_1, one ray per launch lane | identical |
| tri1 and par64 rows of Throughput_6 | | one coherent ray per lane | identical |
| soup rows of Throughput_6 (F62, F63) | | --blockrays, own ray per lane | distinct (not fitted, only compared) |
| L2_CEIL, GPC_CEIL, L2 capacity, memory latencies | | plain loads and chases | no rays |
| F28 (144 triangle scene, different rays per lane, no change) | | parallel rays, each lane a different triangle on the same node path | distinct rays, shared node path |

So the throughput and sharing terms of the model (Memory_4, F123, F127) come from distinct rays; every latency, per element and unit rate constant (F67, F25, F23, F24, F105, F62 short, F194) comes from identical lanes. The model has NO constant that links the two: its unit term c_qW (62 cycles per ray at 32 warps, q = 1) was fitted on distinct rays and is applied to identical lanes too.

## 2. Does the existing model's lane sharing term predict F197? No.
F123's sharing term divides only the MISS cost (S1 x (1 - hit)) by the lane fetches per warp fetch. The fixed part is the same for coherent and distinct lanes. Cycles per ray per SM, 32 warps, q = 1, measured fills as the miss input (the model's best case):

| soup | A coh / inc / ratio | B (F123 form) coh / inc / ratio | measured coh / inc / ratio |
|---|---|---|---|
| 1,000 | 64.6 / 112 / 1.73 | 58.2 / 60.9 / 1.05 | 11.9 / 58.3 / 4.9 |
| 10,000 | 65.5 / 142 / 2.16 | 58.2 / 75.7 / 1.30 | 15.3 / 68.8 / 4.5 |
| 100,000 | 66.1 / 165 / 2.50 | 58.2 / 103 / 1.78 | 16.5 / 97.9 / 5.9 |
| 1,000,000 | 66.6 / 184 / 2.76 | 58.2 / 123 / 2.12 | 19.7 / 89.5 / 4.5 |

- Predicted incoherent over coherent: 1.05 to 2.1 (B), 1.7 to 2.8 (A); measured 4.5 to 5.9. Even with every fetch a miss and sharing ratio 1 (upper bound of the term) B reaches only 1.9 to 3.1.
- The ratio is wrong because of the COHERENT side: both forms predict 58 to 67 cycles per ray for identical lanes where 12 to 20 are measured (3 to 5 times too high). The incoherent absolute level is close for B (1.04, 1.10, 1.06, 1.37 times measured) and 1.7 to 2.1 times high for A.
- W = 32 is outside the Memory_4 fit range (4 to 16): c_qW q W / 16 is an extrapolation, so even the absolute level carries that caveat.
- One warp (32 lanes, distinct rays against the F67 line): A at W = 1 gives 2,700 / 5,800 / 10,500 / 14,000 against 6,576 / 15,524 / 19,165 / 17,591 measured; the F67 line (3,550 to 6,556) is the right coherent value but is 1.9 to 3.3 times low for distinct lanes. B has no latency form, so no prediction.
- Reading: the 4.5 to 5.9 times and 2.7 to 3.3 times are NOT predicted by the sharing term (it cannot: it is a memory term). Which constants need distinct ray measurements: every unit term used for light queries (the F194 per node and per triangle rates, c_qW at one query per ray, the F67 per element costs at one warp). Proposed runs below.

## 3. Proposed runs (not done; no refit)
1. Lane groups (this session, predictions first): g = 2, 4, 8, 16 lanes per ray, same four scenes, 32 warps and one warp. Tells whether the unit's per element cost scales with distinct rays per warp.
2. Distinct rays, one shared node path (F28 style, parallel rays on a scene where all lanes take the same nodes): separates "distinct ray values" from "distinct node sets".
3. F67 per element costs with distinct lanes: the Latency_34 par scenes with a different parallel ray per lane (one warp), to get the node, slot and triangle step costs for distinct rays.
4. F194 style rate sweep (1 to 4,096 triangles) with distinct parallel rays per lane, to get the per triangle and per node rates for distinct lanes.
5. c_qW at one query per ray: Memory_4 style launch at W = 1 to 32 with q = 1 rays (the fit has q 1.9 to 2.2 and W up to 16).
