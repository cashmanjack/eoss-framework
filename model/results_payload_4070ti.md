# Throughput_10: payload and attribute size, RTX 4070 Ti (io, GPU 0), 2026-11-15
OptiX closest hit, 1,048,576 rays on the 100,000 triangle soup (triangles, and the same soup's tight boxes as custom primitives with a box test intersection program). Payload values 0 to 32 (all live: filled by closest hit / miss, folded into the result), attribute values 0 to 8. Random rays (origin in the cube) and coherent camera rays. Code ttu/bench/Payload; data notes/io/payload/payload.json (not in git). ncu on one optixLaunch (128 thread blocks): registers per thread, mean active warps per SM, local memory bytes per ray. Hit, t and primitive of every triangle run with at least 3 payload values equal the payload 8 run (12 runs, 0 mismatching).

| payload values | attr option | registers | warps/SM (random / coherent) | random Mrays/s | coherent Mrays/s | local stores per ray (random / coherent) |
|---|---|---|---|---|---|---|
| 0 | 2 | 103 | 13.6 / 12.7 | 1455 | 4763 | 24.0 / 15.8 B |
| 2 | 2 | 108 | 13.6 / 12.7 | 1456 | 4742 | 24.0 / 15.8 B |
| 8 | 2 | 114 | 13.6 / 12.7 | 1455 | 4755 | 24.0 / 15.8 B |
| 16 | 2 | 122 | 13.6 / 12.7 | 1456 | 4741 | 24.0 / 15.8 B |
| 32 | 2 | 138 | 10.3 / 9.6 | 1407 | 3889 | 24.0 / 15.8 B |
| 0 | 8 | 115 | 13.6 / 12.7 | 1454 | 4747 | 24.0 / 15.8 B |
| 16 | 8 | 132 | 10.3 / 9.6 | 1407 | 3882 | 24.0 / 15.8 B |
| 32 | 8 | 148 | 10.3 / 9.6 | 1407 | 3880 | 24.0 / 15.8 B |

Custom primitives (box test): 2 to 8 payload values 108 to 124 registers, 14.1 / 12.7 warps, 2.2 Grays/s random (2.17 to 2.21), 4.74 coherent; 32 payload values 138 to 148 registers, 10.6 / 9.7 warps, 1.80 random, 3.77 coherent; local stores 66.1 B per ray random (33.0 coherent), loads 33.6 (12.9) at every setting except attribute 4 with payload 2 (77.3 / 44.9 random, once), no change with attributes 0 to 8.
Reading: each payload value costs about 1 register (103 to 122 from 0 to 16 values, 138 at 32; each attribute value about 1.5 registers on triangles, about 1.4 on custom primitives), no spill: local memory traffic is the same 24 B per ray (triangles) at 0 and at 32 payload values (it belongs to the traversal call, not to the payload), the 255 register limit is not reached. The cost is occupancy: crossing 128 registers per thread (65,536 registers over 4 blocks of 128 threads) drops a SM from 4 to 3 resident blocks (13.6 to 10.3 active warps), which costs 3% for random rays (memory latency bound, plenty of other warps) and 18% for coherent rays (compute / issue bound: 4.76 to 3.88 Grays/s). The step is at 16 payload values with 8 attributes, 32 with 2. Time does not depend on the count of payload values below the step.
