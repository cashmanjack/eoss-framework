# BITTU timing with probes on one warp only (2026-11-19, triton GPU 1, RTX 6000 Ada, checked GAS group, 1 block, scene tri1, ray hit; file summary_w_20261119T101542.json)

Corrected = window(stage 1 under the tool) - window(stage 0 twin under the tool), both with the probed warp only recording; harness = stage 1 minus stage 0 group time of the same warp from the plain run (cycles). "5 launches" = median over launches 2 to 5 of the 5 launch run; "1 launch" = the 1 launch run. Non probed: median over the other warps of the group time (stage 1 minus stage 0 harness clocks) in the plain run and in the instrumented run (both stubs and probes of the probed warp present).

| case (warps, probed warp) | launches | window with group | window twin | probe cost | harness group | corrected | corrected / harness | non probed warps, plain | non probed warps, under the tool |
|---|---|---|---|---|---|---|---|---|---|
| gas_w2_p0 | 1 | 10565 | 9641 | 1446 | 1901 | 924 | 0.49 | 1713 | 1316 |
| gas_w2_p0 | 5 | 10516 | 9648 | 1444 | 1901 | 867 | 0.46 | 1713 | 1368 |
| gas_w2_p1 | 1 | 10541 | 9807 | 1460 | 1711 | 734 | 0.43 | 1899 | 1288 |
| gas_w2_p1 | 5 | 10493 | 9801 | 1462 | 1712 | 692 | 0.40 | 1900 | 1231 |
| gas_w4_p0 | 1 | 10991 | 9765 | 1429 | 2365 | 1226 | 0.52 | 2146 | 2226 |
| gas_w4_p0 | 5 | 11020 | 9773 | 1428 | 2011 | 1247 | 0.62 | 2287 | 3562 |
| gas_w4_p3 | 1 | 11240 | 9779 | 1435 | 2174 | 1461 | 0.67 | 2420 | 3683 |
| gas_w4_p3 | 5 | 10840 | 9745 | 1436 | 2245 | 1096 | 0.49 | 2393 | 3446 |
| gas_w8_p0 | 1 | 12031 | 10623 | 1479 | 2372 | 1408 | 0.59 | 2706 | 1874 |
| gas_w8_p0 | 5 | 11612 | 10060 | 1602 | 2304 | 1552 | 0.67 | 2636 | 2954 |
| gas_w8_p7 | 1 | 11908 | 10912 | 1667 | 2677 | 996 | 0.37 | 2708 | 4366 |
| gas_w8_p7 | 5 | 12106 | 9663 | 1558 | 2619 | 2444 | 0.93 | 2501 | 10931 |
| gas_w16_p0 | 1 | 14263 | 11943 | 2492 | 3289 | 2320 | 0.71 | 2875 | 2356 |
| gas_w16_p0 | 5 | 14343 | 23963 | 1989 | 3342 | -9620 | -2.88 | 2716 | 683 |
| gas_w16_p15 | 1 | 14714 | 13377 | 1976 | 2323 | 1337 | 0.58 | 2583 | 2122 |
| gas_w16_p15 | 5 | 14896 | 17717 | 1488 | 2929 | -2821 | -0.96 | 2708 | -1728 |

Cases passing the rule (median ratio within 0.95 to 1.05): 0 of 8; within 0.90 to 1.10: 1 of 8. All counts and decodes held: True.
Ratios (5 launch run): gas_w2_p0 0.46, gas_w2_p1 0.40, gas_w4_p0 0.62, gas_w4_p3 0.49, gas_w8_p0 0.67, gas_w8_p7 0.93, gas_w16_p0 -2.88, gas_w16_p15 -0.96
