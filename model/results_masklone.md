# Lone masked instance: results (2026-11-16, tethys GPU 1; OptiX only; predictions model/predictions_masklone.md, pushed 427b319 before the run)
Raw: notes/flags1/t6_*_20261116T22*.json / .ncu.csv / .ncu32.csv / .summary.json (private). Program ttu/bench/Flags_1 (ias:256:V|R16|L16|E|Q, raymiss=1, ray mask 1). One run each, no fault, GPU idle after.

| scene | instances (visible) | one warp cold | warm | L1 fill sectors, 32 rays | RT unit L1 requests per ray | ns per ray (262,144 rays) |
|---|---|---|---|---|---|---|
| V all visible | 256 (256) | 106,817 | 57,448 | 785 | 3.85 | 5.48 |
| R16 every 16th absent | 240 (240) | 104,839 | 55,605 | 777 | 4.22 | 5.19 |
| L16 every 16th masked | 256 (240) | 103,376 | 54,394 | 785 | 3.88 | 5.22 |
| E odd masked | 256 (128) | 82,137 | 33,332 | 785 | 4.71 | 2.87 |
| Q even only | 128 (128) | 54,663 | 29,465 | 417 | 1.26 | 2.81 |

Reading (rule from the predictions file: (L16 - R16) / (V - R16)): sectors 8 / 8 = 1.0 but V - R16 is only 8 sectors (1 percent), so the rule has no resolution here; time: L16 is 1.4 percent (cold) and 2.2 percent (warm) FASTER than R16, so a lone masked instance costs nothing measurable beyond the tree shape differences between R16 and L16 (the two trees differ by 9 percent in RT L1 requests with equal visible sets). At throughput L16 - R16 = +0.6 percent time.
Interleaved (E against Q, 128 visible each): fills equal V (785, not Q's 417): masked instances cost the same fill sectors as visible ones (368 for 128 instances either way, 2.9 each); one warp time +50 percent cold and +13 percent warm over Q (V over Q: +95 and +95); at 262,144 rays only +2 percent time (V over Q +95). RT unit L1 requests per ray are higher than V's.

Predictions: 1 missed (visible instance 2.9 sectors, not 5 to 13; V 785 not 1,200 to 3,500). 2 (H1, L16 within 3 sectors and 3 percent of R16) held in time (-1.4 / -2.2 / +0.6 percent), missed by 5 sectors in fills (+8). 3 (H2, +16 to +64 sectors) missed low. 4 (E within 3 percent of Q) MISSED for one warp (+50 / +13 percent, fills +88 percent), held at throughput (+2 percent). 5 (R16 6 percent below V) missed: 1 percent in sectors, 3 percent in time.
Not separated: whether the extra fills of E are IAS nodes or instance nodes; why fills scale so little with instance count between V and R16 (16 fewer instances, 8 fewer sectors) but 368 between V and Q (the lines are shared, the tree shape changes).
