# RT-BarnesHut first hit replay (queue 13), checked against ncu RT unit sectors
Method: patched copy dumps the GAS, its triangles and the program's real ray sequence (CPU emulation of the rope loop, count equal to the device recorder: 158.32 and 222.98 traversals per body); model/replay/replay.cpp FIRSTHIT=1 stops at the first leaf slot with a hit. Prediction file committed first: model/predictions_bh_replay_4070ti.md.
| bodies | traversals | nodes / traversal | leaf slots | triangles tested | hit fraction | ncu RT unit L1 sectors / traversal | per node (nodes + 1 instance level) |
|---|---|---|---|---|---|---|---|
| 100,000 | 15.83 M | 3.90 | 1.386 | 2.88 | 0.645 | 3.18 | 0.65 |
| 1,000,000 | 222.98 M | 4.59 | 1.358 | 2.69 | 0.703 | 4.39 | 0.79 |
The committed prediction (19.6 to 22.4 sectors per traversal, 4 sectors per node) FAILED: measured 3.18. Sectors per node fetch are not 4 and not constant: the 1M over 100k growth is 1.38 measured against 1.14 from the replay's nodes (instance level included). So the replay's counts are not directly the RT unit's L1 request count here; a calibration (as the model's KK) or an RT unit side node cache is needed. Replay counts themselves (nodes, triangles) were not verified independently: the hardware does not expose them for triangles.
