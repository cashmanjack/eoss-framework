# RT-BarnesHut: why the RT unit's L1 sectors per traversal are 3.2 and 4.4, not 19.6 to 22.4 (queue 14, 4070 Ti)
Counter checked: ncu on the OptiX launch (probes off). Sectors of the RT unit = l1tex__t_sectors minus l1tex__t_sectors_pipe_lsu. All L1 requests counted on this launch are LSU requests (l1tex__t_requests = l1tex__t_requests_pipe_lsu, l1tex__t_requests_pipe_tex = 0), so the RT unit's sectors have no request count: sectors per request is not available for it.
| bodies | traversals | RT unit sectors | per traversal | of which tag hits | of which misses |
|---|---|---|---|---|---|
| 100,000 | 15,832,132 | 50.24 M | 3.173 | 1.50 | 1.67 |
| 1,000,000 | 222,980,000 | 978.5 M | 4.388 | 1.71 | 2.68 |
Test (post hoc on 100,000 bodies, held out on 1,000,000): requests of the lanes of a warp that ask for the same node in the same traversal step are served once. model/q14/warp_share.cpp replays the real ray sequence (first hit termination, 32 bodies per warp in launch order, lane l runs its i-th traversal in loop iteration i) and counts fetches per raw lane traversal:
| bodies | variant | node fetches | instance node | leaf slots | sectors (4 per node and instance, 2 per leaf slot) | measured |
|---|---|---|---|---|---|---|
| 100,000 | raw (no merging; the failed prediction) | 3.897 | 1.000 | 1.386 | 22.4 | 3.173 |
| 100,000 | same node, same iteration and step across lanes | 0.534 | 0.036 | 0.491 | 3.26 | 3.173 |
| 100,000 | same node, same iteration, any step | 0.468 | 0.036 | 0.461 | 2.94 | 3.173 |
| 100,000 | each id once per warp (an ideal warp cache) | 0.023 | 0.000 | 0.047 | 0.19 | 3.173 |
| 1,000,000 | same node, same iteration and step (3 windows of 1,000 warps, traversal weighted: 0.690, 0.0345, 0.482; windows 0.81 / 0.67 / 0.59 nodes) | 0.690 | 0.035 | 0.482 | 3.86 | 4.388 |
Reading: lane merging at the same step brings the failed prediction from a factor 7.1 (100,000) and about 5.7 (1,000,000) too high to +2.7 percent and -12 percent; an ideal per warp cache would give 0.19, so the unit merges what the lanes of a warp ask for at the same time, it does not keep a warp's history. The leaf slot size is not identified (1.8 sectors per merged leaf slot fits 100,000 exactly; 1 to 2 plausible). Not explained: the growth from 100,000 to 1,000,000 bodies is 1.38 measured and 1.18 predicted (the window to window spread of the merged node count is 0.59 to 0.81, so the 1,000,000 prediction has about 15 percent of uncertainty from the sample). The lane sharing ratio (raw over merged node fetches) is 7.3 at 100,000 and 6.7 at 1,000,000. Whether the merge is in the RT unit or in its request path cannot be told from L1 counters.
Files: model/q14/warp_share.cpp, notes/q14/bh_sectors_*.txt (ncu counters).
