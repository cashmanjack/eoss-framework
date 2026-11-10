"""BVH_5 latency: first query latency (TTUOPEN to last result, warp 0) per start node.

    latency.py OUT.json ...   (runs made with --repeat N; launch 0 is dropped)
"""
import json, statistics as st, sys

for p in sys.argv[1:]:
    j = json.load(open(p))
    L = [l['warp_clk'][0][1] - l['warp_clk'][0][0] for l in j['launches'][1:]]
    same = all(l['same_as_first'] and l['lanes_equal'] for l in j['launches'])
    q = sorted(L)
    print('%-26s scene %-10s start %-5s flush %-4g | median %6.0f  p5 %6d  p95 %6d  (n %d) | results same every launch, all lanes equal: %s | best %s' % (
        p.split('/')[-1], j['scene'], j['start_path'], j['flush_mb'], st.median(L), q[int(0.05 * (len(q) - 1))], q[int(0.95 * (len(q) - 1))], len(L), same, j['lane_best'][0]))
