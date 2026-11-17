# CLOVER (non RT baseline) against the RT methods: predictions, written before any CLOVER timing run (2026-11-17, triton GPU 1, plain CUDA)

Programs: CLOVER warpwise, hubs, and bitonic with the one line patch (threshold lane (K-1) mod 32; F221) as the third variant; stock bitonic is also timed where it is right.
Datasets and k as the RT rows of F207 / F218: TrueKNN rows = 3DRoad 434,874 / Porto first 1M / uniform 1M, k 5, every point is a query. RTNN rows = bunny 362,272 and dragon 3,609,600, k 50, every point is a query
(bunny full; dragon: 100,000 random queries, time scaled by 36.1, because linear scan time is linear in queries; linearity is checked on bunny).
Time = CLOVER's own printed span (device synchronize, kernel or kernels, for hubs including the index build), no host copies. RT times = F207 / F218 rows (fixed TrueKNN: launches and wall; RTNN: search compute time, a radius query, see below).
Correctness: 1,000 sampled queries per run against a CPU double precision brute force with a tie tolerance of 1e-6 (as clover_check2).

Caveat written before: RTNN with a radius returns at most K points inside r. At the F207 radii the queries have 1 to 172 candidates, so RTNN did NOT solve exact K = 50 kNN there. TrueKNN k = 5 rows are exact kNN in intent, but the exactness check found 3.4 / 7.9 / 5.6 percent of queries wrong.
So "RT wins" is about speed at different answer quality; both are reported.

## Predictions (ranges are my 80 percent interval)
1. Warpwise time, k 5, all queries: 3DRoad 0.8 to 5 s, uniform 1M 4 to 30 s, Porto 1M 4 to 30 s (about 1e12 pairs at 20 to 100 G pairs per second; time proportional to N x Q).
2. Hubs against warpwise: 3 to 50 times faster on 3DRoad and Porto (2D, clustered), 1.5 to 10 times on uniform 3D 1M (no clustering).
3. Patched bitonic within a factor 3 of warpwise either way.
4. Correctness: warpwise 0 wrong sampled queries, hubs 0, patched bitonic 0 at k 5 and k 50. Stock bitonic also 0 wrong at k 5 and 50 (lane K mod 32 is a later element than the K-th here: looser only, slower).
5. RT against CLOVER at k 5 (TrueKNN fixed wall: 85.8 ms road, 65.1 ms uniform, 1,282 ms Porto): the RT method wins by more than 10 times over warpwise on road and uniform, and by 3 to 30 times on Porto (its heavy round tail). Against hubs: RT wins on uniform (more than 2 times), is within a factor 3 either way on road, and LOSES on Porto (hubs faster than 1.28 s).
6. k 50, bunny (362,272 queries): warpwise 0.3 to 2 s; RTNN's search compute (0.3 to 0.4 ms) is far below it but returns about 1 to 13 candidates per query, not 50 neighbours: reported as not comparable.
7. Dragon scaled to 3.6M queries: warpwise 50 to 600 s; hubs 5 to 100 s.
8. Scaling check on bunny: warpwise time at Q = 36,000 times 10.06 is within 15 percent of the full time.
Success rule: a numeric prediction counts as held if the measurement is inside the interval; the winner statements (5) count as held if the order is as written.
