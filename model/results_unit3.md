# Unit_3 results (2026-11-17, tethys GPU 1, RTX 4090; predictions model/predictions_unit3.md committed first)

Runs: stage 0 (largest scene, 32 warps), one smallest TTU launch (soup:1000, coherent, 1 warp, 1 launch), 16 stage 1 runs (4 sizes x incoherent / coherent x 1 and 32 warps, 30 launches), 8 more with ncu (l1tex__m_xbar2l1tex_read_sectors.sum, 32 warps, 2 launches). All exit 0, one block each, no fault, GPU idle after. The checked remap build of Unit_1, rays from files: 1,024 distinct random rays that hit nothing (exact replay) against one representative no hit ray in all lanes. Raw data notes/unit3 (private). Runner ttu/bench/Unit_3/run.py.
c = launch span / 1,024 (includes the ramp and tail of one 1,024 ray launch, as in F194).

| soup N | GAS | counts (nodes / slots / tris) | F194 pred | c coherent | c incoherent | incoherent / coherent | L1w coherent | L1w incoherent (F67 line) | extra sectors per ray (incoh minus coh) | M_est lines |
|---|---|---|---|---|---|---|---|---|---|---|
| 1,000 | 69 KB | 5.95 / 2.69 / 5.2 | 16.3 | 11.9 | 58.3 | 4.9 | 3,841 | 6,576 (3,550) | 2.5 | 2.9 |
| 10,000 | 0.65 MB | 9.3 / 3.8 / 7.7 | 22.6 | 15.3 | 68.8 | 4.5 | 5,622 | 15,524 (4,960) | 15.9 | 16.6 |
| 100,000 | 6.5 MB | 11.7 / 4.4 / 8.0 to 10.0 | 26.0 to 27.4 | 16.5 | 97.9 | 5.9 | 6,412 | 19,165 (5,870) | 41 | 30 |
| 1,000,000 | 65 MB | 13.6 / 4.7 / 8.6 to 10.9 | 28.9 to 30.4 | 19.7 | 89.5 | 4.5 | 7,043 | 17,591 (6,556) | 59 | 47 |
Sector baseline of the coherent run (setup loads and stores, about 37.9k sectors per launch, 37.9 to 38.0k at every size) is subtracted; M_est = (extra sectors per ray / 4) x 285 / c incoherent: a lower bound on lines in flight per SM, valid only if memory bound.

## Against the predictions
1. Coherent within 30 percent of the F194 fit on all four: MISSED (-27, -35, -38, -34 percent: the fit over predicts light queries, the node cost is lower than 1.34 here). F194's rates are not a transferable fit for 5 to 14 node queries.
2. Incoherent at 1,000 within 30 percent of c_U: MISSED (58 against 16, 3.6 times); 10,000 at 1.3 to 3 times c_U: borderline missed (3.0); 100,000 and 1,000,000 at 60 to 200: HELD (98, 90).
3. Incoherent / coherent at 1,000,000 at least 2: HELD (4.5), and it is 4.5 to 5.9 at EVERY size, including 1,000 where the 69 KB scene is in the node state and the extra fills are 2.5 sectors per ray.
4. Fills at 1,000,000 between 6 and 14 lines per ray: 14.7 lines (59 sectors): just above, MISSED by 5 percent. M_est 3 to 47 lines per SM: not a memory bound measurement (see reading).
5. One warp incoherent within 25 percent of the F67 line: MISSED (1.85, 3.1, 3.3, 2.7 times); coherent within 25 percent: HELD (+7, +7, +8, +4 percent).

## Reading
- Incoherent rays cost 4.5 to 5.9 times coherent ones at the same counts, already when the BVH is in the L1 node state and almost nothing is fetched (soup:1,000). So most of the difference is not memory: the unit does about 3.3 to 4.2 cycles per element (node, slot or triangle) for distinct rays against about 0.7 per triangle for identical rays. F194's rates are rates for identical lanes; they must not be used for distinct rays. A likely cause (hypothesis, not tested): the unit does the work of a warp's identical lanes once. A test: lanes that agree in groups of 2, 4, 8, 16 (same counts, shrinking number of distinct rays).
- Memory: the cost grows with elements (58 to 98 cycles for 13.9 to 24 elements) and the fills grow 20 times (2.5 to 59 sectors) while c grows 1.5 times, so the one SM is not memory bound at these sizes: the unit sustains at least 47 lines (188 sectors) in flight per SM at 1,000,000 (lower bound from Little, 285 cycle L2 latency). The real number of misses in flight is not determined. Needs a latency bound scene (a chain of distinct lines per lane with little work) to get it from the other side.
- The one warp incoherent latency is 2.7 to 3.3 times the F67 line: 32 distinct rays in a warp take about three times as long as one ray, so the lanes of a warp are not serviced in parallel by the unit.
