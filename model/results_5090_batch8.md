# RTX 5090 (pc14) session 8: tethys's L2rw benchmark on the 5090 (F426), scored against model/predictions_5090_batch8.md (commit a7009db, before the run), 2026-11-19

Run: ttu/bench/L2rw/l2rw.cu built with nvcc -O3 -arch=sm_120; ttu/bench/pc14/run_l2rw_pc14.py (tethys's run_l2rw.py with the idle test replaced by notes/pc14_gpucheck.sh, nothing else changed): patterns 0 coalesced 128 B, 1 lane4 (one sector per lane, 4 B accesses), 2 lane32 (one sector per lane, 32 B accesses), write quarters 0 to 4, footprints 8, 32, 80 MB, 3 repetitions (135 runs), ncu on repetition 0 (45 runs). One smoke config first. GPU 0, no fault, no failed run, no busy refusal. Plain CUDA only. Raw (private): notes/pc14_core/l2rw. Scorer: model/score_5090_batch8.py. TB/s is requested bytes as the program prints them (sectors x 32 B).

```
runs 135 notes/pc14_core/l2rw/l2rw_20261119T162051.jsonl
| footprint MB | pattern | TB/s at write quarters 0 / 1 / 2 / 3 / 4 | ns per sector (x1000) 0 / 4 | cost ratio c_w / c_r | wq 2 against the linear mean of the ends | rep spread (max/min TBps, wq 0 and 4) |
|---|---|---|---|---|---|---|
| 8 | coalesced 128 B | 5.54 / 5.50 / 5.46 / 3.62 / 3.49 | 5.77 / 9.18 | 1.59 | 0.78 | 1.002, 1.001 |
| 8 | lane4 (4 B writes) | 6.42 / 8.97 / 5.96 / 4.35 / 3.42 | 4.98 / 9.35 | 1.88 | 0.75 | 1.001, 1.000 |
| 8 | lane32 (32 B) | 6.39 / 8.86 / 6.79 / 4.41 / 3.35 | 5.00 / 9.55 | 1.91 | 0.65 | 1.000, 1.001 |
| 32 | coalesced 128 B | 5.56 / 5.51 / 5.46 / 3.63 / 3.49 | 5.76 / 9.17 | 1.59 | 0.79 | 1.005, 1.001 |
| 32 | lane4 (4 B writes) | 6.43 / 9.06 / 5.96 / 4.35 / 3.42 | 4.98 / 9.35 | 1.88 | 0.75 | 1.001, 1.000 |
| 32 | lane32 (32 B) | 6.40 / 8.87 / 6.79 / 4.41 / 3.35 | 5.00 / 9.55 | 1.91 | 0.65 | 1.000, 1.000 |
| 80 | coalesced 128 B | 5.56 / 5.45 / 5.32 / 3.89 / 3.45 | 5.76 / 9.27 | 1.61 | 0.80 | 1.006, 1.001 |
| 80 | lane4 (4 B writes) | 6.43 / 9.04 / 5.96 / 4.35 / 3.42 | 4.98 / 9.35 | 1.88 | 0.75 | 1.001, 1.000 |
| 80 | lane32 (32 B) | 6.40 / 8.86 / 6.79 / 4.42 / 3.35 | 5.00 / 9.55 | 1.91 | 0.65 | 1.000, 1.000 |

ncu (repetition 0): hit rate pct, lts pct of peak, DRAM MB, read / write sectors M
8 coalesced 128 B wq0 hit 100.00 lts 75.90 dram 0.4 rd 174 wr 0 | wq4 hit 100.00 lts 97.25 dram 0.4 rd 0 wr 174
8 lane4 (4 B writes) wq0 hit 100.00 lts 89.70 dram 0.5 rd 1393 wr 0 | wq4 hit 100.00 lts 95.42 dram 0.5 rd 0 wr 1392
8 lane32 (32 B) wq0 hit 100.00 lts 89.16 dram 0.4 rd 2785 wr 0 | wq4 hit 100.00 lts 93.43 dram 1.6 rd 0 wr 2784
32 coalesced 128 B wq0 hit 100.00 lts 75.88 dram 0.7 rd 174 wr 0 | wq4 hit 99.99 lts 97.55 dram 2.4 rd 0 wr 174
32 lane4 (4 B writes) wq0 hit 100.00 lts 89.75 dram 0.8 rd 1393 wr 0 | wq4 hit 100.00 lts 95.46 dram 3.0 rd 0 wr 1393
32 lane32 (32 B) wq0 hit 100.00 lts 89.29 dram 0.7 rd 2785 wr 0 | wq4 hit 100.00 lts 93.45 dram 2.8 rd 0 wr 2785
80 coalesced 128 B wq0 hit 99.11 lts 75.40 dram 49.0 rd 174 wr 0 | wq4 hit 99.30 lts 96.58 dram 2541.7 rd 0 wr 174
80 lane4 (4 B writes) wq0 hit 99.89 lts 89.71 dram 49.1 rd 1393 wr 0 | wq4 hit 99.86 lts 95.39 dram 17039.6 rd 0 wr 1393
80 lane32 (32 B) wq0 hit 99.95 lts 89.19 dram 49.1 rd 2785 wr 0 | wq4 hit 99.95 lts 93.46 dram 24626.9 rd 0 wr 2785
lane32 costs (32 MB): c_w/c_r 1.91; predicted speedup per round 3.89 4.25 4.70 3.43 total 3.71; measured 2.99 3.34 4.00 2.98 total 3.17; error per round +30% +27% +18% +15%
lane4 costs (32 MB): c_w/c_r 1.88; predicted speedup per round 3.86 4.22 4.68 3.41 total 3.69; measured 2.99 3.34 4.00 2.98 total 3.17; error per round +29% +26% +17% +14%
coalesced costs (32 MB): c_w/c_r 1.59; predicted speedup per round 3.60 3.93 4.41 3.22 total 3.48; measured 2.99 3.34 4.00 2.98 total 3.17; error per round +20% +18% +10% +8%
unweighted sector ratio per round 2.91 3.17 3.65 2.72 errors -3% -5% -9% -9%
```
## Scoring
1. L2 hit rate at least 98 percent at 8 and 32 MB: HELD (99.99 to 100.00); at 80 MB at least 90 percent: HELD (99.1 to 99.95).
2. Read only: coalesced 4.5 to 6.8 TB/s HELD (5.56), lane32 4.5 to 7.5 HELD (6.40); both above the 4090 (3.9, 5.05) HELD (lane4 6.43).
3. Write cost ratio: coalesced 1.5 to 2.5 HELD (1.59, the 4090: 1.5); lane32 1.5 to 3.0 HELD (1.91); at least 2.5 (the 4090's) NO (35 percent predicted); below 1.5 NO.
4. Partial sector writes (lane4) within 15 percent of full sector writes (lane32): HELD (1.88 against 1.91, 2 percent; the 4090 2.51 against 2.57).
5. Footprints: 8 and 32 MB within 10 percent: HELD (within 1.5 percent in every cell); 80 MB within 15 percent of 32 MB: HELD (largest 7 percent, coalesced wq 3, 3.89 against 3.63).
6. The 50 percent write mix faster per sector than the linear mean of the two ends by 10 to 30 percent: HELD for coalesced (20 to 22 percent: ratio 0.78 to 0.80), MISSED for lane32 (35 percent, ratio 0.65; lane4 25 percent, in range). Also: with one write in four accesses the per lane patterns run 38 to 40 percent FASTER than read only (8.9 against 6.4 TB/s), the coalesced pattern is unchanged (5.50 against 5.56): reads alone do not fill the L2.
7. ncu L2 throughput 90 to 100 percent for read only and write only alike: write only HELD (93.4 to 97.6), read only MISSED (75.9 coalesced, 89.2 and 89.7 per lane). The read only coalesced stream, at 76 percent by the metric and 5.56 TB/s, is not at the L2's limit; the write only streams are at 93 to 98 percent.
Not predicted, seen: at 80 MB the write only streams write back to DRAM (ncu DRAM bytes 2.5 GB coalesced, 17 and 25 GB per lane, against under 3 MB at 8 and 32 MB) at up to about 0.9 TB/s without any change of the L2 write rate (3.35 against 3.35 TB/s): dirty lines leave the 96 MB L2 while the footprint is below it, and it costs nothing visible at this rate. Reads at 80 MB: 49 MB of DRAM, 99.1 to 99.95 percent hit.

## Against the 4090 (F423) and the kNN speedup method (F425)
Write over read cost per sector: lane32 1.91 here against 2.5 on the 4090, coalesced 1.59 against 1.5, partial equals full on both; the 5090's isolated 512 byte 16 B stream (F421) gave 2.0. So the chips differ in the per lane pattern (writes cost less on the 5090) and not in the coalesced one.
The F425 method (time = read sectors x read cost + write sectors x write cost, ncu sector counts of F420) with the 5090's own costs predicts the speedup per round: lane32 costs 3.89, 4.25, 4.70, 3.43 (total 3.71), lane4 3.86, 4.22, 4.68, 3.41 (3.69), coalesced 3.60, 3.93, 4.41, 3.22 (3.48), against measured 2.99, 3.34, 4.00, 2.98 (total 3.17): 15 to 30 percent high (lane32), 8 to 20 percent high (coalesced); the unweighted sector ratio 2.91, 3.17, 3.65, 2.72 is 3 to 9 percent low. So with costs measured on the chip itself the method still does not predict the 5090 speedup; it works on the 4090 (5 to 11 percent) only.
