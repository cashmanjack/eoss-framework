# Results Phase 10d, 2026-11-17/18 (tethys, RTX 4090 GPU 1; predictions model/predictions_phase10d.md, committed first e7dc6f6, addendum 4cd525c)

Code: setup3.cu (compute types pgfew, pgmany, pgfar, pgodd, dramfew, seqpg, all with ld.global.cg), ttu/bench/Conc_3 (conc3_run.py, clksamp), ttu/bench/Conc_2/clksamp.c (window sampler), model/conc3_analyze.py. Only the Conc_1 checked lean TTU group was run (stage 0 first, tiny launches first, timeout 30 s per run, GPU idle before every run, 0 wrong and 0 unstored records in every run of all three groups). Raw: notes/conc3/raw (private).
Disclosures: (1) the smoke run (one arm, sib pgfew, both scenes) was read before the matrix: sib pgfew random RT 0.205 (214.9 against 1,048.6 rays/us). (2) The first two groups (pg, mem) ran before the control arm pgodd was thought of; the addendum 1b with the seen numbers was committed before the third group. (3) The pgodd group is small: its alone runs had no launch at 2,730 MHz, so its "matched" column equals the pooled column (not clock matched); only pg and mem groups are clock matched.

## Design
Same line choice hash in all page test streams: every lane of a warp reads its own 128 byte line (32 lines per warp load). Pages per warp load were counted from the access pattern (model/conc3_analyze.py pattern): lines, 64 KB pages, 2 MB pages per warp load.
|stream|distinct lines|where|lines, 64K pages, 2M pages per warp load|served from|
|---|---|---|---|---|
|memcg (F229)|about 16.8 M (1 GB)|strided 4 MB|32, 32, 2.9|DRAM|
|dramfew|2.1 M (256 MB)|contiguous|32, 32, 16|DRAM|
|pgfew|32,768 (4 MB)|contiguous|32, 30.6, 2|L2|
|pgmany|32,768|32 KB apart over 1 GB|32, 32, 32|L2|
|pgodd|32,768|241 lines (30.8 KB) apart over 1.01 GB|32, 32, 32|L2|
|pgfar|32,768|512 KB apart over 16 GB|32, 32, 32|L2|
|seqpg|1 GB, 4 lines per warp load|a new 2 MB page every load|4, 1, 1|DRAM and L2|

## 1. Page test (3 repetitions each, 5 launches per run, clock matched ratios; pooled (F229 way) = matched / 0.925 within rounding)
RT ratio against alone at matched clock, our TTU kernel, G 32 blocks, stream blocks 128 (sib 4 per SM, hi 2 per SM). First number sib (TPC siblings of the RT SMs), second hi (other TPCs).
|stream|random rays sib / hi|coherent rays sib / hi|stream useful GB/s (lines per second, sib random)|
|---|---|---|---|
|memcg 1 GB DRAM (F229)|0.020 / 0.134|0.418 / 0.261|118 (7.4 G)|
|dramfew 256 MB DRAM|0.023 / 0.141|0.440 / 0.273|130 (8.1 G)|
|pgfew (2 pages per load, L2)|0.190 / 0.649|0.765 / 0.691|969 (60.6 G)|
|pgmany (32 pages per load, L2)|0.076 / 0.671|0.778 / 0.700|422 (26.4 G)|
|pgodd (control, odd stride)|0.085 / 0.732 (pooled, see disclosure 3)|0.852 / 0.774 (pooled)|418 (26.1 G)|
|pgfar (8,192 pages)|0.076 / 0.672|0.770 / 0.700|419 (26.2 G)|
|seqpg (coalesced, new page per load)|0.172 / 0.337|0.430 / 0.431|1,507 (11.8 G)|
|pgfew, sleep 150 us|0.922 / 0.928|0.909 / 0.913|78 (4.9 G)|
|pgmany, sleep 150 us|0.795 / 0.917|0.896 / 0.887|83 (5.2 G)|
|pgodd, sleep 150 us (pooled)|0.865 / 1.003|0.976 / 0.957|80 (5.0 G)|
|memcg, sleep 150 us (F229 level)|0.191 / 0.257|0.472 / 0.330|89 (5.6 G)|
Answer to the question:
- Request count does NOT drive the sibling slowdown. pgfew issues 60.6 G lines per second (8 times the DRAM stream) and leaves random RT at 0.19; at about 5 G lines per second (sleep 150 us) the DRAM stream costs 0.19 and the L2 streams 0.80 to 0.92.
- Translation (page diversity per warp load) matters, but only a little and only on the sibling SMs: pgmany is 2.5 times lower than pgfew on siblings (0.076 against 0.190) with the same lines per load, 1.0 times on other TPCs (0.671 against 0.649); at the low rate 0.795 against 0.922. It is not TLB capacity: 512 pages (pgmany), 504 (pgodd) and 8,192 (pgfar) give the same ratio, so it is the number of different pages inside one warp load (or per TPC TLB reach of the nearest level), and not the stride (pgodd, odd stride, equals pgmany to within 0.01, so no power of two L2 slice conflict).
- The collapse to 0.02 is not explained by either: pgmany has more page diversity and 3.5 times the request rate of the DRAM stream and still costs only 0.076; the DRAM stream (2.9 pages per load, memcg, or 16, dramfew: same 0.020 to 0.023) costs 0.02. So the factor that separates 0.076 from 0.020 is that the stream's lines come from DRAM (long latency misses) and not from L2, at the same lines per warp load. A mechanism (leftover guess, not tested): DRAM latency holds shared per TPC miss resources far longer than L2 hits do.
- Other TPCs (hi) feel only the DRAM side (0.134 to 0.141 for DRAM streams, 0.65 to 0.73 for L2 streams, no page effect).
- Coalesced seqpg (4 lines per load, 1 page per load) is in between (0.172 sib, 0.337 hi): fewer pages and requests per load than pgmany but a DRAM and L2 mixed stream at 1,507 GB/s useful.
- Coherent rays keep the flip for DRAM streams (sib 0.42 to 0.44 over hi 0.26 to 0.27) and for L2 streams sib is just above hi (0.77 against 0.69).
- Scoring of my predictions (I predicted request count, 30 percent for translation): pgfew sib random under 0.1: MISSED (0.190); pgmany within 1.5 of pgfew: MISSED (2.5 times); pgfar within 1.5 of pgmany: held (1.0); dramfew 0.02 to 0.08: held (0.023); seqpg sib above 0.4: MISSED (0.172); pgfew and pgmany lines per second at least 1.5 times the DRAM stream: held (8 and 3.5 times); seqpg lines per second at least 4 times: MISSED (1.6 times; its useful bytes are 13 times); pgfew hi 0.08 to 0.3: MISSED (0.649); hi 3 to 7 times sib: held for pgfew (3.4); coherent sib above hi for pgfew and pgmany: held (0.765 against 0.691, 0.778 against 0.700); seqpg coherent above 0.6: MISSED (0.43); with sleep the rates within 20 percent: held; RT ratios 0.1 to 0.5 with sleep: MISSED (0.80 to 0.92); 0 wrong: held. The pre set decision rule gave "translation" on its letter (pgmany 0.40 of pgfew on siblings) but the rule did not foresee the DRAM factor; the honest summary is the three bullets above.

## 2. Clock matched F229 (mem group, 3 repetitions, an alone run before every arm; matched = same SM clock within 1 percent)
The alone launches ran at 2,520 MHz in 77 to 97 percent of launches (pg group 81 and 77 percent, mem group 93 and 97 percent, random and coherent) and at 2,640 to 2,730 MHz in the rest (the first launches after a run that held the clock up); at 2,730 MHz alone gives 1,141 to 1,145 rays/us (random, +8.0 percent over 1,057) and 2,261 (coherent, +7.7 percent over 2,101). Every arm ran at 2,730 MHz. The matched baseline is therefore that 8 percent higher alone rate, and every F229 style ratio drops by 7.4 to 7.6 percent (matched over pooled 0.92 to 0.93 in every row), not the 3 to 8 percent range of the prediction's lower half only: the 8 percent is exactly the clock ratio 2,730 / 2,520 (1.083).
|arm (blocks)|random sib pooled to matched|random hi pooled to matched|coherent sib|coherent hi|
|---|---|---|---|---|
|mem 1 / 2 / 4 (hi 1 / 2 / 4)|0.022 to 0.021 (all three)|0.147 / 0.146 / 0.145 to 0.136 / 0.135 / 0.134|0.472 / 0.468 / 0.464 to 0.439 / 0.434 / 0.431|0.278 / 0.263 / 0.268 to 0.258 / 0.244 / 0.249|
|memcg 1 / 2 / 4|0.023 / 0.023 / 0.022 to 0.021 / 0.021 / 0.020|0.146 / 0.146 / 0.145 to 0.134 / 0.135 / 0.134|0.465 / 0.458 / 0.448 to 0.432 / 0.426 / 0.417|0.279 / 0.275 / 0.268 to 0.259 / 0.255 / 0.249|
|mem sleep 150 us|0.208 to 0.192|0.278 to 0.257|0.521 to 0.484|0.358 to 0.333|
|mem sleep 800 us|0.785 to 0.726|0.852 to 0.792|0.882 to 0.820|0.865 to 0.803|
|memseq sib 1, 4 (hi 1, 2)|0.755, 0.166 to 0.697, 0.154|0.505, 0.519 to 0.466, 0.480|0.925, 0.412 to 0.859, 0.383|0.634, 0.637 to 0.589, 0.592|
- Compute side and block count independence are unchanged; the coherent flip (sib above hi) is unchanged in size (matched sib over hi 1.73 to 1.78 for mem, 1.67 to 1.73 for memcg, against 1.65 to 1.76 pooled). The placement effect (6.5 times, random sib below hi) is unchanged (0.021 against 0.135).
- Matched baselines were unavailable for 1 to 5 of 15 launches in a few rows (sleep 800 us, coherent memcg hi); those launches were left against the pooled median (unscaled) and the row median is not moved by them (see 'scaled' in the analysis output).
- Scoring: the F229 ratios are about 7.5 percent too high where they are above 0.1, and 0.001 to 0.002 where 0.02: held; alone launches at higher clock exist and the matched baseline is 7.7 to 8.0 percent higher: held on the upper edge of my 3 to 8 percent; qualitative conclusions stand: held.

## 3. Sampler fix
The sampler keeps a ring of the last 400 s of samples (about 1 ms apart, 1.05 ms measured) and writes rows only inside launch windows (plus 10 ms) when the runner sends them: pg group 1.7 MB, mem group 2.9 MB, odd group about 0.3 MB (the first version wrote 3.07 GB). The runner gzips any raw file over 100 MB after a lossless check; the old file notes/conc2/raw/clk_clk_20261117T221634.csv (3,067,805,696 bytes) was gzipped (246,697,494 bytes, 12.4 times; zcat compared byte for byte with the original, then the plain copy removed on Jack's instruction), its _windows.csv stays. Prediction "5 to 10 times" missed (12.4 times); "under 20 MB per group": held.

## Mistakes
First smoke start set CUDA_VISIBLE_DEVICES=1 together with TTU_GPU=1 (the second GPU index no longer existed): the harness asserted, no GPU run happened. The venv python has no numpy on tethys: the runner needs /usr/bin/python3. A first line of the line rate formula in the analysis was wrong (12,288 loads per iteration used where the warp load count 384 is right) and was fixed before any rate was read. pgmany's power of two stride was a confound I did not control at first; pgodd settles it. I saw the smoke number before the matrix (disclosure 1).

## Open
Why DRAM sourced lines cost the siblings 4 times more than L2 sourced lines at the same lines per warp load (an arm with a footprint of 40 to 120 MB, so the L2 hit rate goes from 100 to 0 percent, would give a dose response); whether the page effect on siblings is the TPC's first level TLB (pages per warp load 1, 2, 4, 8, 16, 32 would give the knee); why seqpg (coalesced) costs the siblings 0.17; stage 1 of the transform sequence (needs Jack's approval).
