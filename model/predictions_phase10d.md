# Predictions, 2026-11-18 (tethys, RTX 4090 GPU 1): page test for F238, clock matched F229, sampler fix

Committed before any run of these items. Only the Conc_1 checked lean TTU group (stage 0 first, tiny launches first, timeout 30 s). No transform sequence. Raw files timestamped, never overwritten. No smoke number is read as a result.

## 1. Page test (F238): translation or request count?

New plain CUDA stream types (ld.global.cg, 16 bytes per thread, 32 loads per iteration, 384 threads per block, same iteration counting as before). In types 1 to 4 every lane of a warp reads its own 128 byte line (32 lines per warp load, as the strided stream of F229), the line choice is the same hash in all of them; only where the lines live changes:
- pgfew: 32,768 lines (4 MB) contiguous: 2 large pages (64 small), L2 resident after warm up.
- pgmany: the same 32,768 lines, 32 KB apart over 1 GB: 512 large pages (16,384 small), still 4 MB of distinct lines, L2 resident.
- pgfar: the same lines 512 KB apart over 16 GB: 8,192 large pages, still L2 resident.
- dramfew: 2,097,152 lines (256 MB) contiguous, random lines: DRAM bound, 128 large pages (against F229 stream: 1 GB, 512 pages).
- seqpg: coalesced, each warp load reads 512 contiguous bytes (4 lines) and the next load jumps to a new 2 MB page (1 GB over 512 pages): fewer requests per load, many pages.
Arms: sib 4 blocks per SM and hi 2 per SM (matched 128 blocks), random rays (soup 100,000) and coherent rays (tri1), RT G 32. Each of pgfew and pgmany also with a sleep of 150 us per iteration (rate matched through the sleep). The reference is the F229 strided stream (memcg, 1 GB, DRAM). Rates reported per arm: iterations per second, lines per second (32 per iteration for the scattered types, 4 per load for seqpg) and page touches per second (new 2 MB page per load for pgmany, pgfar, seqpg, mem).

Competing explanations: (T) translation: the collapse needs many pages per warp load (the TLB of the TPC); (R) request count: it needs many lines per warp load whatever the pages.
- If T: pgfew leaves random RT on sib above 0.5, pgmany collapses it (under 0.1), pgfar at least as low as pgmany, dramfew above 0.3.
- If R: pgfew and pgmany both collapse (under 0.1) and seqpg stays above 0.5.
My prediction (before running): R. pgfew random RT on sib under 0.1 (0.02 to 0.08), pgmany within a factor 1.5 of pgfew, pgfar within 1.5 of pgmany, dramfew 0.02 to 0.08, seqpg on sib above 0.4. My reasoning: F229 showed the collapse is steep in the stream rate (a 25 percent cut in rate gave 10 times more RT), which fits a shared queue of requests better than a TLB with a fixed hit rate; and the 120 GB/s DRAM stream was only 12 percent of peak, a request queue limit and not DRAM. I give T 30 percent.
- Stream rates: pgfew and pgmany (L2 resident) at least 1.5 times the DRAM stream's lines per second; seqpg lines per second at least 4 times the strided rate.
- hi placement: same order as sib but 3 to 7 times higher (as 0.146 against 0.022 in F229): pgfew hi random 0.08 to 0.3.
- Coherent rays: sib above hi for pgfew and pgmany (flip stays); seqpg coherent above 0.6 on both.
- With the 150 us sleep the pgfew and pgmany rates fall to within 20 percent of each other and the RT ratios rise (0.1 to 0.5 on sib).
- 0 wrong RT records in every run.
Decision rule (set now): translation drives it if pgmany random RT on sib is at most half of pgfew's (pgfew at least 2 times higher) at equal lines per load; request count drives it if pgfew and pgmany are within a factor 1.5 and both under 0.2 (also at the matched sleep); otherwise mixed, said as mixed.

## 2. Clock matched F229 (memory stream arms)

Every arm and every alone run with the NVML sampler (about 1 ms between samples), alone baseline run between arms (alone before each arm, not only at the start of the repetition), 3 repetitions of the arms of F229 (mem and memcg, sib 4 and hi 2 blocks, unthrottled and sleep 150 us and 800 us, and memseq 1 and 4 blocks) plus the arms of item 1. Clock matched ratio = RT of the arm over the median RT of those alone launches whose mean SM clock is within 1 percent of the arm's mean clock (if no alone launch matches, the alone rate is scaled by the clock ratio and labeled so).
- Memory stream arms run at 2,715 to 2,730 MHz as the ALU arms did (the persistent kernel holds the top bin) while alone runs at 2,520 to 2,685: so the F229 ratios against the unmatched alone were too high; at the strongest collapse (0.022) the change is below 0.003; at 0.15 to 0.5 it is 7 to 9 percent (clock matched ratio lower by that fraction).
- Alone runs spend part of their launches at 2,685 to 2,730 MHz (the coherent alone range was 2,520 to 2,685): if so matched baselines are 3 to 8 percent higher than the F229 alone median.
- Qualitative F229 conclusions stand (sib below hi for random, flip for coherent, throttle dependence).

## 3. Sampler fix

The sampler keeps a ring of the last 300 s of samples in memory and writes only rows inside launch windows (plus 10 ms margin) when the runner sends the window (CLOCK_MONOTONIC t0, t1 of every launch from the raw json). Expected file size per group under 20 MB (about 1,000 rows per second of launch). Any raw file over 100 MB gets gzipped after a lossless check (zcat compared with the original), including the old 3 GB file (expected compression 5 to 10 times, since the rows repeat the same clock values; the original is removed only after the check, which is Jack's explicit instruction).
