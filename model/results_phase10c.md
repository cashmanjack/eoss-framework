# Results Phase 10c, 2026-11-17/18 (tethys, RTX 4090 GPU 1; predictions model/predictions_phase10c.md, committed first 333b503, addendum 317f5f9)

Code: setup3.cu (compute types memcg, memseq, throttle as the fourth field of --compute, launch stamps t0 and t1 in every launch record), ttu/bench/Conc_2 (conc2_run.py, clksamp.c), model/conc2_analyze.py, Omm_1 (any hit loads: ahl=), Div_1 (run_cheap.sh, lanehist.h modes, rtnn overlay). Only the Conc_1 checked lean TTU group was run (stage 0 first, tiny launches first, timeout 30 s each, 0 wrong and 0 unstored records in every run). Raw: notes/conc2, notes/omm1c, notes/div1c (private).
Disclosures: the first cg repetition (notes/conc2/smoke) was started by mistake as the full group instead of smoke; it is counted as one of the three repetitions of cg (3 in all). Two smoke runs of lvlsmoke (notes/conc2/smoke2) were read for plumbing only and showed memseq above the strided stream. The raw NVML sampler file (3 GB, 95 million rows at about 30 us) is kept as is (notes/conc2/raw/clk_*.csv); the rows inside launch windows are in the _windows.csv next to it.

## 1. F229 mechanism (3 repetitions, spread under 2 percent except where shown)
RT ratio against RT alone in the same group. Streaming kernel = 16 bytes per thread at a 4 MB stride (one line and one page per lane), useful bandwidth counted as iterations x 196,608 bytes.

|arm (blocks)|random rays|coherent rays|compute GB/s with RT (alone)|
|---|---|---|---|
|sib mem (cached) 1, 2, 4 per SM|0.023, 0.023, 0.022|0.482, 0.470, 0.465|122, 120, 118 (128)|
|sib memcg (L1 bypass) 1, 2, 4|0.023, 0.022, 0.023|0.464, 0.453, 0.450|121, 120, 118 (127)|
|hi mem 1, 2, 4|0.150, 0.146, 0.145|0.273, 0.276, 0.277|110, 110, 109 (128)|
|hi memcg 1, 2, 4|0.145, 0.146, 0.145|0.278, 0.270, 0.263|110, 110, 109 (127)|
|sib mem 4, sleep 150 us|0.211|0.524|88.5 (96)|
|sib mem 4, sleep 800 us|0.775|0.890|23.4 (24)|
|hi mem 2, sleep 150 us|0.284|0.358|87.7 (96)|
|hi mem 2, sleep 800 us|0.857|0.866|23.9 (24)|
|(memcg arms equal mem arms within 0.01)| | | |
|memseq sib 1 / 4|0.769 / 0.160|0.927 / 0.424|1,494 / 1,289 (2,737 / 4,990)|
|memseq hi 1 / 2|0.501 / 0.521|0.629 / 0.637|2,625 / 5,054 (4,617 / 8,283)|

- L1 eviction does NOT explain the 0.022: with L1 bypass loads the RT ratio is the same (0.022 to 0.023 on sib, 0.145 to 0.150 on hi, coherent 0.45 to 0.48 and 0.26 to 0.30), and the compute rate is the same within 1 percent.
- The block count (1, 2 or 4 per SM) does not change the compute rate or the RT ratio: one block per SM already saturates this stream at about 120 GB/s (12 percent of DRAM peak). So the first "bandwidth levels" idea (block count) gave one level. The throttle gives three levels (125, 96, 24 GB/s alone): RT rises 0.022, 0.211, 0.775 (sib) and 0.146, 0.284, 0.857 (hi); the dependence is steep and not proportional to bytes: a 25 percent cut in compute rate raises sib RT 10 times.
- The sibling effect is real at matched blocks (hi 2 per SM and sib 4 per SM are both 128 blocks: 0.146 against 0.022, 6.6 times; hi 1 and sib 2, 64 blocks: 0.150 against 0.023) and disappears at the lowest rate (0.775 to 0.808 sib against 0.857 hi).
- It is not the bytes: the coalesced stream (memseq) moves 10 to 40 times more useful bytes per second and costs RT random rays much less (sib 1 block 0.77, 4 blocks 0.16, against 0.022). Caveat: the memseq blocks start at offsets 37 iterations apart so they trail each other in L2 and part of their bytes are L2 hits (rates above DRAM peak); so memseq is a high request, mostly L2 resident stream, not a DRAM stream. What the strided stream has that the coalesced one lacks is one line and one page per lane per access; a TLB or request count mechanism is the leading candidate; not tested (no arm with the strided pattern confined to few pages).
- Coherent flip: sib above hi (0.47 against 0.28) at every unthrottled arm, matched blocks or not (sib over hi 1.65 to 1.76), shrinking to 0.52 against 0.36 at 96 GB/s and 0.89 against 0.87 at 24 GB/s. The compute side loses more beside sibling coherent RT (0.56 of alone) than beside other TPCs (0.75). With coherent rays the RT working set is a few lines, so its loss cannot be capacity; it is the shared memory path. Bound: the flip does not depend on block count or L1 bypass; it exists only at high compute rates; origin of the direction (sib better than hi for coherent, worse for random) not found. A guess (not tested): coherent RT beside other TPCs is slowed by chip wide L2 hot line contention with 256 stream blocks, the sibling arm has half the stream blocks.
- Scoring of my predictions: held: memcg within a factor 2 of cached; compute bandwidth of memcg within 15 percent; placement survives matching (3 times or more); coherent memcg close to cached; sib throttled random 0.08 to 0.4 at the 96 GB/s level (0.211) and 0.4 to 0.9 at 24 (0.775 to 0.808); hi 24 GB/s 0.8 to 1.0 (0.857); sib below hi by 1.3 times at 96 GB/s (1.35); memseq compute at least 2 times the strided rate and RT 5 times 0.022 (0.16 at 4 blocks); coherent at 24 GB/s sib above 0.8 (0.885) and hi above 0.6 (0.866); compute side lower on sib. Missed: sib random rising at 1 block per SM (flat 0.023); coherent flip 1.0 to 1.5 at matched blocks (1.65 to 1.76); hi at 96 GB/s 0.5 to 0.9 (0.284); sib below hi at 24 GB/s by 1.3 (1.1); my "1/4 and 1/16" levels were really 0.76 and 0.19 of the alone rate.

## 2. ALU speedup: clocks (NVML, about 30 us per sample, 5 repetitions, launches 1 to 5 of each run)
|arm|SM clock median (min to max)|memory clock|power W|RT ratio|
|---|---|---|---|---|
|alone, random rays|2,520 (2,520 to 2,520)|10,251|65|1|
|hi alu, random|2,715 (2,715 to 2,730)|10,251|98|1.077|
|sib alu, random|2,730 (2,520 to 2,730)|10,251|99|1.077|
|alone, coherent|2,520 (2,520 to 2,685)|10,251|70|1|
|hi alu, coherent|2,730 (2,520 to 2,730)|10,251|125|1.072|
|sib alu, coherent|2,730 (2,730 to 2,730)|10,251|95|1.072|
- The SM clock is 7.7 to 8.3 percent higher while the ALU kernel runs (2,520 against 2,715 to 2,730 MHz) and the RT gain is 7.2 to 7.7 percent: the clock explains the speedup quantitatively (the sizes agree within about 1 percent). Memory clock identical. Cause of the higher clock not studied (the persistent kernel keeps the GPU at its top boost bin while the alone run sits one or two bins lower; I cannot lock clocks, not admin). Consequence: every F229 ratio against an alone run was taken against a run 7 to 8 percent slower in clock; mem arms were not clock sampled, so their ratios may be about 8 percent too high if they also boosted. The qualitative results stand.
- Scoring: 7 percent present: held; clock differs by under 2 percent: MISSED (it differs by 8 percent); memory clock same: held; power at least 30 W higher: held (33 and 55 W, 25 W for sib coherent: missed there).

## 3. Micromaps with dependent loads in the any hit (8 processes, all variants interleaved, 4 million rays, 512 MB table)
Time over A (any hit on every candidate), medians; AHL = dependent loads per any hit call; results identical to A for every B variant (0 differences in all 32 files).
|AHL, N|D|B2|B4|B6|B8|C2|C4|C6|C8|
|---|---|---|---|---|---|---|---|---|---|
|4, 100,000|0.45|0.76|0.72|0.71|0.75|0.54|0.63|0.67|0.75|
|4, 1,000,000|0.41|0.72|0.67|0.67|0.70|0.49|0.60|0.64|0.70|
|16, 100,000|0.13|0.57|0.21|0.20|0.21|0.15|0.18|0.19|0.21|
|16, 1,000,000|0.12|0.61|0.21|0.20|0.20|0.14|0.18|0.19|0.21|
Any hit invocations per ray (same as F228): A 1.50 (1.83 at 1M), B2 0.82 (1.00), B4 0.22 (0.26), B6 0.054 (0.066), B8 0.014 (0.017).
- Yes: the saving now shows in time. With 16 dependent loads B4 and finer take 0.20 to 0.21 of A (a 5 times gain, D is 0.12 to 0.13), with 4 loads only 0.67 to 0.72 (1.4 times). The time stops falling after level 4: B6 and B8 are not faster than B4 even though invocations fall 4 and 16 times further, and C6 is 1.4 to 1.6 times D: a floor of the unit side micromap lookup (about 0.2 of A at 16 loads, 0.2 to 0.25 of A at 4 loads on top of D's 0.41 to 0.45) that rises again at level 8 (0.75 at AHL 4).
- Scoring: identical results held; A over D 3 to 8 at AHL 4: missed (2.2 to 2.4); above 10 at AHL 16: just missed (7.7 to 8.3); B2 0.6 to 0.9 at AHL 4: held (0.72 to 0.76); B4 0.2 to 0.45 and B6 0.08 to 0.25 at AHL 4: missed (0.67 to 0.72); at AHL 16 held (0.21, 0.20); B8 0.05 to 0.2: just missed (0.20 to 0.21); C6 within 1.0 to 1.6 of D: held; B4 under 0.5 of A at both AHL: held at 16 only. The any hit with global loads is not a real texture (random permutation, no locality).

## 4. Cheaper probe on RTNN (kernel time in ms, 3 repeats, same build, probe off = plain; r 0.05 / r 0.025, K 50, 100,000 uniform points)
|mode|r 0.05 ms (over plain)|r 0.025 ms (over plain)|mean lanes per callback|callbacks with 8 or fewer lanes|share of warp time at 8 or fewer lanes|
|---|---|---|---|---|---|
|plain (probe off)|0.600 to 0.643 (median 0.601)|0.141 to 0.151 (0.142)| | | |
|full (F231 probe)|0.750 (1.25)|0.175 to 0.184 (1.28)|13.589 / 12.022|0.463 / 0.496|0.343 / 0.324 to 0.327|
|sampled, 1 warp in 8|0.664 to 0.671 (1.11)|0.159 to 0.160 (1.12)|13.566 / 11.893|0.463 / 0.501|0.340 / 0.322 to 0.325|
|no timing, all warps|0.634 to 0.645 (1.06)|0.154 to 0.159 (1.11)|13.589 / 12.022 (identical)|0.463 / 0.496 (identical)|none|
- The probe cost falls from 25 to 28 percent to 11 to 12 percent (sampled) or 6 to 11 percent (no timing); not lower in these modes; what the remaining 6 to 12 percent is (the active mask read and branch, or the counter update) was not separated. The sampled statistics are within 1.1 percent (mean lanes) and 0.003 (time share) of the full probe, and the no timing lane histogram is identical to the full one (no timing information, as designed). The F231 RTNN time shares (0.33 to 0.35) stand within 0.003 on one eighth of the warps.
- Scoring: sampled under 1.06 (missed: 1.11 to 1.12); no timing 1.03 to 1.12 (held); sampled lane statistics within 3 percent (held), time share within 0.03 (held); no timing histogram identical (held). The F231 RTNN row is rerun: it stands, the overhead was real but did not change the statistics.
