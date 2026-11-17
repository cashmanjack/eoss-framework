# Bandwidth bound, predictions BEFORE any bandwidth run or scoring (2026-11-17, tethys GPU 1; CPU scoring on existing files; one plain CUDA bandwidth run; no TTU code)

Idea: time of a launch >= bytes per launch / DRAM bandwidth. Bytes per launch = node traffic (the replay DRAM feature D of F209, bytes per ray, times rays) + ray reads (32 B per ray) + result writes (32 B per ray); F213 measured 268 MB and 268 to 270 MB for 8,388,608 rays. Script model/bound_bw.py (forms and byte formulas in its header, written before any bandwidth number or score), bandwidth program model/bound_bw_bandwidth.cu, bytes per cell: `model/bound_bw.py bytes` (output below, no bandwidth needed).
Disclosure: the measured times of all sets and the F213 ncu bytes (natural 985 MB read) were known to me before this file; the byte formulas use only the replay feature and buffer sizes, not ncu bytes. The expectations below are therefore not blind for Rungholt. The Sponza, Conference and prior works cells were not looked at against a bound before.

## Bandwidth
DRAM bandwidth = the copy rate of model/bound_bw_bandwidth.cu on the 4090 (268 MB in, 268 MB out, L2 flushed, median of 21 launches), the traffic shape of a ray launch. Prediction: copy 850 to 930 GB/s (peak 1,008; F213's natural order moved 1,255 MB in 1.488 ms = 843 GB/s at 85.8 percent of peak). If the copy rate is below 800 or above 960 GB/s the prediction is marked missed; the measured value is used anyway. Prior works rows ran on triton (RTX 6000 Ada, GDDR6, 960 GB/s peak, no bandwidth run possible from here): the bandwidth used there is 960 GB/s x (4090 copy rate / 1,008), an assumption that the copy efficiency transfers (labeled; the answer is not sensitive to it, see expectation P1).

## Forms (fixed now)
Model time of order i = measured natural time / F123 speedup of order i (the F123 form (F123, F205, F208, no refit) predicts a speedup only; the natural time is the anchor, so natural itself is not scored). Bound b_i = bytes_i / BW. Three scored predictions of the speedup of order i over natural:
 F123: f_i.  Bound alone: b_nat / b_i.  Max form: max(t_nat, b_nat) / max(t_nat / f_i, b_i).
Error = rms log (predicted speedup / measured speedup), the F205 / F208 scoring. Sets: Rungholt orders (F205 random rays, 6 orders, the F213 launches); F205 pooled (Sponza, Conference, Rungholt, 18 cells); F208 pooled (six camera and bounce sets, 42 cells). A cell is "binding" when b_i > t_nat / f_i (the bound is above the model time, so the max form changes the prediction). Validity of the bound itself: bound / measured time of the same order must be at most 1; above 1 the "bound" is not a bound (its bytes are too high).
Prior works rows (model/priorworks/, read only): device buffer / fixed versions, model time = the F207 prediction (pred.json), measured = results_priorworks.md. The bytes there are a LOWER bound (ray or query reads + result writes, NO node traffic: no replay for these builds): RTNN N x (12 + 50 x 4) B, TrueKNN sum over rounds of active x (12 + 8k) B (8 B per neighbour is ASSUMED, source not on this machine), Arkade 10,000 x (12 + 50 x 8) B. Scored: where the bound is above the F207 prediction (binding), and the median absolute log error of the model against max(model, bound).

## Written expectations (probabilities are my guesses)
 P1 (prior works): the bound binds in 0 of 11 rows (probability 85 percent; the nearest is RTNN bunny r1: 76.8 MB is about 0.09 ms against 0.123 predicted); the bound over measured stays below 0.5 in every row. A binding row would be one of the RTNN rows.
 R1 (Rungholt, bound validity): with the replay bytes the bound ABOVE the measured time for natural and oracle d1 (bound 2.9 and 2.2 ms against 1.49 and 1.16 ms at about 900 GB/s), because F213 showed the replay feature over estimates DRAM bytes by 1.4 to 3.7 times (probability 80 percent). Consequence expected: the max form is WORSE than F123 on the Rungholt orders (probability 70 percent), the bound alone is worse than F123 (probability 85 percent).
 R2 (Rungholt, if it helps): I do not expect the max form to pass any better line than F123's 0.132 on the Rungholt orders; "helps" is defined as max form rms below F123 rms minus 0.01 (probability 15 percent).
 S1 (Sponza, Conference, camera, bounce: no node traffic, D = 0, every order has the same bytes of 537 MB): the bound alone equals predict 1.0 on those sets; its bound over measured is 0.55 to 0.8 for every order of every set (the ray traffic alone is 0.57 ms at 940 GB/s against times of 0.67 to 1.0 ms) (probability 70 percent). The bound binds in at most 3 of the 18 + 42 cells of the F205 and F208 scene sets that have D = 0 (probability 75 percent): a sorted order on a scene that already fits in L2 cannot be predicted faster than the ray traffic, and F123 does not predict above 1.08 there.
 S2 (pooled, sign): the max form changes the pooled F205 rms by less than 0.005 (probability 55 percent) and the pooled F208 rms by less than 0.005 (probability 80 percent). If it changes either by more, the direction is worse (probability 65 percent), through the Rungholt cells.
 Pass line for "the bandwidth bound is a useful model term": max form pooled rms below F123 on Rungholt orders by 0.01 AND not above F123 on F205 pooled and F208 pooled AND bound at most the measured time in every Rungholt cell (all three). Probability 10 percent. If it fails the verdict is "not a usable bound with the replay bytes", and the remaining diagnostic (labeled, in sample, not in the pass line) with the ncu bytes of F213 shows whether the bytes or the idea is wrong.

## Bytes per cell (replay node traffic included for scenes), from `model/bound_bw.py bytes` before any bandwidth run
| set | order | bytes MB (ray 64 B per ray + replay node traffic) |
|---|---|---|
| F205 sponza | natural | 537 |
| F205 sponza | morton-o k5 | 537 |
| F205 sponza | oracle d1 | 537 |
| F205 sponza | oracle d2 | 537 |
| F205 sponza | oracle d3 | 537 |
| F205 sponza | oracle d4 | 537 |
| F205 sponza | oracle full | 537 |
| F205 conference | natural | 537 |
| F205 conference | morton-o k5 | 537 |
| F205 conference | oracle d1 | 537 |
| F205 conference | oracle d2 | 537 |
| F205 conference | oracle d3 | 537 |
| F205 conference | oracle d4 | 537 |
| F205 conference | oracle full | 537 |
| F205 rungholt | natural | 2657 |
| F205 rungholt | morton-o k5 | 1753 |
| F205 rungholt | oracle d1 | 2047 |
| F205 rungholt | oracle d2 | 1834 |
| F205 rungholt | oracle d3 | 1749 |
| F205 rungholt | oracle d4 | 1495 |
| F205 rungholt | oracle full | 1113 |
| F208 conference_camera | natural | 537 |
| F208 conference_camera | shuffled | 537 |
| F208 conference_camera | morton | 537 |
| F208 conference_camera | oracle d1 | 537 |
| F208 conference_camera | oracle d2 | 537 |
| F208 conference_camera | oracle d3 | 537 |
| F208 conference_camera | oracle d4 | 537 |
| F208 conference_camera | oracle full | 537 |
| F208 conference_bounce | natural | 537 |
| F208 conference_bounce | shuffled | 537 |
| F208 conference_bounce | morton | 537 |
| F208 conference_bounce | oracle d1 | 537 |
| F208 conference_bounce | oracle d2 | 537 |
| F208 conference_bounce | oracle d3 | 537 |
| F208 conference_bounce | oracle d4 | 537 |
| F208 conference_bounce | oracle full | 537 |
| F208 sponza_camera | natural | 537 |
| F208 sponza_camera | shuffled | 537 |
| F208 sponza_camera | morton | 537 |
| F208 sponza_camera | oracle d1 | 537 |
| F208 sponza_camera | oracle d2 | 537 |
| F208 sponza_camera | oracle d3 | 537 |
| F208 sponza_camera | oracle d4 | 537 |
| F208 sponza_camera | oracle full | 537 |
| F208 sponza_bounce | natural | 537 |
| F208 sponza_bounce | shuffled | 537 |
| F208 sponza_bounce | morton | 537 |
| F208 sponza_bounce | oracle d1 | 537 |
| F208 sponza_bounce | oracle d2 | 537 |
| F208 sponza_bounce | oracle d3 | 537 |
| F208 sponza_bounce | oracle d4 | 537 |
| F208 sponza_bounce | oracle full | 537 |
| F208 rungholt_camera | natural | 538 |
| F208 rungholt_camera | shuffled | 926 |
| F208 rungholt_camera | morton | 593 |
| F208 rungholt_camera | oracle d1 | 537 |
| F208 rungholt_camera | oracle d2 | 537 |
| F208 rungholt_camera | oracle d3 | 539 |
| F208 rungholt_camera | oracle d4 | 546 |
| F208 rungholt_camera | oracle full | 544 |
| F208 rungholt_bounce | natural | 1083 |
| F208 rungholt_bounce | shuffled | 1719 |
| F208 rungholt_bounce | morton | 1337 |
| F208 rungholt_bounce | oracle d1 | 1111 |
| F208 rungholt_bounce | oracle d2 | 888 |
| F208 rungholt_bounce | oracle d3 | 693 |
| F208 rungholt_bounce | oracle d4 | 661 |
| F208 rungholt_bounce | oracle full | 640 |

| prior works row | bytes MB (lower bound, no node traffic) |
|---|---|
| rtnn_bunny_r1 | 76.8 |
| rtnn_bunny_r2 | 76.8 |
| rtnn_dragon_r1 | 765.2 |
| rtnn_dragon_r2 | 765.2 |
| trueknn_road_k5 | 140.7 |
| trueknn_road_k659 | 24800.1 |
| trueknn_porto1m_k5 | 168.8 |
| trueknn_uni1m_k5 | 197.2 |
| arkade_gowalla | 4.1 |
| arkade_glove3d | 4.1 |
| arkade_manuscript | 4.1 |
