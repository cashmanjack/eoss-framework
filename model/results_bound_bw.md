# Bandwidth bound: results against the committed predictions (2026-11-17, tethys RTX 4090 GPU 1; predictions model/predictions_bound_bw.md, commit f9e48b7, before any bandwidth run or score; script model/bound_bw.py; bandwidth program model/bound_bw_bandwidth.cu; raw bandwidth output notes/bound_bw/bwb_bw_20261117T164739.txt (private); plain CUDA, no TTU code, no fault, GPU idle before and after)

Bandwidth (268 MB buffers, L2 flushed, median of 21 launches): copy 901.0 GB/s (the value used; prediction 850 to 930 HELD, 89 percent of the 1,008 GB/s peak), read only 786.7, write only 933.5. Prior works rows (triton, 960 GB/s peak): 858 GB/s assumed (960 x 901/1,008), not measured on that GPU.

Scores (rms log error of the predicted speedup over the natural order; model time = measured natural time / F123 speedup; "binds" = bound above the model time):
| set | n | F123 rms | bound alone rms | max(F123, bound) rms | cells where bound binds | bound / measured time: natural, min, max over orders |
|---|---|---|---|---|---|---|
| F205 sponza | 6 | 0.083 | 0.062 | 0.083 | 0 | 0.60, 0.62, 0.65 |
| F205 conference | 6 | 0.014 | 0.064 | 0.014 | 0 | 0.66, 0.69, 0.71 |
| F205 rungholt | 6 | 0.132 | 0.153 | 0.153 | 6 | 1.98, 1.40, 2.08 |
| F208 conference_camera | 7 | 0.086 | 0.091 | 0.086 | 0 | 0.89, 0.70, 0.88 |
| F208 conference_bounce | 7 | 0.055 | 0.061 | 0.055 | 0 | 0.66, 0.64, 0.71 |
| F208 sponza_camera | 7 | 0.051 | 0.065 | 0.051 | 0 | 0.79, 0.67, 0.80 |
| F208 sponza_bounce | 7 | 0.031 | 0.021 | 0.031 | 0 | 0.68, 0.65, 0.69 |
| F208 rungholt_camera | 7 | 0.053 | 0.141 | 0.068 | 1 | 0.82, 0.81, 1.19 |
| F208 rungholt_bounce | 7 | 0.075 | 0.242 | 0.137 | 3 | 1.13, 0.83, 1.45 |

pooled F205 (18 cells): F123 0.091  bound alone 0.102  max form 0.101  bound binds in 6 of 18
pooled F208 (42 cells): F123 0.061  bound alone 0.126  max form 0.079  bound binds in 4 of 42

Diagnostic (labeled, in sample: the bytes are the F213 ncu bytes, known after the fact, not part of the pass rules): Rungholt random, ncu DRAM bytes (read + write) in place of the replay bytes
| order | replay bytes MB | ncu bytes MB | replay bound ms | ncu bound ms | measured ms | ncu bound / measured |
|---|---|---|---|---|---|---|
| natural | 2657 | 1255 | 2.949 | 1.393 | 1.489 | 0.94 |
| oracle d1 | 2047 | 670 | 2.272 | 0.744 | 1.157 | 0.64 |
| oracle d2 | 1834 | 669 | 2.036 | 0.743 | 1.006 | 0.74 |
| oracle d3 | 1749 | 670 | 1.941 | 0.744 | 0.935 | 0.80 |
| oracle d4 | 1495 | 670 | 1.659 | 0.744 | 0.894 | 0.83 |
| oracle full | 1113 | 670 | 1.235 | 0.744 | 0.885 | 0.84 |
| morton-o k5 | 1753 | 779 | 1.945 | 0.865 | 1.090 | 0.79 |
rms log speedup error, 6 orders (Morton included): F123 0.132  bound alone 0.214  max form 0.132  (bound binds in 0 of 6)

Prior works rows (device buffer / fixed versions; triton RTX 6000 Ada; bandwidth 858 GB/s = 960 GB/s peak x the copy fraction measured on the 4090; bytes are a LOWER bound: no node traffic)
| row | bytes MB | bound ms | model ms | measured ms | bound / model | bound / measured | binds (bound > model) | max(model, bound) / measured |
|---|---|---|---|---|---|---|---|---|
| rtnn_bunny_r1 | 76.8 | 0.0895 | 0.123 | 0.311 | 0.729 | 0.2878 | no | 0.395 |
| rtnn_bunny_r2 | 76.8 | 0.0895 | 0.191 | 0.425 | 0.470 | 0.2106 | no | 0.448 |
| rtnn_dragon_r1 | 765.2 | 0.8918 | 1.737 | 4.317 | 0.513 | 0.2066 | no | 0.402 |
| rtnn_dragon_r2 | 765.2 | 0.8918 | 5.297 | 8.973 | 0.168 | 0.0994 | no | 0.590 |
| trueknn_road_k5 | 140.7 | 0.1640 | 5.633 | 17.590 | 0.029 | 0.0093 | no | 0.320 |
| trueknn_road_k659 | 24800.1 | 28.9013 | 538.196 | 87136.320 | 0.054 | 0.0003 | no | 0.006 |
| trueknn_porto1m_k5 | 168.8 | 0.1967 | 11.781 | 1068.720 | 0.017 | 0.0002 | no | 0.011 |
| trueknn_uni1m_k5 | 197.2 | 0.2299 | 16.602 | 9.330 | 0.014 | 0.0246 | no | 1.779 |
| arkade_gowalla | 4.1 | 0.0048 | 39.606 | 160.161 | 0.000 | 0.0000 | no | 0.247 |
| arkade_glove3d | 4.1 | 0.0048 | 110.982 | 853.487 | 0.000 | 0.0000 | no | 0.130 |
| arkade_manuscript | 4.1 | 0.0048 | 0.240 | 0.909 | 0.020 | 0.0053 | no | 0.264 |
rows 11, bound binds in 0; median |log error| model 1.139, max form 1.139; rms 2.303 / 2.303

Scoring of the written expectations: P1 HELD (bound binds in 0 of 11 prior works rows; bound over measured at most 0.29; nearest is RTNN bunny r1 at 0.73 of the model time, as I wrote). R1 HELD (replay bytes: bound 1.40 to 2.08 times the measured time in all six Rungholt orders, natural 1.98, so it is not a bound); consequence HELD (max form 0.153 against F123 0.132, bound alone 0.153). R2 HELD (max form does not beat F123 on Rungholt). S1 half: no D = 0 set binds (HELD) but the bound over measured range 0.55 to 0.8 MISSED on Conference camera (0.70 to 0.89, natural 0.89). S2: the "less than 0.005" parts MISSED (pooled F205 0.091 to 0.101, pooled F208 0.061 to 0.079), the direction (worse, through the Rungholt cells) HELD. Pass line for "useful model term" FAILED (max form is worse than F123 on the Rungholt orders and on both pooled sets, and the bound is above the measured time in every Rungholt cell).

Reading (marked as such):
1. With the replay bytes the bound is not a bound. The only cells where it binds (10 of 60 scene cells: 6 Rungholt random, 1 Rungholt camera, 3 Rungholt bounce) are the ones where the replay feature over estimates DRAM traffic (F213: measured read 0.27 to 0.70 of the feature), and the bound there is 1.2 to 2.1 times the measured time. In those cells it makes the prediction worse (Rungholt random 0.132 to 0.153, Rungholt bounce 0.075 to 0.137).
2. Nowhere does a VALID bound bind. Prior works: the lower bound bytes (queries and results only) are 0.01 to 0.29 of the measured time and 0.00 to 0.73 of the F207 prediction, so the misses of F207 are not a bandwidth effect (TrueKNN k 659 moves at least 24.8 GB and the bound is 29 ms against 87 s measured). Scenes that fit in L2 (D = 0): the bound is the ray traffic alone, 0.60 ms at 901 GB/s, equal to predict 1.0 for the speedup, and it is 0.60 to 0.89 of the measured time, so it is a real FLOOR that is nearly reached on camera ray sets (Conference camera 0.89 of the natural time of 0.672 ms) but it never exceeds the F123 time.
3. Diagnostic (in sample, labeled, not in the pass line): with the ncu DRAM bytes of F213 the bound is valid (0.64 to 0.94 of the measured time, natural order 0.94: the natural Rungholt launch runs at 94 percent of the copy bandwidth) and binds in 0 of 6 orders, so max form equals F123 (0.132); the bound alone is 0.214. So the idea is sound for the natural order but a floor that is 6 to 36 percent below the sorted times cannot explain the F205 depth trend either (consistent with F213). What is missing is a byte estimate: a bound needs the DRAM bytes of the actual launch, and the replay's L2 model is not it.
Caveats: one bandwidth run on one GPU (the copy rate is a streaming rate, mixed read and write traffic with random node reads may reach less, which would raise the bound); prior works bandwidth assumed, byte formulas for TrueKNN (8 B per neighbour) not checked against source; no node traffic for prior works (lower bound only); scenes: the F205 and F208 times were known before the predictions.
