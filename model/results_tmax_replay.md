# tmax replay check against F182 (2026-11-16 night, tethys, against model/predictions_tmax_replay.md, committed before the replay was run)

CPU replay (model/replay/replay.cpp, ORDER 0) on the soup of 100,000 triangles and the 262,144 rays of F182. Inputs: the Flags_1 host got a dump option (one OptiX build of the same GAS as F182, gopaque = 1, GAS bytes, triangles and rays written, no launch; the GAS bytes stay in notes/tmaxreplay, private). make_table.py box matching (100,000 of 100,000 assigned, 7,064 nodes, 70,390 leaf slots; fresh tree, F167). Script model/tmax_replay.py. No GPU timing run.

```
tmax      nodes  leaves  tris   n+l   | ratio(n+l) meas ratio  err%% | ratio(n) | hit  meas  | time pred  meas  err%%
0           6.3    0.6    1.1    6.9 | 0.381 0.256   48.8 | 0.490 | 0.000 0.00 | 0.242 0.188   28.6
0.01        6.6    0.8    1.5    7.4 | 0.408 0.283   43.9 | 0.512 | 0.037 0.04 | 0.253 0.188   34.7
0.02        6.9    1.0    1.8    7.9 | 0.434 0.312   39.1 | 0.533 | 0.072 0.07 | 0.264 0.219   20.7
0.05        7.6    1.5    2.8    9.2 | 0.506 0.390   29.7 | 0.592 | 0.170 0.17 | 0.295 0.246   20.1
0.1         8.7    2.3    4.1   11.0 | 0.607 0.517   17.4 | 0.675 | 0.306 0.31 | 0.339 0.289   17.3
0.2        10.3    3.4    6.0   13.7 | 0.753 0.703    7.1 | 0.796 | 0.499 0.51 | 0.402 0.371    8.3
0.5        12.3    4.8    8.5   17.1 | 0.941 0.931    1.1 | 0.951 | 0.746 0.75 | 0.483 0.469    2.9
1          12.9    5.2    9.2   18.0 | 0.995 0.993    0.3 | 0.996 | 0.819 0.82 | 0.506 0.496    2.0
2          12.9    5.2    9.2   18.1 | 1.000 0.999    0.1 | 1.000 | 0.825 0.83 | 0.508 0.507    0.2
1e+16      12.9    5.2    9.2   18.1 | 1.000 1.000    0.0 | 1.000 | 0.825 0.83 | 0.508 0.508    0.0
```
(columns: replay nodes, leaf slots entered, triangles tested per ray; ratio = (n + l) at tmax over its unlimited value, against the measured unit L1 requests ratio of F182; time pred = the line through the measured endpoints (0.188 and 0.508 ns) applied to the replay requests ratio scaled to 80.8.)

## Against the predictions
1. Hit fraction within 0.02 of the measured value: HELD at all ten tmax (replay 0 / 0.037 / 0.072 / 0.170 / 0.306 / 0.499 / 0.746 / 0.819 / 0.825 / 0.825 against the measured 262,144 ray values 0 / 0.038 / 0.075 / 0.175 / 0.314 / 0.509 / 0.754 / 0.823 / 0.829 / 0.829, largest difference 0.010; the replay is slightly low). CORRECTION 2026-11-17: the first version of this file compared against the F182 text series 0.04 / 0.12 / 0.19 / 0.38 ..., which was taken from the 32 ray probe launch, and wrongly reported 'hardware hit fraction above exact geometry at short tmax' (7 of 10 held, a cause hunt for 'extra hits'). A full brute force over all 262,144 rays (double precision) and four new OptiX runs show the hardware equal to exact geometry on every ray (see model/results_hwhits.md). The replay is low because its box matching assignment misses the nearest hit in about 3.5 percent of rays (7 of 200 checked by brute force: it returns a farther hit).
2. Request shape within 10 percent at every tmax: MISSED. The replay's (n + l) ratio is too high at short tmax: +49 percent at tmax 0, +44 / +39 / +30 / +17 at 0.01 / 0.02 / 0.05 / 0.1; within 10 percent only from 0.2 up (+7, +1, +0.3, +0.1, 0). The tmax 0 case is not alone (the rule said failure only at 0 to 0.02 would be an origin box effect: it extends to 0.1).
3. Absolute: n + l per ray unlimited 18.1 and n 12.9, predicted 65 to 100 and 45 to 80: MISSED by 3.6 and 3.5 times. The measured requests are sectors, not node fetches: 80.8 / 18.1 = 4.5 requests per (n + l); at tmax 0 it is 3.0.
4. Time: with the unweighted replay requests the line gives errors of +29 / +35 / +21 / +20 / +17 percent at tmax 0 to 0.1, +8 at 0.2, then within 3 percent: MISSED below 0.2 (rule: 10 percent at every tmax).
Verdict by the reading rule written before: the replay is NOT accepted as a requests model for tmax below about 0.2 (about three triangle edges) with n + l unweighted; it is accepted from 0.2 up .

## After seeing the data (not a prediction, fitted on the same 10 points)
Requests = a n + b l (replay nodes and leaf slots entered per ray) with a = 2.24 and b = 9.92 fits all ten measured requests within 3.1 percent (n, l, tris: 2.5 percent, three parameters; n and tris: 3.9). So the replay's counts carry the shape of F182 once a node counts about 2 sectors and a leaf slot about 10 (the leaf includes its triangle data); the raw sum n + l weights them equally and is wrong at short tmax, where the node count dominates. Two parameters on ten points from the same series: a candidate form to test on another scene (F44, F47, F70, RTNN radii), not a validation.
## For F185
tmax IS already an input of the replay and the replay moves in the right direction and by the right size once weighted; the unweighted (n + l) replay overstates the work of short rays by 17 to 49 percent for tmax up to 0.1, so any use of the replay for radius queries must use per type weights. 
