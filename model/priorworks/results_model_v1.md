# Model v1 scored on the 14 real rows (2026-11-17, triton, no run, no refit)

Form: model/priorworks/FORM_combined_v1.md (committed eb22f98 before this script). Ratio = measured / predicted, closes when 0.5 to 2. NOT blind (see the form file). Units ms.

| program | row | measured | callback form | chain | v1 = max | ratio v1 | ratio callback alone |
|---|---|---|---|---|---|---|---|
| RTNN | bunny_r1 | 0.3112 | 0.1642 | 0.000989 | 0.1642 | 1.89 | 1.89 |
| RTNN | bunny_r2 | 0.4252 | 0.2154 | 0.01187 | 0.2154 | 1.97 | 1.97 |
| RTNN | dragon_r1 | 4.317 | 2.131 | 0.01718 | 2.131 | 2.03 | 2.03 |
| RTNN | dragon_r2 | 8.973 | 3.705 | 0.05745 | 3.705 | 2.42 | 2.42 |
| TrueKNN | road k5 (13 probe rounds, 13 chain rounds) | 17.59 | 3.651 | 22.52 | 24.63 | 0.71 | 4.82 |
| TrueKNN | road k659 (13 probe rounds, 13 chain rounds) | 8.714e+04 | 1.481e+05 | 80.1 | 1.481e+05 | 0.59 | 0.59 |
| TrueKNN | porto1m k5 (20 probe rounds, 20 chain rounds) | 1069 | 8.361 | 2332 | 2336 | 0.46 | 127.82 |
| TrueKNN | uni1m k5 (5 probe rounds, 5 chain rounds) | 9.335 | 6.873 | 0.58 | 7.254 | 1.29 | 1.36 |
| Arkade | gowalla fixed | 182.4 | 64.93 | 39.61 | 64.93 | 2.81 | 2.81 |
| Arkade | gowalla orig | 768.9 | 221.2 | 440.1 | 440.1 | 1.75 | 3.48 |
| Arkade | glove3d fixed | 977.2 | 353.4 | 111 | 353.4 | 2.77 | 2.77 |
| Arkade | glove3d orig | 3998 | 1199 | 1233 | 1233 | 3.24 | 3.33 |
| Arkade | manuscript fixed | 1.049 | 0.308 | 0.1505 | 0.308 | 3.41 | 3.41 |
| Arkade | manuscript orig | 5.346 | 1.138 | 1.672 | 1.672 | 3.20 | 4.70 |

- RTNN: median abs log error v1 0.69 (callback alone 0.69), inside factor 2: 2 of 4, worst ratio 2.4
- TrueKNN: median abs log error v1 0.43 (callback alone 1.05), inside factor 2: 3 of 4, worst ratio 2.2
- Arkade: median abs log error v1 1.10 (callback alone 1.21), inside factor 2: 1 of 6, worst ratio 3.4
- ALL: median abs log error v1 0.74 (callback alone 1.03), inside factor 2: 6 of 14, worst ratio 3.4
