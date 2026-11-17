# Prior works: critical path bound scored (2026-11-17, triton GPU 1)

Ratio = measured / predicted, closes when 0.5 to 2. "F207" = throughput form alone. Predictions committed in d181c1f before this script was run.

## RTNN

| dataset | r | measured | F207 | ratio | critical path | ratio |
|---|---|---|---|---|---|---|
| bunny | 0.00030682 | 0.311 | 0.123 | 2.53 | 0.123 | 2.53 |
| bunny | 0.0012273 | 0.425 | 0.191 | 2.23 | 0.191 | 2.23 |
| dragon | 0.15333 | 4.317 | 1.737 | 2.49 | 1.737 | 2.49 |
| dragon | 0.6133 | 8.973 | 5.297 | 1.69 | 5.297 | 1.69 |

## TrueKNN (fixed build, launches = sum of per round GT spans)

| dataset | k | measured | F207 | ratio | critical path | ratio | rounds where chain binds |
|---|---|---|---|---|---|---|---|
| road | 5 | 17.59 | 5.63 | 3.12 | 26.34 | 0.67 | 6 |
| road | 659 | 87136.32 | 538.20 | 161.90 | 560.75 | 155.39 | 1 |
| porto1m | 5 | 1068.72 | 11.78 | 90.72 | 2340.49 | 0.46 | 15 |
| uni1m | 5 | 9.33 | 16.60 | 0.56 | 17.00 | 0.55 | 1 |

## Arkade (event time of the launch; rows equal F207 by construction)

| dataset | variant | measured | prediction | ratio |
|---|---|---|---|---|
| gowalla | fixed | 160.161 | 39.606 | 4.04 |
| gowalla | orig | 762.846 | 441.384 | 1.73 |
| glove3d | fixed | 853.487 | 110.982 | 7.69 |
| glove3d | orig | 3974.557 | 1234.453 | 3.22 |
| manuscript | fixed | 0.909 | 0.240 | 3.79 |
| manuscript | orig | 5.235 | 3.984 | 1.31 |

Median absolute log error (F207 / critical path): rtnn 0.86 / 0.86, trueknn 2.82 / 0.69, arkade 1.25 / 1.25.
