# Prior works on their datasets: results against the committed predictions (2026-11-17, triton GPU 1)

Ratio = measured / predicted (1 = exact; success rule set before: within a factor 2 = ratio 0.5 to 2). Medians of the runs, ms.

## RTNN (one version: the output buffer is already device memory, the distance is already float)

| dataset | r | predicted | measured (runs) | ratio |
|---|---|---|---|---|
| bunny | 0.00030682 | 0.123 | 0.311 (3) | 2.53 |
| bunny | 0.0012273 | 0.191 | 0.425 (3) | 2.23 |
| dragon | 0.15333 | 1.737 | 4.317 (3) | 2.49 |
| dragon | 0.6133 | 5.297 | 8.973 (3) | 1.69 |

## TrueKNN (original = managed memory frame buffer, fixed = device buffer; launches only = sum of per round global timer spans; wall = sum of the program's per round times)

| dataset | k | predicted rounds | measured rounds | predicted launches | fixed launches | ratio | original launches | original / fixed | fixed wall | original wall |
|---|---|---|---|---|---|---|---|---|---|---|
| road | 5 | 13 | 13 | 5.63 | 17.59 (3) | 3.12 | 24305.68 (3) | 1381.7 | 85.8 | 24372.9 |
| road | 659 | 13 | 13 | 538.20 | 87136.32 (3) | 161.90 | nan (0) | nan | 91480.9 | nan |
| porto1m | 5 | 20 | 20 | 11.78 | 1068.72 (3) | 90.72 | 62388.63 (3) | 58.4 | 1282.5 | 62609.5 |
| uni1m | 5 | 5 | 5 | 16.60 | 9.33 (3) | 0.56 | 57948.37 (3) | 6207.8 | 65.1 | 58006.3 |

## Arkade (KN 50, NORM 2, 10,000 queries; fixed = device buffer + float square, original = pinned host buffer + double pow; CUDA event time of the launch (the GT lines are not printed with the probes off))

| dataset | r | predicted fixed | measured fixed GT | ratio | predicted original pinned | measured original GT | ratio | original / fixed (GT) | event time original / fixed |
|---|---|---|---|---|---|---|---|---|---|
| gowalla | 0.7881 | 39.606 | 160.161 (3) | 4.04 | 441.384 | 762.846 (3) | 1.73 | 4.8 | 4.8 |
| glove3d | 0.60924 | 110.982 | 853.487 (3) | 7.69 | 1234.453 | 3974.557 (3) | 3.22 | 4.7 | 4.7 |
| manuscript | 0.55414 | 0.240 | 0.909 (3) | 3.79 | 3.984 | 5.235 (3) | 1.31 | 5.8 | 5.8 |
