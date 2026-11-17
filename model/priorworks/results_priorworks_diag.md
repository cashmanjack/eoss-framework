# Prior works, variants (a) and (b) against the F207 measurements (2026-11-17, triton GPU 1, no refit)

Ratio = measured / predicted; a row closes when 0.5 to 2. "F207" = mean count form, "(a)" = per warp slowest lane, "(b)" = measured callback cost swapped in (forms committed before the runs).

## RTNN

Stub = same traversal, intersection program returns at once. cb = callback cycles per candidate per SM = (full - stub) x 142 x 2.5e9 / (1.10 x cube candidates in total); (b) cq = 4.9 + cb - cb_ref.

Reference (uniform 100,000, r 0.0232, 10.6 candidates per query): full 0.1784 ms, stub 0.1750 ms, cb_ref 1.01 cycles per candidate.

| dataset | r | measured ms | stub ms | cb | F207 pred | ratio | (a) pred | ratio | (b) pred | ratio | (b) with (a) counts | ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bunny | 0.00030682 | 0.311 | 0.304 | 20.28 | 0.123 | 2.53 | 0.123 | 2.52 | 0.164 | 1.89 | 0.167 | 1.86 |
| bunny | 0.0012273 | 0.425 | 0.393 | 2.57 | 0.191 | 2.23 | 0.214 | 1.99 | 0.215 | 1.97 | 0.246 | 1.73 |
| dragon | 0.15333 | 4.317 | 3.843 | 4.13 | 1.737 | 2.49 | 1.883 | 2.29 | 2.131 | 2.03 | 2.370 | 1.82 |
| dragon | 0.6133 | 8.973 | 10.644 (stub slower: invalid) | -0.86 | 5.297 | 1.69 | 5.763 | 1.56 | 3.705 | 2.42 | 3.993 | 2.25 |

Median abs log error: F207 0.86, a 0.76, b 0.69, ab 0.61. Callback share of the measured time (full - stub over full): bunny_r1 7%, bunny_r2 9%, dragon_r1 11%, dragon_r2 -18%.

## TrueKNN (fixed build, device buffer; the probes are always on; (b) uses the measured callback warp cycles of every round, F160 constants 1.485 and 12.8 in flight)

| dataset | k | measured launches ms | F207 pred | ratio | (a) pred | ratio | (b) pred | ratio | callback share of warp cycles |
|---|---|---|---|---|---|---|---|---|---|
| road | 5 | 17.59 | 5.63 | 3.12 | 12.69 | 1.39 | 3.64 | 4.83 | 0.35 |
| road | 659 | 87136.32 | 538.20 | 161.90 | 815.73 | 106.82 | 148113.83 | 0.59 | 1.00 |
| porto1m | 5 | 1068.72 | 11.78 | 90.72 | 34.25 | 31.20 | 8.36 | 127.81 | 0.29 |
| uni1m | 5 | 9.33 | 16.60 | 0.56 | 26.01 | 0.36 | 6.87 | 1.36 | 0.43 |

Median abs log error: F207 2.82, a 2.23, b 1.05.

## Arkade (probe builds; (b): T = T_ref x F_d / F_ref, F = callback warp cycles; ref = 100,000 uniform 2D points and queries, r 0.005, same variant)

| dataset | variant | measured ms (probe run) | measured ms (F207, no probes) | F207 pred | ratio | (a) pred | ratio | (b) pred | ratio | callback share of warp cycles |
|---|---|---|---|---|---|---|---|---|---|---|
| gowalla | fixed | 182.44 | 160.16 | 39.61 | 4.61 | 43.64 | 4.18 | 64.93 | 2.81 | 0.80 |
| gowalla | orig | 768.93 | 762.85 | 441.38 | 1.74 | 486.20 | 1.58 | 221.23 | 3.48 | 0.95 |
| glove3d | fixed | 977.22 | 853.49 | 110.98 | 8.81 | 114.88 | 8.51 | 353.36 | 2.77 | 0.80 |
| glove3d | orig | 3998.01 | 3974.56 | 1234.45 | 3.24 | 1277.75 | 3.13 | 1199.03 | 3.33 | 0.95 |
| manuscript | fixed | 1.05 | 0.91 | 0.24 | 4.37 | 0.15 | 6.97 | 0.31 | 3.41 | 0.73 |
| manuscript | orig | 5.35 | 5.23 | 3.98 | 1.34 | 2.99 | 1.79 | 1.14 | 4.70 | 0.82 |

Median abs log error (fixed rows): F207 1.53, a 1.94, b 1.03.
