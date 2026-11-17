# RTNN and Arkade on the F207 datasets: replay counts against recorder counts, and warp time split (2026-11-17, triton GPU 1)

## RTNN counts (replay = F207 CPU count x 1.10 with the K = 50 stop rule; recorder = intersection program calls per query)

| dataset | r | replay mean | recorder mean | recorder / replay | replay warp max sum / lanes | recorder warp max sum / lanes | recorder max lane | per query median ratio |
|---|---|---|---|---|---|---|---|---|
| bunny | 0.00030682 | 1.106 | 1.055 | 0.95 | 1.22 | 1.56 | 6 | 0.91 |
| bunny | 0.0012273 | 14.648 | 14.817 | 1.01 | 19.29 | 19.50 | 50 | 1.02 |
| dragon | 0.15333 | 11.418 | 11.438 | 1.00 | 14.35 | 14.71 | 76 | 1.00 |
| dragon | 0.6133 | 83.135 | 83.033 | 1.00 | 92.21 | 105.44 | 316 | 0.98 |

## RTNN time: launch event time (no probes, no recorder, 3 runs) against the F207 prediction, and what is inside "time search compute"

| dataset | r | F207 predicted | launch event ms (rt_rec build) | event / predicted | "search compute" ms (rt_rec build, same runs) | not in the launch ms (host: allocate and fill the output buffer) | output buffer MB | host us per MB | "search compute" ms (F207 build) | launch estimate for the F207 build = that minus host | estimate / predicted | prediction with recorder counts | event / that |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bunny | 0.00030682 | 0.123 | 0.0737 | 0.60 | 0.333 | 0.260 | 72 | 3.6 | 0.311 | 0.052 | 0.42 | 0.123 | 0.60 |
| bunny | 0.0012273 | 0.191 | 0.2120 | 1.11 | 0.459 | 0.247 | 72 | 3.4 | 0.425 | 0.179 | 0.94 | 0.191 | 1.11 |
| dragon | 0.15333 | 1.737 | 1.5073 | 0.87 | 4.409 | 2.902 | 722 | 4.0 | 4.317 | 1.415 | 0.81 | 1.738 | 0.87 |
| dragon | 0.6133 | 5.297 | 7.5774 | 1.43 | 10.481 | 2.904 | 722 | 4.0 | 8.973 | 6.069 | 1.15 | 5.305 | 1.43 |

The rt_rec build carries the (disabled) probe code, so its launch is longer than the F207 build's (dragon r2: 10.48 against 8.97 ms of search compute); the estimate column removes the same host part from the F207 build's search compute and is the better guess of that build's launch.

## RTNN warp time split (probe runs, mean of 2; the probes lengthen the launch, see the probe launch ms)

| dataset | r | probe launch ms | probe cost | callback share of warp cycles | traversal window share (trace - callback) | outside trace share | warps in flight per SM | slowest warp ms | slowest warp / launch | measured cycles per query per SM | model c0 + cq q | callback part: measured / model (4.9 q) | traversal part: measured / model (110) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bunny | 0.00030682 | 0.0973 | 1.32 | 0.15 | 0.77 | 0.09 | 11.3 | 0.0289 | 0.30 | 72 | 120 | 1.05 | 0.56 |
| bunny | 0.0012273 | 0.2601 | 1.23 | 0.28 | 0.71 | 0.01 | 12.8 | 0.1098 | 0.42 | 208 | 188 | 0.74 | 1.36 |
| dragon | 0.15333 | 1.9958 | 1.32 | 0.25 | 0.74 | 0.02 | 14.6 | 0.1588 | 0.08 | 148 | 171 | 0.60 | 1.01 |
| dragon | 0.6133 | 9.3634 | 1.24 | 0.34 | 0.66 | 0.00 | 14.6 | 0.4617 | 0.05 | 745 | 522 | 0.61 | 4.49 |

## Arkade counts (replay = F207 CPU count x 1.10; recorder = intersection program calls per query / 2)

| dataset | replay mean | recorder mean | recorder / replay | replay warp max sum / lanes | recorder warp max sum / lanes | per query median ratio |
|---|---|---|---|---|---|---|
| gowalla | 13658.1 | 12486.5 | 0.91 | 48877.12 | 44524.42 | 0.91 |
| glove3d | 44172.6 | 113154.2 | 2.56 | 128636.67 | 236572.45 | 2.40 |
| manuscript | 162.6 | 157.9 | 0.97 | 167.01 | 171.28 | 0.96 |

## Arkade time split (probe runs, 2 runs each; launch = CUDA event with probes)

| dataset | variant | launch ms | callback share of warp cycles | traversal window share | warps | SMs used (of 142) | mean warp life ms | slowest warp life ms | slowest / launch | model B ms (F207) | slowest warp / model B | in the slowest warp: warp calls / its max lane count | life cycles per call / model cycles per call | its max lane count / model B count | product of the three | callback cycles per call (all warps) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gowalla | fixed | 185.261 | 0.80 | 0.20 | 313 | 78 | 156.145 | 185.230 | 1.00 | 39.606 | 4.68 | 2.10 | 2.45 | 0.91 | 4.68 | 3914 |
| gowalla | orig | 773.123 | 0.95 | 0.05 | 313 | 78 | 674.621 | 773.052 | 1.00 | 440.064 | 1.76 | 2.06 | 0.94 | 0.91 | 1.76 | 20299 |
| glove3d | fixed | 964.983 | 0.80 | 0.20 | 313 | 78 | 842.242 | 964.929 | 1.00 | 110.982 | 8.69 | 2.01 | 2.61 | 1.66 | 8.69 | 3980 |
| glove3d | orig | 4017.107 | 0.95 | 0.05 | 313 | 78 | 3662.273 | 4017.021 | 1.00 | 1233.133 | 3.26 | 2.03 | 0.96 | 1.67 | 3.26 | 20562 |
| manuscript | fixed | 0.980 | 0.73 | 0.27 | 313 | 78 | 0.835 | 0.958 | 0.98 | 0.151 | 6.37 | 2.34 | 3.15 | 0.86 | 6.37 | 4016 |
| manuscript | orig | 5.502 | 0.82 | 0.14 | 313 | 78 | 4.306 | 5.453 | 0.99 | 1.672 | 3.26 | 2.43 | 1.56 | 0.86 | 3.26 | 22607 |
