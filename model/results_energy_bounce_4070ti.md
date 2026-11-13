# Energy per bounce ray, RTX 4070 Ti (io, GPU 0), 2026-11-13
OptiX closest hit, bounce rays (origin = camera hit, random direction) on the 8 million and 16 million triangle soups (11.2 and about 11 million rays per launch), code ttu/bench/Morton/energy_bounce.py, host mode egy / egysort in ttu/bench/Morton/morton.cu (OptiX only, no TTU code, NVML total energy counter).
Method: per configuration one process: idle window 10 s, loop window 10 s of back to back launches (synchronize every 20, about 60 ms), idle window 10 s; 3 repetitions, rotated order; clocks sampled in the loop. Every result equals the pixel order result (0 wrong in all 24 windows). "Net" = window energy minus the mean of the two idle powers times the window time (idle in the process, 32 to 35 W: context alive, not the 7 W of an unused card).
Configs: pixel (natural order), morton (origin Morton, 5 bits per axis, sorted on the CPU, trace only), oracle (full hit subtree key, a bound, trace only), net (pixel order file through the GPU sort: key generation, radix sort, gather, trace, scatter, all inside the window).

| scene | config | Mrays/s | W | nJ per ray | nJ per ray net of idle | rays per J (millions) | SM MHz | time vs pixel | energy per ray vs pixel (total / net) |
|---|---|---|---|---|---|---|---|---|---|
| 8M | pixel | 2315 | 279.8 | 120.9 | 106.4 | 8.27 | 2826 | 1 | 1 / 1 |
| 8M | morton trace | 2694 | 269.2 | 99.9 | 87.5 | 10.0 | 2834 | 1.16 | 1.21 / 1.22 |
| 8M | oracle trace | 2994 | 279.4 | 93.3 | 82.2 | 10.7 | 2846 | 1.29 | 1.30 / 1.29 |
| 8M | morton net | 1089 | 203.6 | 186.9 | 156.4 | 5.35 | 2865 | 0.47 | 0.65 / 0.68 |
| 16M | pixel | 2216 | 282.2 | 127.4 | 112.0 | 7.85 | 2827 | 1 | 1 / 1 |
| 16M | morton trace | 2574 | 271.0 | 105.3 | 91.8 | 9.50 | 2853 | 1.16 | 1.21 / 1.22 |
| 16M | oracle trace | 2839 | 280.5 | 98.9 | 86.9 | 10.1 | 2829 | 1.28 | 1.29 / 1.29 |
| 16M | morton net | 1043 | 205.1 | 196.7 | 163.9 | 5.08 | 2865 | 0.47 | 0.65 / 0.68 |

Spread over the 3 repetitions: under 1.1% in nJ per ray and 0.3% in Mrays/s for every cell, so the differences above are resolved. Raw: notes/io/energy2 (not in git).
Reading: energy per ray follows time per ray. Trace only, Morton gains 16% in time and 21% in energy (the card draws 4% less power, 269 against 280 W, because fewer cache misses mean less DRAM traffic); the oracle gains 28 to 29% in time and 29 to 30% in energy (same power as pixel order). Net of the sort costs both lose, and energy loses less than time (0.65 against 0.47) because the sort, gather and scatter kernels draw about 204 W against 280 W for traversal. The sorted launch still uses 1.5 times the energy per ray, so sorting bounce rays does not pay in energy either.
Caveats: whole card energy (SMs, L2, DRAM, fans excluded by the card's own counter definition), one chip, soups only, ray sets as in Finding 107, SM clock differs by up to 1.4% between configs (net runs at 2865, pixel at 2826), the card runs close to its power limit (280 W of 285 W), idle subtraction uses in process idle (32 to 35 W).
