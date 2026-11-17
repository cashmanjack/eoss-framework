# Arkade reproduction against the paper's Table 2 (2026-11-17, triton RTX 6000 Ada, GPU 1)

Program: the repo's s02-withTrueknn, unmodified, KN 50, built for NORM 0 (L infinity) and NORM 1 (L1); it doubles the radius and relaunches all 10,000 queries each round; the printed total search seconds is compared. Start radius from the repo's initial_estimate.py rule. Inputs and the pass rule (within a factor 3) were committed before any run (08d0149, model/priorworks/repro_arkade_prep.py). Raw: notes/q35/raw (private). Two runs each, medians. The paper used an RTX 4070 Ti (a smaller GPU than ours).

| dataset | norm | paper s | ours s (2 runs) | ours / paper | reproduces (rule) |
|---|---|---|---|---|---|
| Gowalla (unit sphere, repo script) | L infinity | 0.022 | 0.155 to 0.165 | 7.3 | no |
| Gowalla | L1 | 0.032 | 0.421 | 13 | no |
| Glove 3D (F207 file) | L infinity | 0.0025 | 0.033 | 13 | no |
| Glove 3D | L1 | 0.0063 | 0.056 | 8.9 | no |
| Manuscript (F207 file) | L infinity | 0.010 | 0.0087 | 0.87 | yes |
| Manuscript | L1 | 0.011 | 0.0185 | 1.7 | yes |

Reading: Manuscript reproduces; Gowalla and Glove 3D are 7 to 13 times slower than the paper on a faster GPU, so the GPU does not explain it. Not isolated: the datasets (the paper names no preprocessing for Glove 3D; our Gowalla has 1,280,969 distinct location ids against 1,270,969 in the paper) or the start radius (the repo script takes a random sample, ours is seeded). F207's Arkade rows used L2 (NORM 2), which is not one of the paper's norms, and raw lat/lon for Gowalla, so they cannot be compared with the paper at all.
