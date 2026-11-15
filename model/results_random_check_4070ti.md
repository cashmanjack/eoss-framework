# Pre registered random ray check on the 4070 Ti: results (2026-11-15, io GPU 0, OptiX only)
Predictions: commit a93c71d (pushed before any ordered launch was timed). Measured with `random_check.py measure`: 8,388,608 random rays, 10 launches per order, median, every result identical to the natural order (0 wrong in all 21 launch sets).

| scene | order | predicted | measured | log error |
|---|---|---|---|---|
| soup 300,000 | Morton k5 / d1 / d2 / d3 / d4 / full | 1.40 / 1.13 / 1.27 / 1.35 / 1.39 / 1.39 | 2.18 / 1.36 / 1.98 / 2.14 / 2.18 / 2.18 | -0.44 / -0.19 / -0.45 / -0.46 / -0.45 / -0.45 |
| soup 1,500,000 | same | 1.51 / 1.09 / 1.24 / 1.40 / 1.46 / 1.51 | 2.35 / 1.23 / 1.68 / 2.11 / 2.26 / 2.30 | -0.44 / -0.12 / -0.31 / -0.41 / -0.44 / -0.43 |
| Bistro Exterior | same | 0.95 / 1.06 / 1.25 / 1.26 / 1.27 / 1.27 | 1.22 / 1.21 / 1.27 / 1.30 / 1.32 / 1.32 | -0.25 / -0.13 / -0.02 / -0.04 / -0.04 / -0.04 |

rms log error: 0.419 (300,000), 0.375 (1,500,000), 0.117 (Bistro), 0.331 over all 18. The model is wrong in one direction on every soup point (mean signed error -0.28: it predicts about 1.4 to 1.5 where 2.2 to 2.35 is measured). Bistro is predicted well from depth 2 on, but the Morton order is predicted to lose (0.95) and gains 1.22.

Why random rays were harder than bounce rays (0.36 on triton, 0.33 here, against 0.09 to 0.10 for bounce rays on this chip):
1. Dynamic range. The rms of the measured log speedups is 0.65 (triton random), 0.59 (these 18 points), 0.29 (bounce rays, F113): errors are 0.55, 0.56 and 0.33 of the spread to be explained. The relative error is similar; bounce rays only gain 1.1 to 1.3 times, so any model with the right sign looks accurate.
2. The same systematic flattening in both random ray sets and both chips. The miss coefficients of the model were fitted on our own harness kernel (Memory_4), whose grouping gain is 8 to 13 percent because its own loads and stores dominate (Finding 100 and the ablation note), while OptiX gains 2 times and more. A slope fitted on a kernel that hides the effect underpredicts a large effect everywhere; for 1.1 to 1.3 times the error cannot be large.
3. Not scene specific: the real scene, where only 35 percent of the rays hit anything (q 1.31, small gain), had the small error for the same reason as 1.

Not tested here: refitting on OptiX data (would give the same 0.2 in sample but is no longer a prediction), other GPUs, ray sets other than uniform random.
