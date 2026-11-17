# tmax replay check against F182: predictions (2026-11-16 night, tethys, written BEFORE the replay is run; CPU only for the replay)

Question (F185 open): does the CPU replay (model/replay/replay.cpp, takes tmax per ray) reproduce F182's measured tmax sweep (soup of 100,000, 262,144 random rays, unit L1 requests per ray 20.7 / 22.9 / 25.2 / 31.5 / 41.8 / 56.8 / 75.2 / 80.2 / 80.7 / 80.8 and hit fraction 0 / 0.04 / 0.12 / 0.19 / 0.38 / 0.53 / 0.75 / 0.78 / 0.81 / 0.81 at tmax 0 / 0.01 / 0.02 / 0.05 / 0.1 / 0.2 / 0.5 / 1 / 2 / unlimited; time per ray 0.188 / 0.188 / 0.219 / 0.246 / 0.289 / 0.371 / 0.469 / 0.496 / 0.507 / 0.508 ns)?

Method: the Flags_1 host gets a dump option (GAS bytes as built with gopaque = 1 like F182, the triangles, and the 262,144 rays, no change to anything else; one OptiX build, no timing). make_table.py (box matching, fine on a fresh tree, F167) builds the node table; replay ORDER 0 (depth first, nearest first, hit only return) on the same rays with tmax set to each value of the sweep. Metrics per ray: nodes fetched (n), leaf slots entered (l), triangles tested.

Predictions:
1. Hit fraction: within 0.02 of the measured value at every tmax (the replay's test is exact geometry).
2. Request shape: R(tmax) = (n + l) per ray divided by its unlimited value equals the measured requests ratio (0.256 / 0.283 / 0.312 / 0.390 / 0.517 / 0.703 / 0.931 / 0.993 / 0.999 / 1) within 10 percent at every tmax, and within 20 percent at tmax 0 and 0.01 (the zero length ray reads the nodes holding the origin: the replay should give 14 to 28 per ray there).
3. Absolute: (n + l) per ray unlimited is 65 to 100 (measured requests 80.8); n alone 45 to 80. Which of the two is the better match is reported, not chosen after the fact: I put 60 percent on n + l.
4. Time: the measured time per ray is linear in the measured requests (checked on the F182 numbers: endpoints line 0.188 + 0.00532 x (R - 20.7) gives 0.246 at 31.5, measured 0.246). Replay time ratio predicted by the same line with the replay requests (scaled to the measured unlimited value) is within 10 percent of the measured time at every tmax.
Reading rule: the replay is accepted for tmax as a requests model if prediction 2 holds at all ten values; if it fails only at tmax 0 to 0.02 the failure is a zero length / origin box effect (fixed per ray cost the replay lacks), reported as such.
