# Extra hardware hits at short tmax: result (2026-11-17, tethys; prediction model/predictions_hwhits.md)

THERE ARE NO EXTRA HITS. The question rested on a wrong number: the F182 text gave the hit fraction series 0 / 0.04 / 0.12 / 0.19 / 0.38 / 0.53 / 0.75 / 0.78 / 0.81 / 0.81, which was taken from the 32 ray probe launches (granularity 1/32), not from the 262,144 ray launch (0 / 0.038 / 0.075 / 0.175 / 0.314 / 0.509 / 0.754 / 0.823 / 0.829 / 0.829, in the summary.json files of the same runs). F182, F187 and model/results_flags.md are corrected (the requests and times in F182 were right).

Check done anyway, on all 262,144 rays: new OptiX runs (Flags_1 soup:100000, tmax 0.02 / 0.1 / 0.2 / 1, per ray results written with the new dumpres option) against an exact CPU brute force (model/brute_tfirst.cpp, double precision Moller-Trumbore over all 100,000 triangles, 21 seconds on 20 threads; nearest hit t per ray):
```

tmax 0.02: hw hit 0.0747  exact hit 0.0747  | hw hit, exact miss: 0 (0.0000)   hw miss, exact hit: 0 (0.0000)   both agree 1.0000
  both hit: hw t equals exact t within 1e-5 relative: 1.0000, prim equal 1.0000

tmax 0.1: hw hit 0.3139  exact hit 0.3139  | hw hit, exact miss: 0 (0.0000)   hw miss, exact hit: 0 (0.0000)   both agree 1.0000
  both hit: hw t equals exact t within 1e-5 relative: 1.0000, prim equal 1.0000

tmax 0.2: hw hit 0.5093  exact hit 0.5093  | hw hit, exact miss: 0 (0.0000)   hw miss, exact hit: 0 (0.0000)   both agree 1.0000
  both hit: hw t equals exact t within 1e-5 relative: 1.0000, prim equal 1.0000

tmax 1: hw hit 0.8230  exact hit 0.8230  | hw hit, exact miss: 0 (0.0000)   hw miss, exact hit: 0 (0.0000)   both agree 1.0000
  both hit: hw t equals exact t within 1e-5 relative: 1.0000, prim equal 1.0000
```
So at every tmax the hardware hit decision, hit distance (within 1e-5 relative) and primitive equal the exact geometry on every ray: no extra hits, no missed hits, no precision effect at tmax (hardware equal to double precision). Prediction: both alternatives (extras just beyond tmax, or extras elsewhere) missed, since the premise was false. The replay's slightly low hit fraction (0.01 under) is the replay's own error: its box matching leaf assignment misses the nearest hit in about 3.5 percent of rays (7 of 200 rays checked: it returns a farther hit; 10,523 triangles had no containing leaf slot in the table build).
Mistake to note: the series was copied into F182, my predictions for F187 and the comparison without checking it against the summary files; the F187 conclusion about a hardware hit gap was wrong and is withdrawn.
