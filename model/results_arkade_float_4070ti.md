# Arkade distance function: double (std::pow(float, int) promotes to double) against float powf and a plain square (queue 14, 4070 Ti, L2 build, 2D synthetic data)

| config | variant | launch ms median (min to max of 5) | speedup over double | probe latency per call (cycles) | neighbours found (ids) differing from double | max relative difference of the distances (same id) |
|---|---|---|---|---|---|---|
| a1e5 r 0.005 | double | 21.45 (21.16 to 25.09) | 1.00x | 358329 | 0 of 100000 queries differ (0 of 781477 slots) | 0 |
| a1e5 r 0.005 | float | 25.82 (25.48 to 26.22) | 0.83x | 350010 | 0 of 100000 queries differ (0 of 781477 slots) | 1.2e-07 |
| a1e5 r 0.005 | square | 25.27 (23.35 to 25.73) | 0.85x | 378404 | 0 of 100000 queries differ (0 of 781477 slots) | 1.2e-07 |
| a1e5 r 0.0112 | double | 30.39 (28.76 to 31.44) | 1.00x | 120393 | 0 of 100000 queries differ (0 of 1599706 slots) | 0 |
| a1e5 r 0.0112 | float | 30.55 (27.45 to 31.86) | 0.99x | 91548 | 0 of 100000 queries differ (0 of 1599706 slots) | 1.2e-07 |
| a1e5 r 0.0112 | square | 31.66 (29.02 to 32.75) | 0.96x | 73938 | 0 of 100000 queries differ (0 of 1599706 slots) | 1.2e-07 |
| a1e6 r 0.0016 | double | 24.27 (22.39 to 27.50) | 1.00x | 374335 | 0 of 100000 queries differ (0 of 803816 slots) | 0 |
| a1e6 r 0.0016 | float | 26.67 (22.76 to 30.54) | 0.91x | 275534 | 0 of 100000 queries differ (0 of 803816 slots) | 1.2e-07 |
| a1e6 r 0.0016 | square | 24.10 (22.02 to 24.82) | 1.01x | 268744 | 0 of 100000 queries differ (0 of 803816 slots) | 1.2e-07 |

## Where the Arkade launch time goes: intersection program bisection (100,000 points and queries, r 0.005, L2 build, probes on, one launch each; the probes cost under 5 percent here)
| variant of the intersection program | launch ms | callback program cycles per warp call (probe) |
|---|---|---|
| full program (double precision pow, result list update) | 23.1 | 322,000 |
| returns immediately (empty program) | 23.1 | 19,600 |
| distance in double, no result list access | 22.7 | 172,800 |
| distance as a float square, no result list access | 22.3 | 135,600 |
| double distance, result list scan but no write | 21.3 | 240,000 |
An empty intersection program gives the same 23 ms. So neither the double precision pow (no speedup from float or a plain square, previous table) nor the intersection program body (result list in local memory) sets the launch time; the time is in the traversal and hand-off path of this program (instance level plus custom primitives through OWL, results written to host pinned memory: the raygen part outside the trace window is 27 percent of warp time with the empty program). The cycles per call of the probes are then mostly time inside the program window that is not the program's own work (an empty program shows 19,600 cycles per call here against 125 in the kNN program). Finding 132's attribution of the 100 to 200 times gap to double precision pow is withdrawn.
