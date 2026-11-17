# OptiX per call overhead at saturation: results (2026-11-16 night, tethys GPU 1, RTX 4090, against model/predictions_ovh.md)

Predictions committed and pushed first (e878d4f). One run of the sweep (raw notes/ovh1/{tri1,soup}_{empty,trace}_20261116T223642.json, private), plus one ncu launch per variant (1,048,576 rays). OptiX only, no TTU code of ours, no fault, GPU idle after. Program ttu/bench/Ovh_1, analysis model/ovh_analyze.py. c = cycles per ray per SM (128 SMs, SM clock measured per warp with clock64 / globaltimer: 2.4 to 2.5 GHz under load, ncu 2.49 to 2.52 for every variant). The EMPTY raygen is too short for globaltimer, so its cycles use the TRACE clock at the same size.

## Table (tri1 first, then soup; nrep = optixTrace calls per ray, same ray, calls after the first are warm)
```

== tri1   (c = cycles per ray per SM, f = measured SM GHz, wl = warp latency cycles median / p90)
        N   w/SM |  empty c     wl | trace1 c      f      wl | trace3 c | trace0 c |       d1       d3 | wl1 / wl3 / wl0 / wlE (median)
       32   0.01 | 169588.60    286 | 169588.60   2.44    4988 | 169466.02 | 209780.00 |     0.00   -61.29 | 4988 / 11838 / 314 / 286
     4096   1.00 |  1628.82    274 |  1332.10   2.48    5704 |  1371.37 |  2281.46 |  -296.71    19.63 | 5704 / 13583 / 306 / 274
    16384   4.00 |   202.99    274 |   309.32   2.42    6245 |   355.35 |   305.74 |   106.33    23.01 | 6245 / 14180 / 303 / 274
    65536  16.00 |    54.30    327 |    83.22   2.47    9880 |   124.01 |    94.76 |    28.92    20.40 | 9880 / 23489 / 402 / 327
   262144  64.00 |    20.81    308 |    31.83   2.45    5717 |    60.99 |    19.24 |    11.02    14.58 | 5717 / 19156 / 346 / 308
  1048576 256.00 |     7.39    303 |    19.71   2.46    5405 |    45.42 |     7.53 |    12.32    12.85 | 5405 / 18343 / 330 / 303
  4194304 1024.00 |     3.74    302 |    14.95   2.49    5364 |    41.55 |     3.47 |    11.21    13.30 | 5364 / 18136 / 326 / 302
  8388608 2048.00 |     3.12    302 |    14.36   2.50    5356 |    40.76 |     2.89 |    11.24    13.20 | 5356 / 18098 / 326 / 302
  per call warp latency, one more warm call (wl3 - wl1)/2 at each size: {32: 3425, 4096: 3940, 16384: 3968, 65536: 6804, 262144: 6720, 1048576: 6469, 4194304: 6386, 8388608: 6371}

== soup   (c = cycles per ray per SM, f = measured SM GHz, wl = warp latency cycles median / p90)
        N   w/SM |  empty c     wl | trace1 c      f      wl | trace3 c | trace0 c |       d1       d3 | wl1 / wl3 / wl0 / wlE (median)
       32   0.01 | 173107.04    419 | 234203.64   2.49   16932 | 257882.52 | 195386.67 | 61096.60 11839.44 | 16932 / 31719 / 431 / 419
     4096   1.00 |  1457.25    419 |  2023.96   2.53   19491 |  3334.06 |  1558.15 |   566.71   655.05 | 19491 / 46944 / 432 / 419
    16384   4.00 |   218.80    419 |   657.04   2.50   26666 |  1032.39 |   743.93 |   438.23   187.68 | 26666 / 60677 / 432 / 419
    65536  16.00 |    79.16    546 |   242.43   2.47   61587 |   519.66 |    58.28 |   163.27   138.62 | 61587 / 178644 / 628 / 546
   262144  64.00 |    22.72    486 |   152.34   2.48   48654 |   411.65 |    19.59 |   129.62   129.65 | 48654 / 158416 / 522 / 486
  1048576 256.00 |     7.52    475 |   130.67   2.51   48976 |   377.77 |     7.28 |   123.15   123.55 | 48976 / 156128 / 501 / 475
  4194304 1024.00 |     3.85    472 |   125.36   2.52   49129 |   382.38 |     4.15 |   121.51   128.51 | 49129 / 155836 / 502 / 472
  8388608 2048.00 |     3.54    471 |   124.32   2.52   49124 |   367.99 |     3.50 |   120.78   121.84 | 49124 / 155852 / 502 / 471
  per call warp latency, one more warm call (wl3 - wl1)/2 at each size: {32: 7394, 4096: 13726, 16384: 17005, 65536: 58528, 262144: 54881, 1048576: 53576, 4194304: 53354, 8388608: 53364}
```
(N = 32 and 4,096 rows: event time is dominated by the launch itself (about 16 us), ignore c there; use the warp latency columns.) ncu warp instructions per warp (32 rays): EMPTY 142 (tri1) / 186 (soup, longer hash); TRACE nrep 0: 172 / 228; nrep 1: 675 / 783; nrep 3: 1,859 / 2,071. So one optixTrace call is 503 instructions (first) and 592 (each further call) on tri1; 555 and 644 on soup (the soup raygen has the longer ray hash).

## Reading
- Saturated (8,388,608 rays, 2,048 warps per SM launched), tri1: nrep 1 c = 14.4, EMPTY 3.1 (fixed launch time included: 7.4 at 1M), nrep 0 (call skipped, trace code present) 2.9, d1 = 11.2, one more warm call d3 = 13.2. soup: nrep 1 c = 124, d3 = 122, EMPTY 3.5.
- o_sat (rule: c(tri1, nrep 1) minus 7.1, the harness unit rate 0.14 rays per cycle per SM from F176) = 7.3 cycles per ray per SM. If the unit alone runs at the ramp rate of 0.09 to 0.10 (the whole span rate of F176), o_sat is 3.7 to 4.1. Subtracting the EMPTY raygen's own 3.1 gives 4.2 (or 0.6). So o_sat is 4 to 7 on the trivial scene: BELOW the F185 assumed range of 9 to 28, rule branch "o_sat <= 9". Issue bound of the call itself: 592 instructions / 32 rays / 4 schedulers = 4.6 cycles per ray per SM; d3 minus the unit's 7.1 is 6.1: consistent with the issue bound plus a little, not with 4,000 cycles per 32 rays (125 per ray) unhidden.
- Hidden at saturation? tri1: NOT (o_sat / c = 0.51; OptiX 14.4 against the unit's 7.1, two times). soup: yes (7.3 / 124 = 0.06, below the 0.10 rule). The overhead is hidden once the rays walk a real tree (unit bound), exposed when the unit is nearly idle.
- One warp (32 rays, tri1): first call (cold) warp latency 4,988 cycles, one more warm call 3,425; soup 16,932 and 7,394; EMPTY 286 (tri1), 419 (soup). The overhead per call as a latency does not stay at one warp's value: the extra warm call grows with load, 3,425 (1 warp) / 3,940 (1 warp per SM) / 3,968 (4 per SM) / 6,804 (16) / 6,386 (2,048 warps per SM, saturated), because the instruction stream is shared (592 instructions per warp per call, issue limited at 4 per cycle per SM).
- Time per ray (d1, EMPTY subtracted, tri1): 28.9 at 16 warps per SM (N = 65,536), 11.0 at 64 (N = 262,144), 12.3 / 11.2 / 11.2 at 256 / 1,024 / 2,048: the per ray cost is flat from about 64 warps per SM launched on (first row with the launch constant mostly removed).
## Against the predictions
1. One warp tri1: first call 4,988 (predicted 6,000 to 7,500: MISS low), one more warm call 3,425 (4,300 to 5,100: MISS low), EMPTY 286 (under 400: held). F178's 6,856 and 4,750 came from a raygen with ray loads, result stores and clock probes; this one builds the ray from the index.
2. Warp latency within 20 percent of one warp up to 4 warps per SM: +14 percent at 1 per SM, +25 percent at 4 (marginal miss); knee of the time per ray at 16 to 32 warps per SM (N = 65,536 to 262,144): by d1 held at 262,144 (not at 65,536, +160 percent), by c the 10 percent point is only reached at N >= 4M because the launch constant (about 16 us) dominates until then.
3. Saturated c 10 to 16: held (14.4). EMPTY under 1.5: MISS (3.1; its own 142 instructions, hash and sink, plus part of the launch constant).
4. d3 7 to 12 cycles: MISS slightly above (13.2); the claim "not hidden on a trivial scene" held (o_sat / c 0.51); ncu per extra call 300 to 700 instructions held (592); nrep 1 total 900 to 1,300: MISS low (675).
5. soup c 130 to 190: MISS slightly low (124; the raygen builds its own rays, F182's 0.508 ns per ray included ray loads and result stores; this run 0.386 ns); exposed share under 10 percent held (6 percent).
6. SM clock 2.5 to 3.1 GHz: held at the low end (2.4 to 2.5); same within 5 percent for EMPTY and TRACE: only seen in ncu (2.49 vs 2.5), not in the per warp clocks.
## What it means for F185 (no refit)
- The per call overhead at saturation is 4 to 7 cycles per ray per SM on a trivial scene, below the 9 to 28 assumed range, and about 6 percent of a random ray scene. Adding o = 7 to the F113 validation (estimate by interpolating F185's 9 case: rms log error of the six speedups 0.087 to about 0.098 at 100,000 triangles, not recomputed) still worsens it, so the F185 conclusion holds: no per call term, no refit; the harness unit rate plus the existing terms remain the model. Where it matters: scenes where the unit is nearly idle (one triangle, very short rays): OptiX is about two times the harness there (F178's one warp gap is not a saturation gap).
- Limits: the unit reference of 7.1 is the harness one triangle query from F176, not an OptiX ray with this exact ray set; the soup rays here are index hashed (cube origin, direction from a cube normalized) not the F182 rays; nrep 3 repeats the same ray (warm). Only one scene size per scene (tri1, soup 100,000); the 4090 SM clock was not fixed.
