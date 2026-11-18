# Results Phase 10e (tethys, RTX 4090, 2026-11-18). Predictions: model/predictions_phase10e.md (commits e07d778, addendum 2b d1fffcc)

## 1. Stage 1 of the transform sequence (F251)
The IAS group of F139 and F195 (checked builds rebuilt in /dev/shm/t9x, check_seq ALL PASS both stages), one warp, one launch per matrix and ray, 5 matrices (identity, rotation z 37 degrees with translation, rotation 90 degrees with scale 2, rotation x 37 degrees, non uniform scale) x 6 rays (hit, moved, moved2, miss, t2, back).
- Stage 0 first (no TTU instruction): 30 of 30 agree with the float64 CPU reference on the moved triangle and inputs intact.
- Stage 1: 30 of 30 launches exit 0, "no error", input registers unchanged, six decoded registers identical in all 32 lanes, hit flag equal to OptiX in the same process; the 20 launches that hit return distance, u and v bit identical to OptiX's (sign bit ignored; 0 ulp, my criterion was 2 ulp); the 10 misses return the miss encoding. The unit's u carries the sign of facing (set for front facing hits, clear for the back facing hit), as in F139. No fault, GPU idle after.
- Predictions: (a) 30 of 30 held; (b) bit identical for identity and within 2 ulp for the others: held, stronger (bit identical everywhere); (c) lanes identical: held; (d) the 32 different rays launch: not possible with the host (one ray per launch), not run; (e) no fault: held.
- Meaning: the unit applies the instance transform itself (world to object matrix read from the instance leaf, object space hit, parametric t unchanged) with the sequence exactly as OptiX issues it; the SM code before the group does not transform the ray (the register image holds world space words, F234).
Raw: notes/xform1 (private, x1_s0_*, x1_s1_*). Code: ttu/bench/Xform_1/stage1.py.

## 2. F243 close out: one curve test (F252)
Setup: our checked TTU kernel (random rays, soup 100,000 triangles, 32 blocks) beside a plain stream of random 128 byte lines over a footprint F (new type l2fp, 32 lines per warp load, every lane its own line, ld.global.cg), sib = 4 blocks on each TPC sibling SM, high = 2 blocks on each SM of other TPCs. F = 40, 56, 72, 96, 120 (L2 is 72 MB), 512, 1024 MB (the last two DRAM sourced), unthrottled and with a 150 us sleep per iteration; 3 repetitions, alone run before every arm, clocks sampled in every arm; 0 wrong, 0 unstored in all 144 + 48 launches. Stream load latency: lane 0 of every warp times the first load of every 16th iteration (issue to consumption). O = stream lines per second x latency (lines in flight). dramfew has no latency sampling (type 11), so DRAM points come from F 512 and 1024 (addendum 2b). Table: notes/figures/f243_one_curve.png and the analysis output below (model/conc4_analyze.py).

```
arm                         r  r range G lines/s   lat cyc    lat us  O lines O per SM
hi_l2fp402              0.694 0.693-0.704     20.82      1585      0.58    12090    188.9
hi_l2fp562              0.692 0.681-0.700     19.17      1724      0.63    12107    189.2
hi_l2fp722              0.371 0.369-0.378     14.20      1791      0.66     9318    145.6
hi_l2fp962              0.266 0.266-0.267     11.06      2096      0.77     8489    132.6
hi_l2fp1202             0.239 0.234-0.241      9.80      2276      0.83     8168    127.6
hi_l2fp5122             0.190 0.187-0.195      7.60      2653      0.97     7382    115.3
hi_l2fp10242            0.182 0.180-0.185      7.65      2938      1.08     8230    128.6
hi_l2fp402_d150         0.784 0.779-0.794      5.93       469      0.17     1018     15.9
hi_l2fp562_d150         0.653 0.649-0.658      5.55       671      0.25     1364     21.3
hi_l2fp722_d150         0.587 0.582-0.605      5.21       840      0.31     1604     25.1
hi_l2fp962_d150         0.512 0.491-0.515      5.19      1140      0.42     2169     33.9
hi_l2fp1202_d150        0.461 0.459-0.490      5.20       602      0.22     1148     17.9
hi_l2fp5122_d150        0.381 0.379-0.382      5.30       917      0.34     1780     27.8
hi_l2fp10242_d150       0.386 0.365-0.399      5.30       894      0.33     1735     27.1
sib_l2fp404             0.154 0.153-0.156     45.85      2401      0.88    40329   1260.3
sib_l2fp564             0.098 0.098-0.099     30.98      3509      1.29    39822   1244.4
sib_l2fp724             0.100 0.096-0.100     31.18      3298      1.21    37668   1177.1
sib_l2fp964             0.076 0.048-0.077     23.85      3942      1.44    34436   1076.1
sib_l2fp1204            0.037 0.036-0.037     12.40      7240      2.65    32873   1027.3
sib_l2fp5124            0.024 0.023-0.025      7.93     10419      3.82    30266    945.8
sib_l2fp10244           0.024 0.023-0.024      7.94     10418      3.82    30317    947.4
sib_l2fp404_d150        0.695 0.690-0.696      4.97      9715      3.56    17697    553.0
sib_l2fp564_d150        0.599 0.596-0.601      4.91     10550      3.86    18976    593.0
sib_l2fp724_d150        0.524 0.517-0.525      5.09     11141      4.08    20774    649.2
sib_l2fp964_d150        0.426 0.424-0.430      5.18     12366      4.53    23452    732.9
sib_l2fp1204_d150       0.398 0.395-0.398      5.23     11497      4.21    22046    688.9
sib_l2fp5124_d150       0.314 0.314-0.323      5.36     11496      4.21    22557    704.9
sib_l2fp10244_d150      0.326 0.325-0.329      5.32     11975      4.39    23346    729.6
sib: Spearman(r, O) = -0.727 over 14 arms
high: Spearman(r, O) = -0.248 over 14 arms
pair sib_l2fp404          r 0.154 O 40329 | sib_l2fp5124         r 0.024 O 30266 FAIL
pair sib_l2fp404          r 0.154 O 40329 | sib_l2fp10244        r 0.024 O 30317 FAIL
pair sib_l2fp564          r 0.098 O 39822 | sib_l2fp5124         r 0.024 O 30266 ok
pair sib_l2fp564          r 0.098 O 39822 | sib_l2fp10244        r 0.024 O 30317 ok
pair sib_l2fp724          r 0.100 O 37668 | sib_l2fp5124         r 0.024 O 30266 ok
pair sib_l2fp724          r 0.100 O 37668 | sib_l2fp10244        r 0.024 O 30317 ok
pair sib_l2fp404_d150     r 0.695 O 17697 | sib_l2fp5124_d150    r 0.314 O 22557 FAIL
pair sib_l2fp404_d150     r 0.695 O 17697 | sib_l2fp10244_d150   r 0.326 O 23346 FAIL
pair sib_l2fp564_d150     r 0.599 O 18976 | sib_l2fp5124_d150    r 0.314 O 22557 FAIL
pair sib_l2fp564_d150     r 0.599 O 18976 | sib_l2fp10244_d150   r 0.326 O 23346 FAIL
pair sib_l2fp724_d150     r 0.524 O 20774 | sib_l2fp5124         r 0.024 O 30266 FAIL
pair sib_l2fp724_d150     r 0.524 O 20774 | sib_l2fp10244        r 0.024 O 30317 FAIL
pair sib_l2fp724_d150     r 0.524 O 20774 | sib_l2fp5124_d150    r 0.314 O 22557 FAIL
pair sib_l2fp724_d150     r 0.524 O 20774 | sib_l2fp10244_d150   r 0.326 O 23346 FAIL
matched O pairs (within 1.5x): 14, outside 0.10: 10
```
Findings, in order:
1. Dose response (answers the open question of F243): on the TPC siblings random RT falls steadily as the stream's footprint passes the L2: unthrottled 0.154 (40 MB), 0.098, 0.100, 0.076, 0.037 (120 MB), 0.024 (512 MB and 1 GB, DRAM sourced); with the 150 us sleep (5 G lines per second in every arm) 0.70, 0.60, 0.52, 0.43, 0.40, 0.31 and 0.33. Other TPCs: 0.69 (40, 56 MB) to 0.18 (1 GB) unthrottled, 0.78 to 0.38 throttled. So the DRAM factor is graded by the share of lines that miss the L2, not a switch; an L2 resident stream with the same page spread (20 to 60 pages) already costs the siblings a factor 6 to 10.
2. The one curve hypothesis (RT ratio set by outstanding miss occupancy, one pool per TPC) does NOT hold under the rule fixed before the run: pooled Spearman(r, O) on sib is -0.73 over 14 arms (rule: at most -0.9), and 10 of 14 matched pairs (O within 1.5x, one L2 arm and one DRAM arm) differ in r by more than 0.10 (for example F 40 throttled r 0.70 at O 17,700 against F 512 throttled r 0.31 at O 22,600; F 40 unthrottled r 0.15 against F 512 unthrottled 0.024 at O 40,300 against 30,300). My prediction (rule fails, 85 percent) held; my Spearman prediction (at most -0.8) missed by a little (-0.73); r falls with footprint and sib is below high at equal O both held.
3. What O does show: the unthrottled sib arms all sit at 30,000 to 40,000 lines in flight (950 to 1,260 per streaming SM) whatever the footprint, while r spans 0.024 to 0.154; the throttled ones at 17,700 to 23,300. So the stream saturates a window of outstanding loads in every unthrottled arm and RT's loss depends on what the lines cost (latency 2,400 cycles at F 40 to 10,400 at F 512), not on how many are in flight. Caveat: O by Little's law includes the queueing in front of the SM, 950 to 1,260 lines per SM is far above any plausible per SM miss queue, so it is a load window count, not a measured hardware pool; the test cannot say which hardware queue is involved.
4. Throttled arms have a LONGER latency (9,700 to 12,400 cycles) than unthrottled F 40 to 72 (2,400 to 3,500) although they issue fewer lines: with the stream sleeping, the RT blocks run much faster (r 0.3 to 0.7) and the stream's few loads queue behind RT's traffic on the shared TPC path. A likely reading (not tested): the stream's few loads queue behind RT's traffic on the shared TPC path; not separated from other causes.
5. Confounds, disclosed before the run: footprint also changes pages per warp load (14 to 23 at 40 to 120 MB; F243's page effect was a factor 2.5); L2 hit rate is the footprint proxy, not measured; random rays only; one chip.
Closing text for the thread: notes/performance_model.md.
