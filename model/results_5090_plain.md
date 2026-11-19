# RTX 5090 (tgrogers-pc14): plain CUDA part of the core set, scored against the committed predictions (2026-11-19)

Predictions: model/predictions_5090.md (commit a8b9e46) items 1 and 8, and model/predictions_5090_cache.md (429a889, committed before the cache runs). Runs: ttu/bench/pc14/plain_core.py (chase, mem1, l1sect, l1cap, l2lat), GPU 0 (the display GPU, shared), nvidia-smi checked before every process, timeout -k 2 30, one process per case, raw files with timestamps in notes/pc14_core (private). No TTU code. Cycles are SM cycles of the clock64 counter; the SM clock under load was not recorded (shared GPU, no control).

## Pointer chase (one thread, one SM), median of 10 repeats (spread under 1 percent except where noted)
| Case | Predicted | Measured | Score |
|---|---|---|---|
| L1 hit, 16 KB and 64 KB (.ca) | 41 (30 to 50) | 45.8, 44.7 | held (inside the range, 9 to 12 percent above the point) |
| L2 hit, 1 to 64 MB (.ca) | 290 (250 to 340) | 349.3 flat from 1 to 64 MB | MISSED (20 percent above the point, 9 above the range) |
| L2 hit, .cg and cross (L2 after a kernel boundary) | about 290 | 348.0, 346.7 to 348.8 | qualitative result held (L2 value), number missed as above |
| 80 MB, 88 MB | 290 | 349.2, 350.1 | missed as above |
| 88 MB (90,112 KB) | 290 | 350 | missed as above |
| 90 / 92 MB (92,160 / 94,208 KB) | 290 | 360 / 386 | MISSED |
| 93 MB (95,232 KB) | 290 | 386 | MISSED |
| 96 MB (98,304 KB) | 354 (290 to 520) | 569 | MISSED (above the range) |
| 99 MB / 101 MB | 451 / 572 | 715 / 722 | MISSED |
| 106 MB and above (108,544 KB, 112, 128 MB) | 650 | 862 | MISSED |
| DRAM, 256 MB and 512 MB | 650 (560 to 800) | 862, 867 | MISSED (above the range) |
| cold, 64 KB and 4 MB after a 256 MB read | DRAM level | 863, 861.5 | the level is the DRAM level; number missed as above |
Full ramp (KB footprint: cycles): 90,112: 350; 92,160: 360; 94,208: 386; 95,232: 386; 96,256: 388; 97,280: 486; 98,304 (96 MB): 569; 99,328: 637; 100,352: 691; 101,376: 715; 102,400 to 104,448: 721; 106,496: 767; 108,544 and above: 862. The L2 hit latency is 349 up to about 88 MB. The ramp starts at about 0.94 times the L2 size (earlier than the modelled 0.97), has an intermediate level near 387 (92 to 94 MB, steady across three sizes) and another near 721 (99 to 104 MB), and reaches the DRAM value only at 106 MB, 1.1 times the L2. Nothing like the 4090's shape scaled by C. The 4090 form (hit 1 to 0.97 C) is not a model for this chip. Cause not identified (partitioned L2, or a translation effect; nothing was tested).
Reading: the null hypothesis of Ada cycle constants for L2 (290) and DRAM (650) fails by 20 and 33 percent; L1 passes. In time units the DRAM chase is 862 cycles; the SM clock was not sampled, so no ns number is claimed.

## Memory_1, plain chase (stride 128 B lines, 16 to 160 KB, with and without 98,304 bytes of shared memory reserved)
Without the reservation: 44.6 to 45.8 up to 80 KB, 136 at 96 KB, 225 at 112 KB, 322 at 128 KB, 349 at 160 KB (halfway about 105 KB). With the reservation: 45.5 up to 20 KB, 128 at 24 KB, 199 at 28 KB, 261 at 32 KB, 347 at 48 KB (halfway about 28 KB, L1 about 20 to 24 KB). Predicted (item 8 and cache item 4): L1 hit up to about 80 KB without; L1 about 20 KB (16 to 32) with. Both held. The reserved-memory ceiling is 100 KB per SM (detected 102,400 B), equal to Ada: held. The TTU part of Memory_1 (second trace slowdown) was not run (needs TTU code).

## Cache_1 program (F202), three modes
- l1sect (1 MB array): sequential cycles 82.5 / 120.7 / 197.0 / 349.3 at stride 4 / 8 / 16 / 32, so the miss fraction is 0.12 / 0.26 / 0.51 / 1.0 = stride / 32 (a 32 byte sector). Held. Random order at the L2 value 341 to 349 against a predicted 290 (250 to 340): the value missed by 1 to 9 cycles (same L2 miss as above), the sector result held.
- l1cap (stride 128, sequential): L1 flat to 88 KB, 102 at 92 KB, 146 at 96, 186 at 104, 265 at 112, 336 at 124, 349.5 at 128: soft edge 88 to 124 KB, halfway 105 KB (predicted 80 to 128 KB, soft). Held. Strides 32 and 64: halfway 97.5 KB, 7 percent below the stride 128 value (predicted within 10 percent): 128 byte tag line held. Stride 256: halfway 207 KB (same number of lines as the 105 KB at stride 128).
- l2lat (16 MB, .cg, random, 131,072 lines, one SM): median 407 cycles (predicted 270 to 380: MISSED), 87.5 percent of the lines within 15 percent of the median (predicted 95 percent: MISSED). Two groups: 114,686 lines at 360 to 440 (graded, no near and far split inside it), and 16,386 lines (12.5 percent, exactly 1 in 8) at 610 to 777. The 1 in 8 share is the signature of the timing artifact seen on the 6000 Ada (F203: present at 12.5 percent), but the artifact was not separated here, so "one mode" is not confirmed or refuted. The main group is 17 percent above the l1sect random value (349), the usual overhead of the per load timing method (F202: 291 to 352 against 285).

## Tally
Scored lines above. Held: L1 hit, Memory_1 both arms, shared memory ceiling, sector 32 B, L1 knee, tag line, L2 level vs kernel boundary (qualitative). Missed: L2 hit number, the whole L2 ramp (4 points), DRAM number, l2lat median and spread, l1sect random value (marginal). The Ada L1 numbers carry over; the Ada L2 and DRAM numbers do not.
