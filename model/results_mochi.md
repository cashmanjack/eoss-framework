# Mochi results (predictions: model/predictions_mochi.md, 286faf2; comparison script model/mochi_compare.py)

2026-11-16, triton GPU 1, run B (recorder plus dump, patched copy; private data and patch diff in notes/q24). Measured per frame: intersection calls per query of the LAST launch of the frame (exact, counter per launch index) and the in kernel %globaltimer span of each of the 400 launches of the frame (min, median, max). "B own replay" = replay of run B's own dumped GAS and particles.

Correction to the predictions file: its column "exact box hits (incl. self)" adds 1 to a count that already contains the particle itself, so it is too high by 1 (frame 0: 1.00, not 2.00; frame 9: 2.42 not 3.42). Nothing else uses it.

| frame | measured calls / query | A1 replay (predicted) | B own replay | pred error A1 | error B own | measured lane eff | A1 lane eff | B own lane eff | span median ms | span min | span max | RT part lower bound ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1.075 | 1.082 | 1.082 | +0.6% | +0.6% | 0.602 | 0.602 | 0.602 | 0.1055 | 0.1034 | 0.1075 | 0.0339 |
| 1 | 1.075 | 1.082 | 1.082 | +0.6% | +0.6% | 0.602 | 0.602 | 0.602 | 0.0993 | 0.0973 | 0.1065 | 0.0339 |
| 2 | 1.075 | 1.082 | 1.082 | +0.6% | +0.6% | 0.602 | 0.602 | 0.602 | 0.0993 | 0.0973 | 0.1004 | 0.0339 |
| 3 | 1.075 | 1.082 | 1.082 | +0.6% | +0.6% | 0.602 | 0.602 | 0.602 | 0.0993 | 0.0973 | 0.1004 | 0.0339 |
| 4 | 1.075 | 1.082 | 1.082 | +0.6% | +0.6% | 0.602 | 0.602 | 0.602 | 0.0993 | 0.0973 | 0.1004 | 0.0339 |
| 5 | 2.651 | 2.422 | 2.422 | -8.6% | -8.6% | 0.446 | 0.406 | 0.406 | 0.0993 | 0.0973 | 0.1198 | 0.0357 |
| 6 | 3.713 | 3.109 | 3.107 | -16.3% | -16.3% | 0.460 | 0.412 | 0.412 | 0.1126 | 0.1044 | 0.1321 | 0.0366 |
| 7 | 5.764 | 4.218 | 4.210 | -26.8% | -27.0% | 0.517 | 0.451 | 0.450 | 0.1393 | 0.1239 | 0.1649 | 0.0382 |
| 8 | 8.464 | 5.396 | 5.280 | -36.2% | -37.6% | 0.536 | 0.483 | 0.476 | 0.1812 | 0.1536 | 0.2150 | 0.0398 |
| 9 | 12.941 | 6.497 | 6.427 | -49.8% | -50.3% | 0.544 | 0.497 | 0.501 | 0.2386 | 0.2058 | 0.2765 | 0.0413 |

span ratio frame 9 / frame 0 (medians): 2.26; frames 0 to 4 medians: 0.1055 0.0993 0.0993 0.0993 0.0993

Verdicts against the rules set before the run:
- Counts, frames 0 to 4 (identical state): predicted 1.082, measured 1.075 (+0.6%), lane efficiency 0.602 both: HELD (10% and 5% rules).
- Counts, frames 5 to 9: own replay error -8.6, -16, -27, -38, -50 percent (rule 5%): FAILED; run A1 table error the same (rule 25%: held at frames 5 and 6, failed from frame 7). The measured calls are higher than the replay for 91 percent of the queries at frame 9 (12.9 against 6.4 on average, per query correlation 0.70; frame 0 correlation 1.000). Cause not found; untested guesses: the replay puts primitives into leaf slots by box matching and that is ambiguous in a refit tree whose leaf boxes overlap; or the hardware enters more leaf slots than the replay on a refit tree. Lane efficiency measured 0.446 to 0.544 against 0.406 to 0.501 (frames 5 to 9; rule 0.05: held at frames 5, 6 and 9 with 0.040, 0.048, 0.047 off; FAILED at frames 7 and 8 with 0.066 and 0.053).
- Time: statement 1 (span at least the RT part) held in every frame; statement 2 (bracket 0.034 to 0.17 ms for frames 0 to 4) held: 0.099 to 0.106 ms, about 3 times the RT part; statement 3 (frame 9 over frame 0 above 1.5) held: 2.26 (0.239 against 0.106 ms; against frame 1 to 4 at 0.099: 2.4). Spans of frames 7 to 9 (0.14, 0.18, 0.24 ms) are above the candidate only form (0.038 to 0.041 ms) by 3.7 to 5.8 times, so the node count (6 to 253 per query by replay) matters; but the replay counts are themselves too low from frame 5, so the node term cannot be checked.
- Recorder overhead: the program's own host timers printed 585 ms for launches against 285 ms unpatched (host sync and counter copies per launch), the device span itself carries a globaltimer read and two atomics per ray. Not separated. Timer values come in steps of about 3 microseconds.
