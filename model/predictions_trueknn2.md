# TrueKNN SECOND ATTEMPT: rule and per round predictions, committed BEFORE comparing with the 200,000 / 400,000 measurements (2026-11-16)

Label: second attempt. F156 (first attempt) stays as reported. Honest status: the F156 comparison tables were already seen, and one raw 200,000 point run file (notes/q20/tk/200000_1.txt, round spans and callback counts) was printed before the rule was chosen, so the held out sets are held out only in the sense that no number from them was used to build the rule (rule from the 100,000 point set A alone, three candidate rules compared on A, the one with the smallest in-sample rms chosen). Point files of F156 are gone, so the CPU features use regenerated uniform points (same distribution, fixed seeds); F156 predicted 5,879 warp calls in round 1 of A, this regeneration 5,792.

Rule (one): the launch time of a round is proportional to the number of candidates the intersection program sees, hits = sum over active queries of the exact Chebyshev box count (self included): T = 6.297e-07 ms x hits. That is 224 SM cycles per candidate (142 SMs, 2.5 GHz). Equivalent per call statement: callback cost per warp call grows with the active lanes per call (hits per call), not with the call count; calls alone (F156 style, one pooled cost per call) gives rms log error 1.00 on A against 0.38 for hits and 0.60 for expected heap inserts. Active rule, stop rule, rounds: unchanged from F156.

Calibration (in sample, shown for the shape only): log error of the per round time on A, rounds 1 to 5: -0.68 +0.30 -0.01 +0.17 -0.37 (round 1 is the first launch, cold; round 5 has 2,700 active warps with about 2 active lanes)

## 200000 points (held out)

| B round | radius | active (predicted) | hits (predicted) | hits per active query | launch ms (predicted) |
|---|---|---|---|---|---|
| 1 | 0.004 | 200000 | 220632 | 1.1 | 0.1389 |
| 2 | 0.008 | 200000 | 362846 | 1.8 | 0.2285 |
| 3 | 0.016 | 200000 | 1481988 | 7.4 | 0.9333 |
| 4 | 0.032 | 199766 | 10180963 | 51.0 | 6.4114 |
| 5 | 0.064 | 174 | 22677 | 130.3 | 0.0143 |
| total | | | | | 7.7264 |

## 400000 points (held out)

| C round | radius | active (predicted) | hits (predicted) | hits per active query | launch ms (predicted) |
|---|---|---|---|---|---|
| 1 | 0.004 | 400000 | 481392 | 1.2 | 0.3032 |
| 2 | 0.008 | 400000 | 1047822 | 2.6 | 0.6599 |
| 3 | 0.016 | 400000 | 5518486 | 13.8 | 3.4752 |
| 4 | 0.032 | 307795 | 30361513 | 98.6 | 19.1200 |
| 5 | 0.064 | 1 | 121 | 121.0 | 0.0001 |
| total | | | | | 23.5584 |

What can fail: the cost per candidate differs between rounds by 2 times already on A (161 to 446 SM cycles per candidate by the callback probe), cold first launch rounds, heap insertion cost (F138) at larger sizes. Success rule (set before): total within a factor 1.5 of measured, per round log error reported, rounds equal. No refitting after the comparison.
