# Knee term against the camera shuffle penalty (2026-11-17, tethys; CPU only, existing data, nothing refit; written BEFORE the comparison)

Question: the F123 form under predicts the cost of shuffling a coherent camera order (F208: measured shuffled over natural 0.79 to 0.84, F123 0.95 to 0.99). Does adding the F201 capacity (about 10 distinct node lines per warp step) explain it?
Disclosure: the measured shuffled speedups (0.789, 0.843, 0.839 camera; 0.975, 0.962, 0.808 bounce) were already known to me (F208, printed before this file). The form below was fixed before its numbers were computed; it has no free parameter.
Form (model/predict_knee_shuffle.py): h = 32 / share_ratio (the replay's raw over deduplicated node line counts, the F123 json), the mean distinct lines per warp step if every lane is active (an UPPER bound: inactive lanes lower it);
K(h) = max(1, h / 10); predicted speedup of the shuffled order = F123 prediction x K(h natural) / K(h shuffled). Capacity 10 from F200/F201 (post hoc there, unit bound regime, one SM, no hit rays), applied here to whole launches of real scenes: a stretch, stated.
Table (output of the script, notes/hs2/knee_shuffle_pred_20261117.json):
| set | h natural | h shuffled | K nat | K shuf | F123 | knee form |
|---|---|---|---|---|---|---|
| conference camera | 2.10 | 11.57 | 1.00 | 1.16 | 0.987 | 0.853 |
| sponza camera | 1.74 | 13.05 | 1.00 | 1.30 | 0.964 | 0.739 |
| rungholt camera | 1.21 | 10.58 | 1.00 | 1.06 | 0.954 | 0.901 |
| conference bounce | 10.32 | 17.68 | 1.03 | 1.77 | 0.945 | 0.551 |
| sponza bounce | 14.34 | 18.01 | 1.43 | 1.80 | 0.947 | 0.754 |
| rungholt bounce | 11.79 | 18.78 | 1.18 | 1.88 | 0.817 | 0.513 |
Written expectations: (1) camera sets: the knee form is closer to the measurement than F123 on at least 2 of 3 sets and its mean absolute log error is below F123's 0.17 (probability 65 percent); passing is a mean absolute log error of at most 0.08 on the three camera sets (probability 40 percent). (2) bounce sets: the form OVER penalises (natural bounce is already above the capacity, and F199 shows the cost saturates near 4 to 5 times, so K should flatten; the form has no saturation): predicted 0.51 to 0.75 against measured 0.81 to 0.98, worse than F123 on all three (probability 80 percent). If (2) holds the form is NOT usable as a general term, only as a coherent order locator.
Scoring: log(predicted / measured) per set, rms and mean absolute error per group (camera, bounce, all six) for F123 and the knee form, printed by `predict_knee_shuffle.py compare`. No refit, no retuning after the comparison; a changed form would be a separate labeled step.
