"""RTX 5090 model with chip measured inputs (pc14, 2026-11-19). POST HOC: the RT measurements it is scored against already exist
(model/results_5090_core.md). Not a pre-registered prediction.

Form (fixed in this file before any score was computed; no variant was tried and discarded):
  The Ada model (model/predictions_5090.md items 2 to 4) has three parts per scene:
    S   in kernel second trace (unit + L1 only)            Ada: 782 / 827 / 868 at 1 / 2 / 3 levels (+42 per level beyond)
    W   warm launch (L2 warm)  = S + k_L2 * (L2 - L1)       k_L2 = (W_ada - S_ada) / (L2_ada - L1_ada)   (serial L2 fetches on the path)
    C   cold launch (after 256 MB read) = W + k_D * (DRAM - L2),  k_D = (C_ada - W_ada) / (DRAM_ada - L2_ada)
  k_L2 and k_D per scene come from the Ada numbers only. The 5090 enters ONLY through its plain CUDA pointer chase latencies
  (F403: L1 45.8, L2 349.3, DRAM 862 cycles; one thread, no RT) and the unit part S stays the Ada constant (the 5090's own second
  trace, 790, is NOT used). Ada inputs: L1 41, L2 290, DRAM 650 (the point values of predictions_5090.md item 1).
"""
ADA = dict(L1=41.0, L2=290.0, DRAM=650.0)
CHIP = dict(L1=45.8, L2=349.3, DRAM=862.0)          # model/results_5090_plain.md (F403), plain chase medians
SCENES = [  # name, levels, Ada warm, Ada cold (predictions_5090.md item 3), measured warm, cold, second trace (results_5090_core.md)
    ('1 triangle', 1, 1600, 2550, 1788, 3306, 790), ('grid 12', 1, 1650, 2680, 1791, 3385, 790),
    ('grid 144', 2, 1900, 3300, 2247, 4229, 841), ('grid 200', 2, 1920, 3280, 2177, 4141, 838),
    ('grid 2,500', 3, 2250, 3890, 2551, 5137, 884), ('grid 4,096', 3, 2250, 4000, 2571, 5289, 884),
    ('grid 20,736', 4, 2580, 4640, 2994, 6009, 939)]
S_ADA = {1: 782.0, 2: 827.0, 3: 868.0, 4: 910.0}    # 4 levels: 868 + 42 (assumed step; the other steps are 45 and 41)


def predict(name, lv, w_a, c_a):
    s = S_ADA[lv]
    kL2 = (w_a - s) / (ADA['L2'] - ADA['L1']); kD = (c_a - w_a) / (ADA['DRAM'] - ADA['L2'])
    w = s + kL2 * (CHIP['L2'] - CHIP['L1']); c = w + kD * (CHIP['DRAM'] - CHIP['L2'])
    return s, kL2, kD, w, c


def main():
    print('| Scene | k_L2 | k_DRAM | S (Ada) | Warm pred | Warm meas | err | in 10% | Cold pred | Cold meas | err | in 15% |')
    print('|---|---|---|---|---|---|---|---|---|---|---|---|')
    wh = ch = 0
    for n, lv, wa, ca, wm, cm, sm in SCENES:
        s, kL2, kD, w, c = predict(n, lv, wa, ca)
        ew, ec = wm / w - 1, cm / c - 1
        wh += abs(ew) <= .10; ch += abs(ec) <= .15
        print('| %s | %.2f | %.2f | %.0f | %.0f | %d | %+.1f%% | %s | %.0f | %d | %+.1f%% | %s |' % (n, kL2, kD, s, w, wm, 100 * ew, 'yes' if abs(ew) <= .10 else 'no', c, cm, 100 * ec, 'yes' if abs(ec) <= .15 else 'no'))
    print('warm inside 10%%: %d of 7; cold inside 15%%: %d of 7' % (wh, ch))
    # Latency_6: one triangle, Ada warm 1,630 cold 2,600 (predictions item 4); measured 1,788 and 3,306 (difference 1,518)
    kD = (2600 - 1630) / (ADA['DRAM'] - ADA['L2']); d = kD * (CHIP['DRAM'] - CHIP['L2'])
    print('Latency_6 difference: Ada 970 -> pred %.0f, measured 1518 (%+.1f%%); the old prediction was 950 to 1000 (%+.1f%% off at 975)' % (d, 100 * (1518 / d - 1), 100 * (1518 / 975 - 1)))
    # Memory_5 other TPC: equals the alone (warm) value of the scene. Ada points: tri1 1,600, grid 144 1,900, grid 2,500 2,250
    print('Memory_5 other TPC (= warm of the scene):')
    for (n, lv, wa, ca, wm, cm, sm), meas in zip([SCENES[0], SCENES[2], SCENES[4]], (1810, 2267, 2637)):
        s, kL2, kD, w, c = predict(n, lv, wa, ca)
        print('  %s: pred %.0f, measured %d (%+.1f%%), inside 10%%: %s' % (n, w, meas, 100 * (meas / w - 1), abs(meas / w - 1) <= .10))
    # honesty check: how much of the change is just the two ratios
    print('ratios: (L2-L1) %.3f, (DRAM-L2) %.3f' % ((CHIP['L2'] - CHIP['L1']) / (ADA['L2'] - ADA['L1']), (CHIP['DRAM'] - CHIP['L2']) / (ADA['DRAM'] - ADA['L2'])))


if __name__ == '__main__':
    main()
