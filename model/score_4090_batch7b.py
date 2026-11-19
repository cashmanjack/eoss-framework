"""Part B of model/predictions_4090_batch7.md: RTNN full against stub on the 4090, calibrated v1 form, rescore of F255 and F419 on kernel time (labelled calibrated). python3 model/score_4090_batch7b.py"""
import csv, glob, io, re, statistics as st, numpy as np
med = st.median; SM, CLK, CBREF, N = 128, 2.5e9, 1.01, 100000
def kern(f):
    t = open(f).read(); rows = list(csv.DictReader(io.StringIO(t[t.find('"ID"'):]))); return [float(r['Metric Value'].replace(',', '')) for r in rows if r['Metric Name'] == 'gpu__time_duration.sum' and r['Kernel Name'].startswith('optixLaunch')][0] / 1e6
cand = {'0.05': (93.8, 102.2, 0.191), '0.025': (13.0, 15.3, 0.058)}; out = {}
for r in ('0.05', '0.025'):
    kf = [kern(f) for f in sorted(glob.glob('notes/ncu4090/ncu_rtnn_full_%s_2026*.csv' % r))[-3:]]; ks = [kern(f) for f in sorted(glob.glob('notes/ncu4090/ncu_rtnn_stub_%s_2026*.csv' % r))]
    pl = lambda v, key: [float(re.search(key, open(f).read())[1]) for f in sorted(glob.glob('notes/rtnn_stub4090/plain_%s_%s_*.txt' % (v, r)))]
    sf, ss = pl('full', r'time search compute: (\S+)'), pl('stub', r'time search compute: (\S+)'); ef, es = pl('full', r'KERNEL queries \d+ ms (\S+)'), pl('stub', r'KERNEL queries \d+ ms (\S+)')
    C, q, p0 = cand[r]; Kf, Ks = med(kf), med(ks)
    cb = max((Kf - Ks) * 1e-3 * SM * CLK / (1.10 * C * N), 0); cq = 4.9 + cb - CBREF; pred = (110 + cq * q) * N / (SM * CLK) * 1e3
    print('r', r, 'ncu full', ['%.4f' % x for x in kf], 'stub', ['%.4f' % x for x in ks], '| plain search compute full %.4f stub %.4f, event full %.4f stub %.4f (n %d, %d)' % (med(sf), med(ss), med(ef), med(es), len(sf), len(ss)))
    print('   stub / full kernel %.2f, stub minus full %.4f ms, cb %.2f cycles per candidate (cb_ref 1.01), cq %.2f, calibrated prediction %.4f ms (uncalibrated %.3f, change %+.0f%%), kernel / calibrated %.2f, scored (search compute) / calibrated %.2f, event / calibrated %.2f' % (Ks / Kf, Kf - Ks, cb, cq, pred, p0, 100 * (pred / p0 - 1), Kf / pred, med(sf) / pred, med(ef) / pred))
    out[r] = dict(Kf=Kf, pred=pred, p0=p0, sc=med(sf), ev=med(ef))
# F255 rescore, 15 rows: (name, measured event basis, measured kernel basis, predicted event basis, predicted calibrated)
R = [3.90, 4.06, .58, .56, .56, .54, .55, .63, .53, .51, .48, .496, 1.11, 3.04, 1.07]  # F255 ratios as scored (event / search compute basis)
le = lambda L: (round(float(st.median([abs(np.log(x)) for x in L])), 2), sum(abs(np.log(x)) <= np.log(2) + 1e-9 for x in L))
print('\nF255, 15 rows: as scored', le(R))
K = list(R); K[0] = out['0.05']['Kf'] / out['0.05']['p0']; K[1] = out['0.025']['Kf'] / out['0.025']['p0']; print('RTNN on kernel time (F418)', le(K), '%.2f %.2f' % (K[0], K[1]))
# kNN rows on ncu kernel time (session 6 runs, rep 1 and 2 means), same predictions as F255
def load(f):
    t = open(f).read(); rows = list(csv.DictReader(io.StringIO(t[t.find('"ID"'):]))); d = {}
    for x in rows: d.setdefault(int(x['ID']), {'n': x['Kernel Name'][:30]})[x['Metric Name']] = float(x['Metric Value'].replace(',', ''))
    return [v['gpu__time_duration.sum'] / 1e6 for k, v in sorted(d.items()) if v['n'] == 'optixLaunch'][:4]
fo = [f for f in sorted(glob.glob('notes/ncu4090/ncu_knn_orig_2*.csv')) if 'smoke' not in f]; fr = [f for f in sorted(glob.glob('notes/ncu4090/ncu_knn_reg_2*.csv')) if 'smoke' not in f]
ko = [med([load(f)[i] for f in fo]) for i in range(4)]; kg = [med([load(f)[i] for f in fr]) for i in range(4)]
Po, Pg = [0.733, 1.086, 3.884, 12.22], [0.183, 0.250, 0.767, 3.26]
ratios = [a / b for a, b in zip(ko, Po)] + [sum(ko) / sum(Po)] + [a / b for a, b in zip(kg, Pg)] + [sum(kg) / sum(Pg)]
K2 = list(R); K2[2:7] = ratios[:5]; K2[7:12] = ratios[5:]; K2[12] = (sum(ko) / sum(kg)) / 4.02
print('kNN rows on ncu kernel time: orig', ['%.2f' % x for x in ratios[:5]], 'reg', ['%.2f' % x for x in ratios[5:]], 'speedup %.2f (kernel) over 4.02 = %.2f' % (sum(ko) / sum(kg), K2[12]))
print('Calibrated kernel basis, RTNN with the stub calibration, kNN and speedup on ncu kernel time, Arkade and TrueKNN unchanged:')
C2 = list(K2); C2[0] = out['0.05']['Kf'] / out['0.05']['pred']; C2[1] = out['0.025']['Kf'] / out['0.025']['pred']; print(le(C2), 'RTNN ratios %.2f %.2f' % (C2[0], C2[1]))
H = [C2[0], C2[1], C2[6], C2[11], C2[12], C2[13], C2[14]]; print('headline 7 rows (RTNN 2, kNN orig total, kNN reg total, speedup, Arkade, TrueKNN):', ['%.2f' % x for x in H], 'inside', sum(abs(np.log(x)) <= np.log(2) + 1e-9 for x in H), 'of 7')
H0 = [R[0], R[1], R[6], R[11], R[12], R[13], R[14]]; print('as scored headline', sum(abs(np.log(x)) <= np.log(2) + 1e-9 for x in H0), 'of 7; kernel basis uncalibrated', sum(abs(np.log(x)) <= np.log(2) + 1e-9 for x in [K2[0], K2[1], K2[6], K2[11], K2[12], K2[13], K2[14]]), 'of 7')
