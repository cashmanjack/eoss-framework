"""Part C 3 of model/predictions_4090_batch7.md, 5090 side, after the 5090's numbers arrived (pulled 49b3f58, model/results_5090_batch7.md, F420 and F421 of the 5090 session). Inputs are copied from that file. python3 model/score_4090_batch7d.py
Method as committed: T = R c_r + W c_w, speedup = T(original) / T(register heap). Costs: 5090 stream (coalesced 512 B per warp, 24 MB): read 82.7, write 41.3 sectors per cycle; 4090: the coalesced pattern of l2rw (3.92 and 2.58 TB/s) is the like for like pattern, lane4 (5.05 and 2.01) is the committed primary but exists only on the 4090.
DISCLOSURE: the 5090 measured speedups (3.0 to 4.0 per round, 3.17 total) were seen before this scoring; the 5090 sector mix was not measured when the predictions were written (part C said: use the 4090 write shares), so both versions are given."""
import numpy as np
M = {'orig': [18.0, 27.1, 94.6, 298.8], 'reg': [6.2, 8.5, 25.9, 110.0]}  # M sectors, 5090
W = {'orig': [.749, .773, .830, .780], 'reg': [.283, .299, .399, .392]}  # 5090 measured write shares
W4 = {'orig': [.68, .70, .75, .71], 'reg': [.27, .29, .39, .38]}  # 4090 write shares (F419)
meas = [2.99, 3.34, 4.00, 2.98]; meas_tot = 3.17
def sp(cr, cw, w):
    To = [m * ((1 - x) * cr + x * cw) for m, x in zip(M['orig'], w['orig'])]; Tg = [m * ((1 - x) * cr + x * cw) for m, x in zip(M['reg'], w['reg'])]
    return [a / b for a, b in zip(To, Tg)], sum(To) / sum(Tg)
cr, cw = 1 / 82.7, 1 / 41.3
print('5090 with its own stream costs (write 2.00 reads):')
for name, w in (('5090 write shares (measured, F420 of the 5090 session)', W), ('4090 write shares (the committed prediction input)', W4)):
    s, t = sp(cr, cw, w); print(' %-55s' % name, ' '.join('%.2f' % x for x in s), 'total %.2f' % t, '| error vs measured', ' '.join('%+.0f%%' % (100 * (a / b - 1)) for a, b in zip(s, meas)), 'total %+.0f%%' % (100 * (t / meas_tot - 1)))
s, t = sp(1, 1, W); print(' byte ratio only (5090 sectors)', ' '.join('%.2f' % x for x in s), 'total %.2f' % t, '| error', ' '.join('%+.0f%%' % (100 * (a / b - 1)) for a, b in zip(s, meas)), 'total %+.0f%%' % (100 * (t / meas_tot - 1)))
print('\ncross chip: coalesced write over read cost 4090 %.2f (l2rw coal, 32 MB, time per sector) against 5090 %.2f (stream, sectors per cycle ratio 82.7 / 41.3); the 4090 per lane sector pattern %.2f' % (0.01240 / 0.00816, 82.7 / 41.3, 0.01591 / 0.00633))
print('4090 coal linear total prediction 3.44 (measured 4.20 to 4.48) and 5090 coal linear total above: the coalesced costs predict almost the same speedup on both chips, the measured differ by 1.3 to 1.4 times.')
