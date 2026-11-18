"""Phase 10i test 2: far / near / identical, cycles per ray per SM (c), x(D) = (c_near - c_floor) / (c_far - c_floor). CPU only.
/usr/bin/python3 model/phase10i_knee.py DIR [CLKCSV]   (latest DIR/n_summary_*.json)"""
import glob, json, os, statistics as st, sys
d = sys.argv[1]; res = json.load(open(sorted(glob.glob(os.path.join(d, 'n_summary_*.json')))[-1]))
c = {}
for r in res: c.setdefault((r['size'], r['rays']), []).append(r['c'])
M = {k: st.median(v) for k, v in c.items()}; R = {k: (min(v), max(v)) for k, v in c.items()}
for n in (100000, 1000000):
    fl = M[(n, 'cls32')]; print('size %d: floor (identical) %.1f' % (n, fl))
    print('  D   far c (range)         near c (range)        x')
    for D in (2, 4, 8, 16, 32):
        f, ne = M[(n, 'cls%d' % (32 // D))], M[(n, 'near%d' % D)]
        print('  %2d  %6.1f (%5.1f-%5.1f)  %6.1f (%5.1f-%5.1f)  %s' % (D, f, *R[(n, 'cls%d' % (32 // D))], ne, *R[(n, 'near%d' % D)], ('%.2f' % ((ne - fl) / (f - fl))) if f - fl > 0.1 * fl else 'far within 10%% of floor'))
x = {n: {D: (M[(n, 'near%d' % D)] - M[(n, 'cls32')]) / (M[(n, 'cls%d' % (32 // D))] - M[(n, 'cls32')]) for D in (16, 32)} for n in (100000, 1000000)}
verdict = 'H_rays' if all(v >= 0.75 for a in x.values() for v in a.values()) else 'H_lines' if all(v <= 0.25 for a in x.values() for v in a.values()) else 'mixed'
print('x at D 16 / 32:', {n: {D: round(v, 2) for D, v in a.items()} for n, a in x.items()}, '->', verdict)
