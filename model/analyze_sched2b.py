"""Sched_2 redone: results against model/predictions_sched2.md. /usr/bin/python3 model/analyze_sched2b.py notes/sched2   (reads s2b_<arm>_r<rep>_<ts>.json)"""
import glob, json, os, statistics as st, sys
D = sys.argv[1]
ARMS = {'alone': (1, 0), 'lo4': (4, 0), 'lo5': (5, 0), 'lo9': (9, 0), 'lo13': (13, 0), 'lo29': (29, 0), 'hi5': (5, 4), 'hi13': (13, 12), 'hi29': (29, 28)}
PRED = {'lo4': ('abs <= 40', lambda d: abs(d) <= 40), 'lo5': ('10 to 100', lambda d: 10 <= d <= 100), 'lo9': ('20 to 200', lambda d: 20 <= d <= 200), 'lo13': ('> 5,000 (P3) / < 500 (alternative)', lambda d: d > 5000),
        'lo29': ('> 20,000 (P3)', lambda d: d > 20000), 'hi5': ('<= 100', lambda d: d <= 100), 'hi13': ('<= 100', lambda d: d <= 100), 'hi29': ('<= 100', lambda d: d <= 100)}
res = {}
for a, (W, T) in ARMS.items():
    L = []; ast = []; aen = []; med_run = []
    for f in sorted(glob.glob(os.path.join(D, 's2b_%s_r*_*.json' % a))):
        j = json.load(open(f)); lat = []
        for l in j['launches'][1:]:
            wc = l['warp_clk']; lat.append(wc[T][1] - wc[T][0]); s0 = wc[T][0]
            for i, w in enumerate(wc):
                if i != T: ast.append(w[0] - s0); aen.append(w[1] - s0)
        L += lat; med_run.append(st.median(lat))
    res[a] = dict(med=st.median(med_run), runs=med_run, p10=sorted(L)[len(L) // 10], p90=sorted(L)[9 * len(L) // 10], ast=st.median(ast) if ast else None, aen_min=min(aen) if aen else None, aen=st.median(aen) if aen else None, n=len(L))
base = res['alone']['med']
print('%-6s %3s %3s %8s %7s %7s %7s | %6s %6s %8s | %s' % ('arm', 'W', 'T', 'median', 'p10', 'p90', 'Delta', 'ALU st', 'min en', 'med en', 'runs medians'))
for a, (W, T) in ARMS.items():
    r = res[a]; d = r['med'] - base
    print('%-6s %3d %3d %8.0f %7d %7d %7.0f | %6s %6s %8s | %s' % (a, W, T, r['med'], r['p10'], r['p90'], d, r['ast'], r['aen_min'], r['aen'], [int(x) for x in r['runs']]))
print('\nscoring against model/predictions_sched2.md')
for a, (txt, f) in PRED.items():
    d = res[a]['med'] - base; print('  %-5s Delta %+6.0f   predicted %-38s %s' % (a, d, txt, 'HELD' if f(d) else 'MISSED'))
print('  lo13 alternative (Delta < 500, a returning warp escapes starvation): %s;  all ALU windows overlap the whole TTU window: %s' % ('HELD' if res['lo13']['med'] - base < 500 else 'MISSED', all(res[a]['aen_min'] > res[a]['med'] + 480 for a in ARMS if a != 'alone')))
print('  Delta per ALU warp on the TTU warp scheduler (a): lo9 %.0f (a 2), lo13 %.0f (a 3), lo29 %.0f (a 7); hi13 %.0f (a 3), hi29 %.0f (a 7)' % tuple((res[k]['med'] - base) / n for k, n in (('lo9', 2), ('lo13', 3), ('lo29', 7), ('hi13', 3), ('hi29', 7))))
