"""HG_1 analysis (reads only the row_*.json files written by ttu/bench/HG_1/run.py; CPU only).

    hg1_analyze.py ROWDIR

Per scene (the 8 scenes of model/predictions_hg1.md): group only (harness --clock, one query), harness loop span (--lclock, whole ray),
OptiX trace of the first and the repeated calls; the difference D = OptiX trace - harness loop and the least squares fit
D = a + b x group (b: does the OptiX excess grow with the cost of the query itself, i.e. with nodes) and D = a + b x rounds.
"""
import glob, json, os, sys
import numpy as np

SC = ['tri1', 'grid12', 'grid144', 'grid200', 'grid2500', 'grid4096', 'grid20736', 'soup']


def latest(d, tag):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, 'row_clk_*.json' if tag == 'clk' else 'row_*_%s_*.json' % tag))):
        r = json.load(open(f)); out[r['scene']] = r
    return out


def main(d):
    clk = latest(d, 'clk'); warm = latest(d, 'warm'); n1 = latest(d, 'nrep1'); r3 = latest(d, 'rep3'); rot = latest(d, 'rot3'); cold = latest(d, 'cold')
    print('%-10s %6s %7s %6s | %8s %8s %8s %8s | %8s %8s | %7s %7s' % ('scene', 'group', 'harness', 'loop+', 'OXwarm', 'OXnrep1', 'OXrep3.0', 'OXrot3.0', 'OX 2nd', 'OX 3rd', 'D', 'OX-grp'))
    rows = []
    for s in SC:
        g = clk[s]['group']; h = n1[s]['harness']
        ox = [warm[s]['trace'], n1[s]['trace'], r3[s]['trace'], rot[s]['trace'] if s in rot else float('nan')]
        t2, t3 = r3[s]['trace1'], r3[s]['trace2']
        D = float(np.median([x for x in ox if x == x] )) - h
        print('%-10s %6.0f %7.0f %6.0f | %8.0f %8.0f %8.0f %8.0f | %8.0f %8.0f | %7.0f %7.0f' % (s, g, h, h - g, *ox, t2, t3, D, np.median([x for x in ox if x == x]) - g))
        rows.append((s, g, h, np.median([x for x in ox if x == x]), t2, n1[s]['rounds_mean'], n1[s]['rounds_max']))
    one = [r for r in rows if r[0] != 'soup']
    g = np.array([r[1] for r in one]); D = np.array([r[3] - r[2] for r in one]); X = np.vstack([np.ones_like(g), g]).T
    (a, b), *_ = np.linalg.lstsq(X, D, rcond=None)
    print('\nOne round scenes (tri1 and the 6 grids): D = %.0f + %.4f x group;  mean D %.0f, min %.0f, max %.0f' % (a, b, D.mean(), D.min(), D.max()))
    og = np.array([r[3] for r in one]); (a2, b2), *_ = np.linalg.lstsq(X, og, rcond=None)
    print('OptiX trace = %.0f + %.4f x group (harness group alone on the same scenes: slope 1 would be equal node costs)' % (a2, b2))
    sp = [r for r in rows if r[0] == 'soup'][0]
    print('soup: D = %.0f (rounds mean %.2f, max %d); one round mean D %.0f' % (sp[3] - sp[2], sp[5], sp[6], D.mean()))
    print('\nSecond and third call of OptiX in the same raygen (rep3), against the harness warm second session (782, F176, same GPU):')
    for s in SC:
        print('  %-10s trace 2 %6.0f  trace 3 %6.0f  (trace 1 %6.0f)  trace 2 - 782 = %6.0f   first call extra (trace 1 - trace 2) %6.0f' % (s, r3[s]['trace1'], r3[s]['trace2'], r3[s]['trace'], r3[s]['trace1'] - 782, r3[s]['trace'] - r3[s]['trace1']))
    if cold:
        for s, c in cold.items(): print('cold L2 (256 MB flush) %s: harness %.0f OptiX trace %.0f gap %.2f' % (s, c['harness'], c['trace'], c['gap']))


if __name__ == '__main__':
    main(sys.argv[1])
