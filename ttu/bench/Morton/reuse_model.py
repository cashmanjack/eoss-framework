"""LRU model of the node cache fed with the reuse distance histograms of reuse_run.py: does it predict the measured grouping speedups?
    reuse_model.py WORK TIMING.json        (TIMING = notes/q3/depth4_wset_23.json: measured speedups over the random order, same ray set; 10 launches median, 0 wrong)
Prediction without any fit (P0): time proportional to misses, speedup(key) = miss(random) / miss(key), miss = (cold + distance >= C) / accesses, cache = fully associative LRU of C lines of 128 B.
P1 (one fitted constant f, the share of the random order time that does not depend on misses): speedup = 1 / (f + (1 - f) miss(key) / miss(random)); f fit on one scene, tested on the other.
Reported for C = 60 KB (480 lines, main) and a sweep of C; units SM and TPC; variants nodes only / nodes and leaf slots; dedupe / raw.
"""
import json, os, sys
import numpy as np
KEYS = ('random', 'd1', 'd2', 'd3', 'd4', 'full')


def load(work, scene, key):
    rows = {}
    for l in open(os.path.join(work, 'reuse_%s_%s.txt' % (scene, key))):
        if l[0] == '#': continue
        p = l.split(); tot, cold, over = map(int, p[3:6]); rows[(p[0], p[1], p[2])] = (tot, cold, over, np.array(p[6:], np.int64))
    return rows


def miss(row, C):
    tot, cold, over, h = row
    return (cold + over + h[C:].sum()) / tot


def main(work, timing):
    T = json.load(open(timing)); res = {}
    H = {s: {k: load(work, s, k) for k in KEYS} for s in ('1e5', '3e6')}
    meas = {s: np.array([T[s]['timing'][k]['speedup'] for k in KEYS]) for s in H}
    out = []
    def line(*a): s = ' '.join(str(x) for x in a); print(s); out.append(s)
    line('measured speedup over random (keys %s):' % ','.join(KEYS))
    for s in H: line('  ', s, ' '.join('%.2f' % v for v in meas[s]))
    for unit in ('sm', 'tpc'):
        for variant in ('nodes', 'nodes+leaf'):
            for dd in ('dedupe', 'raw'):
                cfg = (unit, variant, dd)
                line('\n== unit %s, %s, %s ==' % cfg)
                lines_main = 480 if unit == 'sm' else 960      # 60 KB per SM, 120 KB per TPC, 128 B lines
                m = {s: {C: np.array([miss(H[s][k][cfg], C) for k in KEYS]) for C in (lines_main,)} for s in H}
                for s in H:
                    mm = m[s][lines_main]
                    p0 = mm[0] / mm
                    line('  %s miss rate (C %d lines = %d KB): %s' % (s, lines_main, lines_main * 128 // 1024, ' '.join('%.3f' % v for v in mm)))
                    line('  %s cold %s' % (s, ' '.join('%.3f' % (H[s][k][cfg][1] / H[s][k][cfg][0]) for k in KEYS)))
                    line('  %s P0 predicted speedup %s | measured %s | ratio pred/meas %s' % (s, ' '.join('%.2f' % v for v in p0), ' '.join('%.2f' % v for v in meas[s]), ' '.join('%.2f' % v for v in p0 / meas[s])))
                # P1: fit f on one scene, test on the other
                def p1(s, f): mm = m[s][lines_main]; return 1 / (f + (1 - f) * mm / mm[0])
                fs = np.linspace(0, 0.9, 901)
                for tr, te in (('1e5', '3e6'), ('3e6', '1e5')):
                    err = [np.sqrt(np.mean(np.log(p1(tr, f) / meas[tr]) ** 2)) for f in fs]; f = fs[int(np.argmin(err))]
                    line('  P1 f fit on %s = %.3f (rms log error there %.3f); applied to %s: %s | measured %s | rms log error %.3f' % (tr, f, min(err), te, ' '.join('%.2f' % v for v in p1(te, f)), ' '.join('%.2f' % v for v in meas[te]), np.sqrt(np.mean(np.log(p1(te, f) / meas[te]) ** 2))))
                # which capacity C best matches P0 (rms log error over both scenes) and the rank correlation at C main
                best = None
                for C in (60, 120, 240, 480, 720, 960, 1440, 1920, 2880, 3840, 5760, 8000):
                    e = 0
                    for s in H:
                        mm = np.array([miss(H[s][k][cfg], C) for k in KEYS]); e += np.sum(np.log((mm[0] / mm) / meas[s]) ** 2)
                    e = np.sqrt(e / (2 * len(KEYS)))
                    line('    C %5d lines (%4d KB): P0 rms log error over both scenes %.3f' % (C, C * 128 // 1024, e))
    open(os.path.join(work, 'reuse_model.txt'), 'w').write('\n'.join(out) + '\n')


if __name__ == '__main__': main(*sys.argv[1:3])
