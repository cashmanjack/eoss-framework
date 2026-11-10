"""Throughput_2 analysis: per run, median over launches 1..30 of the block span (first warp start to
last warp end) and of the per warp latency; active rays per cycle and warp queries per cycle.
Correctness: lanes of warp 0 that queried must return the same first query result as in the L = 32
run of the same scene and W; tmax0 idle lanes must miss.

    analyze.py WORK
"""
import json, os, statistics as st, sys

work = sys.argv[1]
KEYS = (3, 32, 33, 44, 45)   # hit flag, barycentrics, prim word, t (the leaf address word differs per process)


def load(p):
    j = json.load(open(p)); L = j['launches'][1:]
    span = st.median(max(e for _, e in l['warp_clk']) - min(s for s, _ in l['warp_clk']) for l in L)
    lat = st.median(st.median(e - s for s, e in l['warp_clk']) for l in L)
    same = all(l['same_as_first'] for l in j['launches'])
    return j, span, lat, same


print('%-12s %3s %3s %-5s | %8s %8s | %10s %10s | %s' % ('scene', 'W', 'L', 'idle', 'span', 'lat/warp', 'rays/cyc', 'warpq/cyc', 'checks'))
for scene in ('tri1', 'soup'):
    for w in (1, 4, 16):
        ref = None
        for L in (32, 16, 8, 1):
            for idle in (('mask',) if L == 32 else ('mask', 'tmax0', 'tneg', 'empty')):
                p = os.path.join(work, '%s_w%d_L%d_%s.json' % (scene, w, L, idle))
                j, span, lat, same = load(p)
                o = j['out']
                if L == 32:
                    ref = o
                bad = [l for l in range(32) if l < L and any(o[l][k] != ref[l][k] for k in KEYS)]
                if idle == 'mask':
                    bad += [l for l in range(L, 32) if o[l][12] != 0x77777777]     # masked lanes must not store
                else:
                    bad += [l for l in range(L, 32) if o[l][3] != 0]               # idle lanes with an empty interval must miss
                print('%-12s %3d %3d %-5s | %8.0f %8.0f | %10.4f %10.5f | %s%s' % (
                    scene, w, L, idle, span, lat, w * L / span, w / span, 'ok' if not bad else 'BAD lanes %s' % bad[:6],
                    '' if same else ' (results changed between launches)'))
