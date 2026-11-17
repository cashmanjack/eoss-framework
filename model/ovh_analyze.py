"""Ovh_1 analysis (predictions: model/predictions_ovh.md). usage: ovh_analyze.py TS WORK  (reads WORK/{tri1,soup}_{empty,trace}_TS.json)"""
import json, sys, numpy as np
TS, W = sys.argv[1], sys.argv[2]; NSM = 128
def load(scene, tag):
    j = json.load(open('%s/%s_%s_%s.json' % (W, scene, tag, TS))); out = {}
    for r in j['runs']:
        c = np.array(r['cal'], dtype=np.float64).reshape(-1, 4); nw = r['ncalw']
        dc, dg = c[:, 1] - c[:, 0], c[:, 3] - c[:, 2]
        ghz = float(dc.sum() / dg.sum()) if dg.sum() > 0 else float('nan')   # cycles per ns over all warps (globaltimer is too coarse for the EMPTY raygen: borrowed from TRACE below)
        out[(r['nrep'], r['n'])] = dict(ms=float(np.median(r['ms'])), ghz=ghz, wl_med=float(np.median(dc)), wl_p90=float(np.percentile(dc, 90)), wl_max=float(dc.max()), N=r['n'])
    return out
for scene in ('tri1', 'soup'):
    E, T = load(scene, 'empty'), load(scene, 'trace')
    for (nr, N), r in E.items(): r['ghz'] = T[(1, N)]['ghz']
    print('\n== %s   (c = cycles per ray per SM, f = measured SM GHz, wl = warp latency cycles median / p90)' % scene)
    print('%9s %6s | %8s %6s | %8s %6s %7s | %8s | %8s | %8s %8s | %s' % ('N', 'w/SM', 'empty c', 'wl', 'trace1 c', 'f', 'wl', 'trace3 c', 'trace0 c', 'd1', 'd3', 'wl1 / wl3 / wl0 / wlE (median)'))
    for N in sorted({k[1] for k in T}):
        e, t0, t1, t3 = E[(1, N)], T[(0, N)], T[(1, N)], T[(3, N)]
        c = lambda r: r['ms'] * 1e6 * 0 + r['ms'] * 1e-3 * r['ghz'] * 1e9 * NSM / N   # ms -> s, times cycles/s, SMs, per ray
        cc = lambda r: (r['ms'] * 1e6) * r['ghz'] * NSM / N   # ns * (cycles/ns) * SMs / N
        print('   ns per ray N=%d: empty %.4f trace0 %.4f trace1 %.4f trace3 %.4f' % (N, e['ms'] * 1e6 / N, t0['ms'] * 1e6 / N, t1['ms'] * 1e6 / N, t3['ms'] * 1e6 / N)) if N >= 262144 else None
        d1, d3 = cc(t1) - cc(e), (cc(t3) - cc(t1)) / 2
        print('%9d %6.2f | %8.2f %6.0f | %8.2f %6.2f %7.0f | %8.2f | %8.2f | %8.2f %8.2f | %.0f / %.0f / %.0f / %.0f' % (N, N / 32 / NSM, cc(e), e['wl_med'], cc(t1), t1['ghz'], t1['wl_med'], cc(t3), cc(t0), d1, d3, t1['wl_med'], t3['wl_med'], t0['wl_med'], e['wl_med']))
    print('  per call warp latency, one more warm call (wl3 - wl1)/2 at each size:', {N: round((T[(3, N)]['wl_med'] - T[(1, N)]['wl_med']) / 2) for N in sorted({k[1] for k in T})})
