"""Div_1 summary: model/div_summ.py notes/div1/raw. Groups raw lane histogram dumps by workload and run (probe mode), summarizes per file and per workload across repeats.
Slot layout: ttu/bench/Div_1/lanehist.h (LH_STRIDE 136 u64 per warp). kNN dumps hold intersection slots then any hit slots (2 x warps); everything else one function.
Definitions: callbacks = callbacks entered by a warp with p active lanes; handoff = a callback that starts at least 300 cycles after the previous exit of its warp (or the first); warp time = raygen start to raygen end summed over ALL warps that started;
time attributed to few lanes = interval cycles (previous callback exit or raygen start to this callback exit) of callbacks with 8 or fewer lanes, over warp time."""
import glob, os, re, sys
import numpy as np
S = 136
def load(f, part=None):
    a = np.fromfile(f, np.uint64).reshape(-1, S).astype(np.float64)
    if part is not None: h = len(a) // 2; a = a[:h] if part == 0 else a[h:]
    return a
def summ(a, wt_override=None):
    started = a[a[:, 133] > 0]; wt = np.maximum(started[:, 134] - started[:, 133], 0).sum() if wt_override is None else wt_override
    w = a[a[:, 135] > 0]; lanes = np.arange(1, 33)
    cb = w[:, 1:33].sum(0); cy = w[:, 34:66].sum(0); iv = w[:, 67:99].sum(0); ho = w[:, 100:132].sum(0)
    o = dict(warps=len(started), warps_cb=len(w), callbacks=cb.sum(), cb_per_warp=cb.sum() / max(len(w), 1), mean_lanes=(cb * lanes).sum() / cb.sum(), cb_le8=cb[:8].sum() / cb.sum(), cb_32=cb[31] / cb.sum(),
             handoffs=ho.sum(), ho_lanes=(ho * lanes).sum() / ho.sum(), ho_le8=ho[:8].sum() / ho.sum(), cb_per_ho=cb.sum() / ho.sum(), cb_cyc_share=cy.sum() / wt, iv_share=iv.sum() / wt,
             t_le8=iv[:8].sum() / wt, t_le16=iv[:16].sum() / wt, tw_lanes=(iv * lanes).sum() / iv.sum(), warp_cycles=wt)
    return o, cb, iv
KEYS = ['warps', 'warps_cb', 'callbacks', 'cb_per_warp', 'mean_lanes', 'cb_le8', 'cb_32', 'handoffs', 'ho_lanes', 'ho_le8', 'cb_per_ho', 'cb_cyc_share', 'iv_share', 't_le8', 't_le16', 'tw_lanes']
def row(name, o): return '%-34s ' % name + ' '.join('%s %.4g' % (k, o[k]) for k in KEYS)
if __name__ == '__main__':
    d = sys.argv[1]; files = sorted(glob.glob(os.path.join(d, '*_probe_*.bin'))); runs = {}
    for f in files:
        m = re.match(r'(\w+?)_probe_(\d+T\d+)_(\d)(?:_(.*))?\.bin', os.path.basename(f)); wl, ts, i, rest = m[1], m[2], m[3], m[4] or ''
        runs.setdefault((wl, rest), {}).setdefault(ts + i, []).append(f)
    for (wl, rest), reps in sorted(runs.items()):
        print('==', wl, rest)
        per = {}
        for i, fl in sorted(reps.items()):
            for fidx, parts in ((0, [None]),):
                pass
            if wl == 'knn':
                for part, nm in ((0, 'intersection'), (1, 'anyhit')):
                    tot = np.concatenate([load(f, part) for f in fl]); o, cb, iv = summ(tot, None if part == 0 else np.concatenate([load(f, 0) for f in fl])[:, 134:135].sum() * 0 + np.maximum(np.concatenate([load(f, 0) for f in fl])[:, 134] - np.concatenate([load(f, 0) for f in fl])[:, 133], 0).sum()); per.setdefault(nm, []).append(o)
            else:
                tot = np.concatenate([load(f) for f in fl]); o, cb, iv = summ(tot); per.setdefault('callbacks', []).append(o)
        for nm, L in per.items():
            for k in ('mean_lanes', 'cb_le8', 'ho_lanes', 't_le8', 't_le16', 'cb_cyc_share', 'iv_share'):
                v = [o[k] for o in L]; print('  %-12s %-13s median %.4g  min %.4g max %.4g  spread %.1f%%' % (nm, k, np.median(v), min(v), max(v), 100 * (max(v) - min(v)) / abs(np.median(v)) if np.median(v) else 0))
            print('  %-12s callbacks %.4g handoffs %.4g callbacks/handoff %.3g warps %d (with callbacks %d) cb_per_warp %.3g' % (nm, L[0]['callbacks'], L[0]['handoffs'], L[0]['cb_per_ho'], L[0]['warps'], L[0]['warps_cb'], L[0]['cb_per_warp']))
