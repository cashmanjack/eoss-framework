"""Div_1: summarize a lane histogram dump. lh_summary.py FILE [NLANES_ACTIVE_WARPS_NOTE]. FILE = raw u64 array, LH_STRIDE 136 per warp. Prints per workload:
callbacks per lane count distribution (mean lanes, share of callbacks with <= 8 lanes), handoffs (first callback lanes), callback cycles and interval cycles by lane count, share of warp time (raygen start to end summed over warps)
attributed to callbacks with <= 8 lanes, and the unattributed rest."""
import sys
import numpy as np
S = 136
def load(f):
    a = np.fromfile(f, np.uint64); return a.reshape(-1, S).astype(np.float64)
def summ(a, label=''):
    a = a[a[:, 135] > 0]                     # warps with at least one callback
    allw = a
    cb = allw[:, 1:33].sum(0); cy = allw[:, 34:66].sum(0); iv = allw[:, 67:99].sum(0); ho = allw[:, 100:132].sum(0)
    lanes = np.arange(1, 33); wt = (allw[:, 134] - allw[:, 133]).sum()
    out = {}
    out['warps_with_callbacks'] = len(allw); out['callbacks'] = cb.sum(); out['callbacks_per_warp'] = cb.sum() / len(allw)
    out['mean_lanes_per_callback'] = (cb * lanes).sum() / cb.sum(); out['lane_eff_callbacks'] = out['mean_lanes_per_callback'] / 32
    out['share_callbacks_le8'] = cb[:8].sum() / cb.sum(); out['share_callbacks_32'] = cb[31] / cb.sum()
    out['handoffs'] = ho.sum(); out['mean_lanes_per_handoff'] = (ho * lanes).sum() / ho.sum(); out['share_handoffs_le8'] = ho[:8].sum() / ho.sum(); out['callbacks_per_handoff'] = cb.sum() / ho.sum()
    out['warp_time_cycles'] = wt; out['callback_cycles_share_of_warp_time'] = cy.sum() / wt; out['interval_share_of_warp_time'] = iv.sum() / wt
    out['warp_time_le8_lanes'] = iv[:8].sum() / wt; out['warp_time_le16_lanes'] = iv[:16].sum() / wt; out['callback_cycles_le8_share'] = cy[:8].sum() / cy.sum()
    out['mean_lanes_time_weighted'] = (iv * lanes).sum() / iv.sum()
    return out, cb, cy, iv, ho
if __name__ == '__main__':
    o, cb, cy, iv, ho = summ(load(sys.argv[1]))
    for k, v in o.items(): print('%-36s %s' % (k, ('%.4g' % v)))
    print('lanes: share of callbacks', ' '.join('%d:%.3f' % (p + 1, cb[p] / cb.sum()) for p in range(32) if cb[p] / cb.sum() > 0.005))
