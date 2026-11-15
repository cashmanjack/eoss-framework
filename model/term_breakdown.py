"""Per term error breakdown of the one model (model/reuse_feature.py, mode share with kappa 1) at every validation point that has ncu counts (notes/q11/ncu/ncu_*.json, ttu/bench/Morton/term_ncu.py).
Per point: the predicted speedup over the natural order and its parts (fixed cost, miss cost; shares of the natural order's time), and for each term that can be measured on OptiX the predicted and measured per ray count:
  L1 requests (S1, replay) vs RT L1 sectors (ncu: l1tex sectors minus the LSU pipe), L1 fills (F, replay) vs l1tex xbar to L1 read sectors minus 1 per ray (ray fetch), DRAM bytes (D) vs dram bytes read.
Queries per ray and triangles per ray have no OptiX counter (AD104 ncu has no ray counters): predicted only, listed. Writes notes/q11/term_breakdown.txt/json and three figures (notes/q11/fig_err_*.png).
    /usr/bin/python3 model/term_breakdown.py"""
import json, os, sys, numpy as np
sys.path.insert(0, 'model'); sys.path.insert(0, 'ttu/bench/Memory_4')
import reuse_feature as RF, performance_model as PM
PAR = json.load(open('notes/q11/reuse_feature.json'))['share_k1']['par']
BVH_MIB = {'s3e5': None, 's15e5': None, 'bistro': None, 'bounce100000': 6.5, 'bounce3000000': 186.0}
meta = {'s3e5': json.load(open('notes/q10/check/predict_s3e5.json')), 's15e5': json.load(open('notes/q10/check/predict_s15e5.json')), 'bistro': json.load(open('notes/q10/check/predict_bistro.json'))}
for t in meta: BVH_MIB[t] = meta[t]['bvh_mib']
NCU = {}
for n in RF.D_OF:
    p = 'notes/q11/ncu/ncu_%s.json' % n
    if os.path.exists(p): NCU[n] = json.load(open(p))


def terms(s, k):
    f = s['feats'][k]; A = PAR[0]
    fixed = A * (RF.p0['a_q'] * f['q'] + RF.p0['b_q'] * f['q'] / f['W'] + RF.p0['c_qW'] * f['q'] * f['W'] / 16.0)
    h = s['hist'][k]; miss = PAR[1] * f['S1'] * (1 - RF.share(h, PAR[2], PAR[3])) * (h[4] / h[0]) ** (-PAR[4])
    return fixed, miss


def nstat(name):
    l = open('notes/q10/check/reuse_%s_natural.txt' % name if name in meta else 'notes/io/predict/reuse_%s_natural.txt' % name.replace('bounce', '')).readline().split()
    return int(l[2]), int(l[4])      # nnodes, nleafslots


rows = []; L = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); L.append(s)
P('%-13s %-12s | %5s %5s %6s | %9s | %-27s | %-27s | %-27s' % ('set', 'order', 'pred', 'meas', 'logerr', 'miss share', 'L1 requests pred/ncu (ratio)', 'L1 fills pred/ncu (ratio)', 'DRAM bytes pred/ncu minus ray fetch'))
for n, s in RF.D_OF.items():
    nat_f, nat_m = terms(s, 'natural'); nn, nl = nstat(n); share_nat = s['hist']['natural'][4] / s['hist']['natural'][0]
    kids = (nn + nl - 1) / nn; depth = np.log(nl) / np.log(kids)
    for k in s['feats']:
        if k == 'natural': continue
        fx, ms = terms(s, k); pred = (nat_f + nat_m) / (fx + ms); meas = s['ms']['natural'] / s['ms'][k]
        r = dict(set=n, order=k, pred=pred, meas=meas, err=float(np.log(pred / meas)), miss_share_nat=nat_m / (nat_f + nat_m), fixed=fx, miss=ms, bvh_mib=BVH_MIB[n], sharing_nat=share_nat, depth=depth, q=s['feats'][k]['q'])
        if n in NCU and k in NCU[n] and 'natural' in NCU[n]:
            f = s['feats'][k]; m = NCU[n][k]; f0 = s['feats']['natural']; m0 = NCU[n]['natural']
            r.update(S1=f['S1'], S1_ncu=m['rt_l1_sectors'], F=f['F'], F_ncu=m['l1_fills'] - 1.0, D=f['D'], D_ncu=max(m['dram_bytes'] - 32.0, 0.0),
                     S1_ratio_pred=f['S1'] / f0['S1'], S1_ratio_meas=m['rt_l1_sectors'] / m0['rt_l1_sectors'], F_ratio_pred=f['F'] / f0['F'], F_ratio_meas=(m['l1_fills'] - 1) / (m0['l1_fills'] - 1),
                     ms_ncu_ratio=m0['ms'] / m['ms'])
        rows.append(r)
        t = ('%9.2f | %9.1f / %-6.1f (%.2f)  | %9.1f / %-6.1f (%.2f)  | %9.0f / %-6.0f' % (r['miss_share_nat'], r['S1'], r['S1_ncu'], r['S1'] / r['S1_ncu'], r['F'], r['F_ncu'], r['F'] / r['F_ncu'], r['D'], r['D_ncu'])) if 'S1_ncu' in r else '%9.2f | (no ncu count for this order)' % r['miss_share_nat']
        P('%-13s %-12s | %5.2f %5.2f %+6.3f | %s' % (n, k, pred, meas, r['err'], t))
P('\nWith ncu counts: per term error in log of predicted over measured (rms), absolute per ray and relative to the natural order:')
ncu_rows = [r for r in rows if 'S1_ncu' in r]
for nm, a, b in (('L1 requests (S1)', 'S1', 'S1_ncu'), ('L1 fills (F)', 'F', 'F_ncu')):
    P('  %-18s absolute rms log error %.3f, mean signed %+.3f  (%d points); natural order points only: %s' % (nm, np.sqrt(np.mean([np.log(r[a] / r[b]) ** 2 for r in ncu_rows])), np.mean([np.log(r[a] / r[b]) for r in ncu_rows]), len(ncu_rows),
        ', '.join('%s %+.2f' % (n, np.log(RF.D_OF[n]['feats']['natural'][a] / NCU[n]['natural']['rt_l1_sectors' if a == 'S1' else 'l1_fills'] if a == 'S1' else RF.D_OF[n]['feats']['natural'][a] / (NCU[n]['natural']['l1_fills'] - 1))) for n in NCU)))
for nm, a, b in (('L1 requests ratio', 'S1_ratio_pred', 'S1_ratio_meas'), ('L1 fills ratio', 'F_ratio_pred', 'F_ratio_meas')):
    P('  %-18s change vs natural, rms log error %.3f, mean signed %+.3f' % (nm, np.sqrt(np.mean([np.log(r[a] / r[b]) ** 2 for r in ncu_rows])), np.mean([np.log(r[a] / r[b]) for r in ncu_rows])))
P('  speedup from ncu kernel times vs the event times used for the fit: rms log difference %.3f' % np.sqrt(np.mean([np.log(r['ms_ncu_ratio'] / r['meas']) ** 2 for r in ncu_rows])))
P('\nTotal error against scene size, ray coherence and BVH depth (30 points, signed mean log error of the predicted speedup; positive = over predicted):')
def grp(key, bins, label):
    for lo, hi in bins:
        sel = [r for r in rows if lo <= r[key] < hi]
        if sel: P('  %-26s %6.2f to %-6.2f  %2d points  mean %+.3f  rms %.3f  (sets: %s)' % (label, lo, hi, len(sel), np.mean([r['err'] for r in sel]), np.sqrt(np.mean([r['err'] ** 2 for r in sel])), ','.join(sorted({r['set'] for r in sel}))))
grp('bvh_mib', [(0, 10), (10, 100), (100, 1e9)], 'BVH size MiB'); grp('sharing_nat', [(1, 1.3), (1.3, 1.7), (1.7, 3)], 'natural lane sharing ratio'); grp('depth', [(0, 5), (5, 6), (6, 9)], 'estimated BVH depth')
P('  by set: ' + '; '.join('%s %.2f MiB depth %.1f sharing %.2f -> mean %+.3f rms %.3f' % (n, BVH_MIB[n], [r for r in rows if r['set'] == n][0]['depth'], [r for r in rows if r['set'] == n][0]['sharing_nat'], np.mean([r['err'] for r in rows if r['set'] == n]), np.sqrt(np.mean([r['err'] ** 2 for r in rows if r['set'] == n]))) for n in RF.D_OF))
P('  by order: ' + '; '.join('%s mean %+.3f' % (k, np.mean([r['err'] for r in rows if r['order'] == k])) for k in dict.fromkeys(r['order'] for r in rows)))
P('  correlation of the error with measured speedup (log): %.2f; with log BVH size %.2f; with sharing ratio %.2f; with estimated depth %.2f' % tuple(np.corrcoef([r['err'] for r in rows], x)[0, 1] for x in ([np.log(r['meas']) for r in rows], [np.log(r['bvh_mib']) for r in rows], [r['sharing_nat'] for r in rows], [r['depth'] for r in rows])))
P('Queries per ray (replay, predicted only; no OptiX counter): ' + ', '.join('%s %.2f' % (n, RF.D_OF[n]['feats']['natural']['q']) for n in RF.D_OF) + '. Triangles per ray: not in the model as a separate term (leaf slots are in the node fetch stream), no counter: not compared.')
open('notes/q11/term_breakdown.txt', 'w').write('\n'.join(L) + '\n'); json.dump(rows, open('notes/q11/term_breakdown.json', 'w'), indent=1)
try:
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    col = {'s3e5': '#1f77b4', 's15e5': '#ff7f0e', 'bistro': '#2ca02c', 'bounce100000': '#d62728', 'bounce3000000': '#9467bd'}
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for a, (key, lab, lg) in zip(ax, (('bvh_mib', 'BVH size (MiB)', True), ('sharing_nat', 'lane sharing ratio, natural order (raw / merged fetches)', False), ('depth', 'estimated BVH depth', False))):
        for n in RF.D_OF:
            sel = [r for r in rows if r['set'] == n]; a.scatter([r[key] for r in sel], [r['err'] for r in sel], c=col[n], label=n, s=22)
        a.axhline(0, c='k', lw=0.6); a.set_xlabel(lab); a.set_xscale('log' if lg else 'linear')
    ax[0].set_ylabel('log(predicted / measured speedup)'); ax[0].legend(fontsize=7); fig.tight_layout(); fig.savefig('notes/q11/fig_err_scene.png', dpi=130)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
    for n in NCU:
        sel = [r for r in ncu_rows if r['set'] == n]
        ax[0].scatter([r['S1_ratio_meas'] for r in sel], [r['S1_ratio_pred'] for r in sel], c=col[n], label=n, s=22); ax[1].scatter([r['F_ratio_meas'] for r in sel], [r['F_ratio_pred'] for r in sel], c=col[n], s=22)
    for a, t in zip(ax, ('L1 requests per ray, order / natural', 'L1 fills per ray, order / natural')):
        a.plot([0.1, 2], [0.1, 2], 'k', lw=0.6); a.set_xlabel('measured (ncu), ' + t); a.set_ylabel('predicted (replay)'); a.set_xscale('log'); a.set_yscale('log')
    ax[0].legend(fontsize=7); fig.tight_layout(); fig.savefig('notes/q11/fig_err_terms.png', dpi=130)
except Exception as e: print('figures not made:', e)
