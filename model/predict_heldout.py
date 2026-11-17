"""Held out scene predictions with the F123 'share' form (kappa = 1), NO refit (2026-11-17, tethys).
    /usr/bin/python3 model/predict_heldout.py TAG [TAG ...]      (reads notes/hs/predict_TAG.json and notes/hs/reuse_TAG_*.txt, written by ttu/bench/Morton/random_check.py predict)
Parameters: notes/q11/reuse_feature.json 'share_k1' (fitted on soup 300,000 random rays and bounce rays on the 100,000 soup only, on 4070 Ti OptiX data). Inputs per order: the CPU replay of
the scene (q, S1 = node fetches per ray x k, the reuse distance histogram of one SM's fetch stream, lane fetches per warp fetch), computed here for 128 SMs and 72 MB L2 (the 4090).
Output: predicted speedup of each order over the natural (random) order, F123 form. Also the older Memory_4 form from the predict step, for comparison."""
import json, os, sys
sys.path.insert(0, 'model'); sys.path.insert(0, 'ttu/bench/Memory_4')
import reuse_feature as RF
par = json.load(open('notes/q11/reuse_feature.json'))['share_k1']['par']
out = {}
for tag in sys.argv[1:]:
    pj = json.load(open('notes/hs/predict_%s.json' % tag)); feats = pj['feats']
    hist = {k: RF.hist('notes/hs/reuse_%s_%s.txt' % (tag, RF.KEY[k])) for k in feats}
    s = dict(feats=feats, hist=hist)
    nat = RF.cost(par, s, 'natural', 'share'); rows = {}
    for k in feats:
        if k == 'natural': continue
        rows[k] = nat / RF.cost(par, s, k, 'share')
    out[tag] = dict(f123=rows, memory4=pj['predicted'], hit_fraction=pj['hit_fraction'], q=pj['q'], bvh_mib=pj['bvh_mib'], share_ratio={k: hist[k][4] / hist[k][0] for k in feats})
    print(tag, 'hit %.3f q %.2f BVH %.1f MiB' % (pj['hit_fraction'], pj['q'], pj['bvh_mib']))
    for k, v in rows.items(): print('   %-12s F123 %.3f   Memory_4 form %.3f   lane/warp fetch ratio %.2f' % (k, v, pj['predicted'][k], hist[k][4] / hist[k][0]))
json.dump(out, open('notes/hs/f123_predictions_%s.json' % '_'.join(sys.argv[1:]), 'x'), indent=1)
