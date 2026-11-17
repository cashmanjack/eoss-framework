"""F123 'share' form (kappa 1), NO refit, predictions for camera and bounce rays on graphics scenes (2026-11-17, tethys).
    /usr/bin/python3 model/predict_scenes.py TAG [TAG ...]      reads notes/hs2/predict_TAG.json and notes/hs2/reuse_TAG_*.txt (written by ttu/bench/Morton/scene_check.py predict)
Parameters: notes/q11/reuse_feature.json 'share_k1' (as model/predict_heldout.py). Output: predicted speedup of each order over the natural (pixel) order, F123 form, the Memory_4 form of the predict step
for comparison, and the baseline 'predict 1.0' (no change) that the scorer reports next to them. Writes notes/hs2/f123_scenes_<tags>.json (refuses to overwrite) and prints a markdown table."""
import json, sys
sys.path.insert(0, 'model'); sys.path.insert(0, 'ttu/bench/Memory_4')
import reuse_feature as RF
par = json.load(open('notes/q11/reuse_feature.json'))['share_k1']['par']
ORD = ['shuffled', 'morton', 'oracle d1', 'oracle d2', 'oracle d3', 'oracle d4', 'oracle full']
out = {}
print('| set | rays hit | q | BVH MiB | ' + ' | '.join(ORD) + ' |'); print('|---|---|---|---|' + '---|' * len(ORD))
for tag in sys.argv[1:]:
    pj = json.load(open('notes/hs2/predict_%s.json' % tag)); feats = pj['feats']
    hist = {k: RF.hist('notes/hs2/reuse_%s_%s.txt' % (tag, k.replace(' ', '_'))) for k in feats}
    s = dict(feats=feats, hist=hist); nat = RF.cost(par, s, 'natural', 'share')
    rows = {k: nat / RF.cost(par, s, k, 'share') for k in feats if k != 'natural'}
    out[tag] = dict(f123=rows, memory4=pj['predicted'], baseline=1.0, hit_fraction=pj['hit_fraction'], q=pj['q'], bvh_mib=pj['bvh_mib'], share_ratio={k: hist[k][4] / hist[k][0] for k in feats})
    print('| %s | %.3f | %.2f | %.1f | ' % (tag, pj['hit_fraction'], pj['q'], pj['bvh_mib']) + ' | '.join('%.3f (%.3f)' % (rows[k], pj['predicted'][k]) for k in ORD) + ' |')
json.dump(out, open('notes/hs2/f123_scenes_%s.json' % '_'.join(sys.argv[1:]), 'x'), indent=1)
