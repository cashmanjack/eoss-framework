"""Memory_4 (b): start node files for the rays of prep.py, one slot path per ray, in file order.

    startfiles.py WORK/s<N>

  starts_<order>_warp.txt  every lane of a warp (32 consecutive rays) starts at the longest common
                           prefix of the 32 keys (the treelet queue model: one start node per warp)
  starts_<order>_ray.txt   every ray starts at its own key node (upper bound: a perfect predictor)
Keys come from the reference closest hit (prep.py), so both are oracle starts: a ray whose key is
right finds the same closest hit as from the root; a wrong key shows up as a wrong result.
"""
import json, os, sys

d = sys.argv[1]
m = json.load(open(os.path.join(d, 'prep.json')))
keys = [tuple(k) for k in m['keys']]
fmt = lambda k: ','.join(map(str, k)) if k else 'root'
for o, perm in m['orders'].items():
    ks = [keys[i] for i in perm]
    warp = []
    for w in range(0, len(ks), 32):
        g = ks[w:w + 32]; p = ()
        while len(p) < min(map(len, g)) and all(k[len(p)] == g[0][len(p)] for k in g):
            p = g[0][:len(p) + 1]
        warp += [p] * len(g)
    for mode, s in (('warp', warp), ('ray', ks)):
        open(os.path.join(d, 'starts_%s_%s.txt' % (o, mode)), 'w').write(''.join(fmt(k) + '\n' for k in s))
        depth = [len(k) for k in s]
        print('%s %-11s %-4s start depth mean %.2f, from root %d of %d' % (os.path.basename(d), o, mode, sum(depth) / len(depth), depth.count(0), len(depth)))
