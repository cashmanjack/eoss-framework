"""Memory_4 (b): resolve leaf blocks shared by two nodes' leaves, using a hardware check run.

    fixkeys.py WORK/s<N> LEAFRUN.json LEAFRUN.gas CHECK.json

CHECK: a run of rays_grouped.bin with starts_grouped_ray.txt. For each ray whose start there gave a
wrong result and whose leaf block is exactly at a node's leaf base, the key becomes the previous node
(next lower leaf base), on the hypothesis that the block also holds that node's last triangles.
Rewrites keys_exact.json, prep.json keys, and the start files (run startfiles.py after).
"""
import bisect, json, os, struct, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..', 'BVH_5'))
import gas  # noqa: E402

d, rpath, gpath, cpath = sys.argv[1:5]
G = open(gpath, 'rb').read(); W = gas.walk(G)
bases = sorted((off + struct.unpack_from('<i', G, off + 0x10)[0], off) for off in W if any(k == 'leaf' for _, k, _, _ in gas.slots(G, off)))
bk = [b for b, _ in bases]
m = json.load(open(os.path.join(d, 'prep.json'))); old = m['orders']['grouped_box']; g = m['orders']['grouped']
lj = json.load(open(rpath)); offs = []
for c in range(lj['nchunks']):
    offs += json.load(open(rpath if c == 0 else '%s.p0c%d' % (rpath, c)))['lane_leaf_off']
leaf_of = {old[p]: offs[p] for p in range(len(old))}
cj = json.load(open(cpath)); got = []
for c in range(cj['nchunks']):
    got += json.load(open(cpath if c == 0 else '%s.p0c%d' % (cpath, c)))['lane_best']
keys = [tuple(k) for k in m['keys']]; fixed = other = 0
for pos, i in enumerate(g):
    r = cj['ref'][pos]; bt, bp = got[pos]
    if not r['hit'] or (bt != '0xffffffff' and bp == r['prim']):
        continue
    o = leaf_of[i]; k = bisect.bisect_right(bk, o) - 1
    if bk[k] == o and k > 0:
        p = k - 1
        while p > 0 and bk[p] == bk[k]:
            p -= 1
        keys[i] = W[bases[p][1]]['path']; fixed += 1
    else:
        other += 1
m['keys'] = [list(k) for k in keys]
json.dump(m, open(os.path.join(d, 'prep.json'), 'w')); json.dump(m['keys'], open(os.path.join(d, 'keys_exact.json'), 'w'))
print('%s: %d keys moved to the previous node, %d wrong rays not explained' % (os.path.basename(d), fixed, other))
