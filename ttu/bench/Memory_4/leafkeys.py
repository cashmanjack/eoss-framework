"""Memory_4 (b): exact subtree keys from the hardware, for the same groups and orders as (a).

    leafkeys.py WORK/s<N> LEAFRUN.json LEAFRUN.gas

LEAFRUN: a root start run of rays_grouped.bin (setup3 records, per ray, the leaf block of its best
hit: lane_leaf_off) with the GAS of the same process. Hypothesis (checked by the hardware when the
starts run): a leaf block belongs to the internal node with the largest leaf base at or below it
(leaf base = node + the int at +0x10, for nodes that have leaf slots). Key = that node's slot path;
misses keep prep.py's origin key. Writes starts_<order>_{warp,ray}.txt as startfiles.py does.
"""
import bisect, json, os, struct, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'BVH_5'))
import gas  # noqa: E402

d, rpath, gpath = sys.argv[1:4]
m = json.load(open(os.path.join(d, 'prep.json')))
G = open(gpath, 'rb').read(); W = gas.walk(G)
bases = sorted((off + struct.unpack_from('<i', G, off + 0x10)[0], off) for off in W if any(k == 'leaf' for _, k, _, _ in gas.slots(G, off)))
bk = [b for b, _ in bases]
j = json.load(open(rpath)); offs = []
for c in range(j['nchunks']):
    offs += json.load(open(rpath if c == 0 else '%s.p0c%d' % (rpath, c)))['lane_leaf_off']
grouped = m['orders']['grouped']
keys = [tuple(k) for k in m['keys']]          # base index -> key; misses keep the origin key
amb = 0; changed = 0
for pos, i in enumerate(grouped):            # leaf run used the grouped order: file position pos = base ray i
    o = offs[pos]
    if o < 0:
        continue
    k = bisect.bisect_right(bk, o) - 1
    amb += (k + 1 < len(bk) and bk[k + 1] == bk[k]) or (k > 0 and bk[k - 1] == bk[k])
    new = W[bases[k][1]]['path']
    changed += new != keys[i]; keys[i] = new
print('%s: %d hits, %d keys changed vs box containment, %d hits on a leaf base shared by two nodes' % (
    os.path.basename(d), sum(o >= 0 for o in offs), changed, amb))
fmt = lambda k: ','.join(map(str, k)) if k else 'root'
for o, perm in m['orders'].items():
    ks = [keys[i] for i in perm]; warp = []
    for w in range(0, len(ks), 32):
        g = ks[w:w + 32]; p = ()
        while len(p) < min(map(len, g)) and all(k[len(p)] == g[0][len(p)] for k in g):
            p = g[0][:len(p) + 1]
        warp += [p] * len(g)
    for mode, s in (('warp', warp), ('ray', ks)):
        open(os.path.join(d, 'starts_%s_%s.txt' % (o, mode)), 'w').write(''.join(fmt(k) + '\n' for k in s))
        dep = [len(k) for k in s]
        print('  %-11s %-4s start depth mean %.2f, from root %d' % (o, mode, sum(dep) / len(dep), dep.count(0)))
json.dump([list(k) for k in keys], open(os.path.join(d, 'keys_exact.json'), 'w'))
