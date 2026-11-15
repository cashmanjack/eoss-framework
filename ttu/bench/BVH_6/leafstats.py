"""Leaf slot type byte statistics of OptiX GASes built in our own process (bvhbuild / morton --dump-gas). No TTU code.
    leafstats.py FILE.gas ...        per file: type byte histogram (flag bits, hi = bits 3..6, lo = bits 0..2), blocks advanced per node, leaf bytes vs sum of lo"""
import sys, struct, collections
sys.path.insert(0, __file__.rsplit('/', 1)[0] + '/../BVH_5'); import gas

def stats(path):
    d = open(path, 'rb').read(); W = gas.walk(d); lo_off = struct.unpack_from('<Q', d, 0x10)[0]
    ty = collections.Counter(); per_node = []; nslots = []
    for off in W:
        sl = [d[off + 0x20 + 8 * s + 7] for s in range(12) if d[off + 0x27 + 8 * s] and not (d[off + 0x24 + 8 * s] == 0xff and d[off + 0x25 + 8 * s] == 0 and d[off + 0x26 + 8 * s] == 0)]
        for t in sl: ty[t] += 1
        if sl: per_node.append(sum(t & 7 for t in sl)); nslots.append(len(sl))
    return dict(size=len(d), nodes=len(W), hdr60=struct.unpack_from('<I', d, 0x60)[0], leaf_bytes=len(d) - lo_off, types=ty, blocks_per_node=collections.Counter(per_node), slots_per_node=collections.Counter(nslots), sumlo=sum(t & 7 for t, n in ty.items() for _ in range(n)))

if __name__ == '__main__':
    for p in sys.argv[1:]:
        s = stats(p); print('==', p, 'size', s['size'], 'nodes', s['nodes'], 'header +0x60', s['hdr60'], 'leaf section bytes', s['leaf_bytes'], '= %.1f blocks of 128' % (s['leaf_bytes'] / 128), 'sum of low 3 bits', s['sumlo'])
        print('  low 3 bits (blocks advanced) per slot:', dict(sorted(collections.Counter({k & 7: 0 for k in s['types']}).items())), end=' ')
        lo = collections.Counter()
        for t, n in s['types'].items(): lo[t & 7] += n
        print(dict(sorted(lo.items())))
        print('  flag bits 7 and 6:', dict(collections.Counter({(t >> 6): 0 for t in s['types']})), {k: sum(n for t, n in s['types'].items() if t >> 6 == k) for k in sorted({t >> 6 for t in s['types']})})
        print('  bits 3..6 (hi) values:', {k: sum(n for t, n in s['types'].items() if (t >> 3) & 15 == k) for k in sorted({(t >> 3) & 15 for t in s['types']})})
        print('  blocks per node (sum of low bits) histogram:', dict(sorted(s['blocks_per_node'].items())), ' leaf slots per node:', dict(sorted(s['slots_per_node'].items())))
