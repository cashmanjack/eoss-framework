"""Offline analysis of an R3 trace run (no GPU). Words per session: [R2,R3,R30,R31,R32,R33,R34,R35,R44,R45,R56..R63,R68,R69,R70,R71]."""
import json, sys, collections
j = json.load(open(sys.argv[1])); L = j['lanes']; T = j['trace']; gas = int(j['gas'], 16); ro = j['rec_off']
N = ['R2','R3','R30','R31','R32','R33','R34','R35','R44','R45','R56','R57','R58','R59','R60','R61','R62','R63','R68','R69','R70','R71']
def rec(w):  # record index of a candidate answer
    d = {n: int(x, 16) for n, x in zip(N, w)}
    if d['R3'] != 1 or (d['R33'] >> 25) != 0x50: return None
    return (((d['R45'] << 32) | d['R44']) * 8 - gas - ro) // 8
pend = lambda w: [any(int(w[N.index(a)], 16) or int(w[N.index(b)], 16) for a, b in p) for p in ((('R56','R57'),('R58','R59')) , (('R60','R61'),('R70','R71')))]
# map record index -> primitive from neighbour sessions
m = {}
for i, l in enumerate(L):
    nb = list(l['ours']); k = 0
    for w in T[i]:
        r = rec(w)
        # a neighbour session is not identifiable without the prim; use order: the k-th neighbour of OptiX order intersect ours is unknown -> resolve later
bad = [i for i, l in enumerate(L) if l['cnt'] != l['optix_cnt'] or sorted(l['ours']) != sorted(l['optix'])]
print('bad lanes', len(bad), '| lanes with a nonzero word in R2,R30,R31,R62,R63,R68,R69 (outside the pending/answer words we use) in any session:')
extra = collections.Counter()
for i in range(len(T)):
    for s, w in enumerate(T[i]):
        for n in ('R2','R30','R31','R62','R63','R68','R69','R34','R35'):
            if n in ('R34','R35'): continue
            if int(w[N.index(n)], 16): extra[n] += 1
print(dict(extra))
tags = collections.Counter(); r3 = collections.Counter()
for i in range(len(T)):
    for w in T[i]: tags[int(w[N.index('R33')], 16) >> 25] += 1; r3[int(w[N.index('R3')], 16)] += 1
print('tags (R33>>25) over all sessions', {hex(k): v for k, v in tags.items()}, '| R3 values', dict(r3))
for i in bad[:6] + [x for x in range(32) if x not in bad][:2]:
    print('lane', i, 'BAD' if i in bad else 'ok', 'sessions', len(T[i]), 'optix cand', L[i]['optix_cand'], 'neighbours ours', L[i]['cnt'], 'optix', L[i]['optix_cnt'])
    for s, w in enumerate(T[i], 1):
        d = {n: int(x, 16) for n, x in zip(N, w)}
        print('  s%-2d rec %-8s R3 %d tag %02x pend3C0 %s pend3D0 %s  R30:31 %08x%08x R62:63 %08x%08x R68:69 %08x%08x R2 %x' % (s, rec(w), d['R3'], d['R33'] >> 25, any((d['R56'], d['R57'], d['R58'], d['R59'])), any((d['R60'], d['R61'], d['R70'], d['R71'])), d['R31'], d['R30'], d['R63'], d['R62'], d['R69'], d['R68'], d['R2']))
