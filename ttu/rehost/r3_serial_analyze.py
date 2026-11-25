"""Offline: per query, our session candidates (record index -> primitive through the saved GAS record table) against OptiX's full candidate order; the first diverging session."""
import json, sys, collections
j = json.load(open(sys.argv[1])); T = j['trace']; gas = int(j['gas'], 16); ro = j['rec_off']; RP = j['rec_prim']; OC = j['optix_cands']
N = ['R2','R3','R30','R31','R32','R33','R34','R35','R44','R45','R56','R57','R58','R59','R60','R61','R62','R63','R68','R69','R70','R71']
def parse(w):
    d = {n: int(x, 16) for n, x in zip(N, w)}
    if d['R3'] != 1 or (d['R33'] >> 25) != 0x50: return None, d
    return RP[(((d['R45'] << 32) | d['R44']) * 8 - gas - ro) // 8], d
pend = lambda d: (any((d['R56'], d['R57'], d['R58'], d['R59'])), any((d['R60'], d['R61'], d['R70'], d['R71'])))
rows = []; kinds = collections.Counter()
for i, t in enumerate(T):
    ours = [parse(w)[0] for w in t]; ours_c = [p for p in ours if p is not None]; ox = OC[i]
    if ours_c == ox: rows.append((i, 'exact', len(ours_c), len(ox), None)); continue
    k = next((q for q in range(min(len(ours_c), len(ox))) if ours_c[q] != ox[q]), min(len(ours_c), len(ox)))
    kind = 'ours is a subsequence of OptiX' if set(ours_c) <= set(ox) and all(a in ox for a in ours_c) else 'other'
    rows.append((i, kind, len(ours_c), len(ox), k))
print('lane  kind  ours_sessions(candidates)  optix_candidates  first diverging session (1 based)')
for r in rows: print(r[0], r[1], r[2], r[3], (r[4] + 1) if r[4] is not None else '-')
bad = [r for r in rows if r[1] != 'exact']
print('bad', len(bad), '| first diverging session histogram', dict(collections.Counter(r[4] + 1 for r in bad)))
# detail of the first 6 bad lanes: ours vs optix around the divergence, with the previous answer's pending pattern
for r in bad[:6]:
    i, k = r[0], r[4]; ours = [parse(w) for w in T[i]]
    print('lane', i, 'first divergence at session', k + 1, '| OptiX next candidates', OC[i][max(0, k - 2):k + 4], '| ours', [o[0] for o in ours[max(0, k - 2):k + 4]], '| pending (3C0,3D0) of session', k, ':', pend(ours[k - 1][1]) if k else None, 'of session', k + 1, ':', pend(ours[k][1]) if k < len(ours) else None)
