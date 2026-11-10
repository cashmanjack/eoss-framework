"""Throughput_8 with full overlap: long coherent queries (par:256:s, one query about 29,000 cycles, the BVH
fits in L1) so two blocks launched together overlap almost completely. Per block: rate = 32 W rays / SM
clock span (first warp start to last warp end); overlap = shared global timer interval / the longer block.

    full_overlap.py WORK/w
"""
import json, os, statistics as st, sys

d = sys.argv[1]
print('%-6s | %-26s | %-26s | %6s | %s' % ('SMs', 'block A: sm, rays/cyc', 'block B: sm, rays/cyc', 'pair', 'overlap'))
for p in ('0,0', '2,2', '0,1', '2,3', '10,11', '1,3', '0,2', '2,4'):
    j = json.load(open(os.path.join(d, 'pair_%s.json' % p)))
    rates = {}; ov = []
    for l in j['launches'][1:]:
        bl = l['per_block']
        for b in bl:
            sp = max(e for _, e in b['warp_clk']) - min(s for s, _ in b['warp_clk'])
            rates.setdefault(b['sm'], []).append(32 * len(b['warp_clk']) / sp)
        if len(bl) == 2:
            iv = [(min(s for s, _ in b['gt']), max(e for _, e in b['gt'])) for b in bl]
            inter = max(0, min(iv[0][1], iv[1][1]) - max(iv[0][0], iv[1][0])); ov.append(inter / max(iv[0][1] - iv[0][0], iv[1][1] - iv[1][0]))
    sms = sorted(rates)
    med = {s: st.median(rates[s]) for s in sms}
    print('%-6s | %s | %6.4f | %s' % (p, ' | '.join('SM %3d  %.5f' % (s, med[s]) + ' ' * 6 for s in sms) + (' ' * 29 if len(sms) == 1 else ''),
          sum(med.values()), ('%.0f..%.0f%%' % (100 * min(ov), 100 * max(ov))) if ov else 'solo'))
