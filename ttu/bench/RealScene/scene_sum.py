"""Summarise scene_<name>.json files from scene_run.py.
    scene_sum.py JSON [JSON...]
Per scene and depth: speedup over pixel order (trace only) of Morton keys, oracle groupings and the shuffled order; Morton net of costs (keygen + sort + gather + trace + scatter);
traversal fraction of a frame stand in: (closest hit trace + shadow ray trace) / (that + k x shade kernel) with k = 1 (our minimal shader), 3, 10 (a heavier shader).
Frame totals over depths 1..D: pixel order, best Morton (trace only and net, chosen per depth), oracle full (trace only).
"""
import json, sys

for f in sys.argv[1:]:
    d = json.load(open(f)); rows = d['rows']
    print('== %s: %d triangles, GAS buffer %.1f MB = %.2f x 96 MB L2; camera %s' % (d['scene'], d['triangles'], d['gas_bytes'] / 2**20, d['gas_bytes'] / 2**20 / 96, d['camera']))
    keys = [m['mode'][6:] for m in rows[0]['morton']]; ors = [o['order'] for o in rows[0]['oracle']]
    print(' depth     rays  hit%%  pixel_ms | shuf | %s | net: %s | %s | shadow_ms shade_ms | trav frac k1 k3 k10' % (' '.join(keys), ' '.join(keys), ' '.join(ors)))
    tot = dict(pix=0, mt=0, mn=0, orc=0, sh=0, shade=0, sh_best=0)
    for r in rows:
        nat = r['pixel_ms']; mt = [nat / m['ms'] for m in r['morton']]; mn = [nat / m['net'] for m in r['morton']]; ox = [nat / o['ms'] for o in r['oracle']]
        fl = r.get('floor_ms', 0.0)
        T = nat + r['shadow_ms']
        fr = [T / (T + k * r['shade_ms']) for k in (1, 3, 10)]
        print(' %5d %9d %5.1f %8.3f | %.2f | %s | %s | %s | %.3f %.3f | %.2f %.2f %.2f' % (r['depth'], r['rays'], 100 * r['hit_fraction'], nat, nat / r['shuffled_ms'], ' '.join('%.2f' % v for v in mt), ' '.join('%.2f' % v for v in mn), ' '.join('%.2f' % v for v in ox), r['shadow_ms'], r['shade_ms'], *fr))
        print('       floor (same ray records with tmax 0: ray read and result write only, no traversal work) %.3f ms = %.2f of the pixel order trace time' % (fl, fl / nat))
        tot['pix'] += nat; tot['mt'] += min(nat, min(m['ms'] for m in r['morton'])); tot['mn'] += min(min(m['net'] for m in r['morton']), nat); tot['orc'] += min(nat, min(o['ms'] for o in r['oracle'])); tot['sh'] += r['shadow_ms']; tot['shade'] += r['shade_ms']
    print(' frame (closest hit only, depths 1..%d): pixel %.2f ms; best Morton per depth, trace only %.2f ms (x%.2f), net with a pixel order fallback %.2f ms (x%.2f); best oracle trace only %.2f ms (x%.2f)' % (
        len(rows), tot['pix'], tot['mt'], tot['pix'] / tot['mt'], tot['mn'], tot['pix'] / tot['mn'], tot['orc'], tot['pix'] / tot['orc']))
    T = tot['pix'] + tot['sh']
    print(' frame traversal fraction (closest hit + shadow rays) / (that + k x shade): k1 %.2f, k3 %.2f, k10 %.2f; closest hit share of traversal %.2f' % (T / (T + tot['shade']), T / (T + 3 * tot['shade']), T / (T + 10 * tot['shade']), tot['pix'] / T))
