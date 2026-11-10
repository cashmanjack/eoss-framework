"""Memory_4 summary tables from WORK (after run.py): timing per finished ray and ncu traffic per finished ray.

    summary.py WORK [OUT.json]
"""
import glob, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from analyze import analyze  # noqa: E402
import ncu_analyze  # noqa: E402

VARS = ['interleaved', 'groupedbox', 'grouped', 'b_grouped_warp', 'b_grouped_ray', 'b_interleaved_ray']
work = sys.argv[1]; res = dict(timing=[], ncu=[])
for n in ('100000', '3000000'):
    d = os.path.join(work, 's' + n)
    print('\n## soup:%s  timing (finished rays, all queries counted)' % n)
    print('%-18s %-4s %3s | %8s %8s %6s | %9s %9s | %5s %5s' % ('variant', 'state', 'W', 'rays/cyc', 'cyc/ray', 'vs int', 'lat/ray', 'per query', 'q/ray', 'wrong'))
    for st in ('warm', 'cold'):
        for w in (1, 4, 8, 16):
            base = None
            for v in VARS:
                p = os.path.join(d, 't', '%s_w%d_%s.json' % (v, w, st))
                if not os.path.exists(p):
                    continue
                r = analyze(p); r.update(variant=v, state=st, scene=n); res['timing'].append(r)
                if v == 'interleaved':
                    base = r
                rel = (r['rays_per_cycle'] / base['rays_per_cycle']) if base else float('nan')
                print('%-18s %-4s %3d | %8.4f %8.1f %6.2f | %9.0f %9.0f | %5.2f %5d' % (v, st, w, r['rays_per_cycle'], r['cycles_per_ray'], rel,
                                                                                    r['lat_mean'], r['query_lat_mean'], r['queries_per_ray'], r['wrong']))
    nd = os.path.join(d, 'n')
    if '--timing' not in sys.argv and glob.glob(os.path.join(nd, '*_B.csv')):
        print('\n## soup:%s  ncu, RT core traffic per finished ray (32 B sectors)' % n)
        print('%-18s %-4s %3s | %7s %6s | %8s | %8s %6s | %8s' % ('variant', 'state', 'W', 'L1 sect', 'L1 hit', 'L1 fill', 'L2 read', 'L2 hit', 'DRAM B'))
        for st in ('warm', 'cold'):
            for w in (4, 8, 16):
                for v in VARS:
                    if not os.path.exists(os.path.join(nd, '%s_w%d_%s_B.csv' % (v, w, st))):
                        continue
                    r = ncu_analyze.one(nd, w, st, v); r.update(scene=n); res['ncu'].append(r)
                    print('%-18s %-4s %3d | %7.1f %5.1f%% | %8.1f | %8.1f %5.1f%% | %8.0f' % (v, st, w, r['l1_sectors_per_ray'], 100 * r['l1_hit'],
                                                                                      r['l1_fill_sectors_per_ray'], r['l2_read_sectors_per_ray'], 100 * r['l2_hit'], r['dram_bytes_per_ray']))
out = [a for a in sys.argv[2:] if not a.startswith('--')]
if out:
    json.dump(res, open(out[0], 'w'), indent=1)
