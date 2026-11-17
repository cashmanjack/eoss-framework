"""Flags_1 summary table (reads the .summary.json files written by ttu/bench/Flags_1/run.py; CPU only).  flags_analyze.py DIR"""
import glob, json, os, sys
d = sys.argv[1]
S = {}
for f in sorted(glob.glob(os.path.join(d, '*.summary.json'))):
    r = json.load(open(f)); S[r['name']] = r   # the latest file of a name wins
order = ['t1_none', 't1_term', 't1s_none', 't1s_term', 't2_noah', 't2_ah_none', 't2_ah_disable', 't2_ah_enforce', 't2_geom_opaque', 't3_none', 't3_cull_back', 't3_cull_front', 't3_cull_both',
         't4_C_one_instance', 't4_D_none_masked', 't4_A_63_masked', 't4_B_nearest_visible'] + sorted([k for k in S if k.startswith('t5_')], key=lambda k: float(k.split('_')[2]))
print('%-22s %7s %7s %7s | %5s %5s %5s %5s | %7s %7s %7s | %6s %6s' % ('config', 'tr cold', 'tr 2nd', 'tr 3rd', 'hit', 'prim', 'front', 'ah', 'ns/ray', 'RTsect', 'fills', 'fill32', 'inst/r'))
for k in order:
    if k not in S: continue
    r = S[k]
    print('%-22s %7.0f %7.0f %7.0f | %5.2f %5d %5d %5.1f | %7.3f %7.2f %7.2f | %6.0f %6.0f' % (k, r['trace0'], r['trace1'], r['trace2'], r['hit'], r['prim'], r['front'], r['ah'], r['big_ns_per_ray'], r['rt_sectors'], r['fills'], r['fills32'], r['thread_inst']))
