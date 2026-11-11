"""Tables from netsort_s*_*.json (netsort_run.py). For every scene, size and ray set: baselines (rays/us), best key by net time, net speedup
  full  = keygen + sort + gather + trace + scatter (rays copied into sorted order, results copied back)
  index = keygen + sort + trace (lower bound on the overhead: rays generated in the raygen from their index, results written in place)
and trace only speedup (sorted launch alone). Speedups against the natural launch order (the order the ray set is generated in) and against a random shuffle.
    netsort_summary.py WORK"""
import glob, json, os, sys
rows = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(sys.argv[1], 'netsort_s*_*.json')))]
rows.sort(key=lambda j: (j['scene'], j['rays'] if j['kind'] != 'bounce' else 0, j['kind']))
print('%-8s %-9s %9s | %7s %7s | %-8s | %6s %6s | %6s %6s | %6s | %s' % ('scene', 'set', 'rays', 'nat r/us', 'shuf', 'best key', 'full/nat', '/shuf', 'idx/nat', '/shuf', 'trace/nat', 'ms: keygen sort gather trace scatter'))
for j in rows:
    nat, shuf = j['baseline']['natural'], j['baseline']['shuffled']; n = j['rays']
    best = min(j['rows'], key=lambda m: m['net']); bi = min(j['rows'], key=lambda m: m['keygen'] + m['sort'] + m['ms'])
    full = best['net']; idx = bi['keygen'] + bi['sort'] + bi['ms']
    assert all(m['wrong'] == 0 for m in j['rows'])
    print('%-8d %-9s %9d | %7.0f %7.0f | %-8s | %6.2f %6.2f | %6.2f %6.2f | %6.2f | %.2f %.2f %.2f %.2f %.2f   (idx best %s)' % (
        j['scene'], j['kind'], n, n / nat / 1000, n / shuf / 1000, best['mode'][6:], nat / full, shuf / full, nat / idx, shuf / idx, nat / bi['ms'],
        best['keygen'], best['sort'], best['gather'], best['ms'], best['scatter'], bi['mode'][6:]))
