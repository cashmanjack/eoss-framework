"""Summary tables from morton_s*_{sweep,all,depth}.json: rays per us, speed up over random, share of the oracle's gain recovered.
    summarize.py WORK  > report"""
import glob, json, os, sys
work = sys.argv[1]
for scene in (100000, 3000000):
    rows = []
    for part in ('sweep', 'all', 'depth'):
        p = os.path.join(work, 'morton_s%d_%s.json' % (scene, part))
        if os.path.exists(p):
            j = json.load(open(p)); rows += [(part, r) for r in j['results']]; groups = j['groups']; n = j['rays']
    if not rows: continue
    rnd = [r for _, r in rows if r['order'] == 'random' and r['mode'] == 'plain'][0]['rays_per_us']
    orc = [r for _, r in rows if r['order'] == 'oracle full' and r['mode'] == 'plain'][0]['rays_per_us']
    print('\nsoup:%d, %d rays per launch, groups %s; random plain %.0f rays/us, oracle (full subtree key) %.0f (%.2fx)' % (scene, n, groups, rnd, orc, orc / rnd))
    print('%-30s %-9s %9s %8s %10s %6s' % ('order', 'mode', 'rays/us', 'vs rand', 'oracle gain', 'wrong'))
    seen = set()
    for part, r in rows:
        k = (r['order'], r['mode'], round(r['rays_per_us']))
        if (r['order'], r['mode']) in seen: continue
        seen.add((r['order'], r['mode']))
        print('%-30s %-9s %9.0f %8.2f %9.0f%% %6d' % (r['order'], r['mode'], r['rays_per_us'], r['rays_per_us'] / rnd, 100 * (r['rays_per_us'] - rnd) / (orc - rnd), r['wrong']))
