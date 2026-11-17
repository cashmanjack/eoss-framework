# Summarize the CLOVER vs RT matrix (notes/q38/raw, timestamp 20261117T184752 = the matrix; RTNN k 50 rows 20261117T185232). Prints markdown.
import glob, re, statistics as st, os
R = os.path.expanduser('~/eoss/notes/q38/raw/')
def clv(tag, ds, k):
    f = sorted(g for g in glob.glob(R + f'cl_{tag}_a*_{ds}_N*_Q*_k{k}_20261117T1847*.txt') if '_Q36000_' not in g and '_Q20000_' not in g and '_Q5000_' not in g and '_N20000_' not in g)[-1]; t = open(f).read()
    ns = [int(x) for x in re.findall(r'^(\d+), rep', t, re.M)]
    m = re.search(r'Q (\d+) K \d+ sampled (\d+) wrong (\d+)', t)
    return st.median(ns) / 1e6, len(ns), int(m.group(1)), int(m.group(3)), int(m.group(2))
rows = [('3DRoad', '3droad', 5), ('Porto 1M', 'porto1m', 5), ('uniform 1M', 'uni1m', 5), ('bunny', 'bunny', 50), ('dragon', 'dragon', 50)]
print('| dataset | k | queries | warpwise ms | hubs ms | bitonic patched ms | bitonic stock ms | wrong in 1000 sampled (warp / hubs / bit patched / bit stock) |')
print('|---|---|---|---|---|---|---|---|')
for name, ds, k in rows:
    cells = []; wr = []; Q = None
    for tag in ['warp', 'hubs', 'bitp', 'bits']:
        try: ms, n, Q, w, c = clv(tag, ds, k); scale = 3609600 / Q if (ds == 'dragon' and tag == 'warp') else 1
        except IndexError: cells.append('not run'); wr.append('-'); continue
        cells.append(f'{ms*scale:.1f}' + (f' (x{scale:.1f} from {Q} queries)' if scale != 1 else '') + f' [{n}]'); wr.append(str(w))
    print(f'| {name} | {k} | all points | ' + ' | '.join(cells) + ' | ' + ' / '.join(wr) + ' |')
print()
print('RTNN K 50 rows (rt_rec fresh rebuild, 3 runs; launch = CUDA event of the search launch; "total on device" = sort + geometry + search compute):')
for ds, tag in [('bunny', 'r2'), ('bunny', 'r99'), ('dragon', 'r2'), ('dragon', 'r99')]:
    fs = sorted(glob.glob(R + f'rtnnk50_{ds}_{tag}_20261117T185232_*.txt')); L = []
    for f in fs:
        t = open(f).read(); g = lambda p: float(re.search(p, t).group(1))
        L.append((g(r'KERNEL queries \d+ ms ([\d.]+)'), g(r'time sort and/or partition queries: ([\d.]+)'), g(r'time create and upload geometry: ([\d.]+)'), g(r'time search compute: ([\d.]+)')))
    m = [st.median(x[i] for x in L) for i in range(4)]
    print(f'  {ds} {tag}: launch {m[0]:.3f} ms, sort {m[1]:.1f}, geometry {m[2]:.2f}, search compute {m[3]:.2f}, total on device {m[1]+m[2]+m[3]:.1f} ms')
