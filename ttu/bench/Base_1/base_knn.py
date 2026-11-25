"""Baseline 2: our growing radius kNN OptiX program (knn_host), the build in ~/thirdparty/div1/knn (original heap in global memory, not the register heap variant).
usage: python base_knn.py [REPS]   -> notes/base1/knn_<ts>.json. Change no code."""
import re, sys, base_common as B
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
D = B.HOME + '/thirdparty/div1/knn'; BIN = D + '/knn_host'; DATA = D + '/u1e5.csv'
def parse(t):
    p = {}
    m = re.search(r'Total Search Time: (\d+)ms', t); p['total_search_ms'] = float(m.group(1)) if m else None
    m = re.search(r'Loaded (\d+) pts and (\d+) queries in (\d+)ms', t); p['load_ms'] = float(m.group(3)) if m else None
    p['rounds'] = [dict(active=int(a), radius=float(r)) for a, r in re.findall(r'\[Round \d+\] Active=(\d+), Radius=([0-9.]+)', t)]
    return p
doc = B.header('kNN (growing radius, own OptiX program)', BIN, [DATA], '100,000 uniform random 3D points (u1e5.csv, identical to u100000.csv), queries = points, K 16, start radius 0.004 doubling. The earlier 4090 tables used 200,000 points; that file no longer exists, so this input is smaller. The program times only the search loop (no per round timers, no build timer).')
cmd = [BIN, DATA, DATA, '16', '0.004']; runs = []
for i in range(REPS):
    x = B.run(cmd, D, B.env(), 120); x['parsed'] = parse(x['stdout']); runs.append(x)
doc['configs'] = [dict(name='k16_r0.004', cmd=' '.join(cmd).replace(B.HOME, '~'), runs=runs, summary=B.summarize(runs, ['total_search_ms', 'load_ms']) | {'wall_s': B.stats([x['wall_s'] for x in runs if x['rc'] == 0])})]
print(B.write('knn', doc))
