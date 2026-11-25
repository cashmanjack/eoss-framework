"""Baseline 4: TrueKNN (Nagarajan, Mandarapu, Kulkarni, ICS 2023), our port of Arkade's sample03 as built in ~/thirdparty/div1/tk/ark/b (device buffer build, clock recorder always on).
usage: python base_trueknn.py [REPS]   -> notes/base1/trueknn_<ts>.json. Change no code. The recorder cannot be switched off in this build."""
import re, sys, os, base_common as B
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
D = B.HOME + '/thirdparty/div1'; BIN = D + '/tk/ark/b/sample03-trueknn'
def parse(t):
    p = {}
    for k, pat in (('build_s', r'Build time: ([0-9.]+) seconds'), ('true_knn_s', r'True KNN time: ([0-9.]+) seconds'), ('total_s', r'Total time: ([0-9.]+)')):
        m = re.search(pat, t); p[k] = float(m.group(1)) if m else None
    p['rounds'] = [dict(round=int(a), time_s=float(b)) for a, b in re.findall(r'Round: (\d+) Time: ([0-9.]+) seconds', t)]
    p['recorder'] = [dict(round=int(m[0]), radius=float(m[1]), gt_span_ms=float(m[2]), total_warp_cycles=int(m[3]), f0_cycles=int(m[4]), f0_calls=int(m[5])) for m in re.findall(r'RND (\d+) radius ([0-9.]+) GT span_ms ([0-9.]+) CBK warps \d+ total_warp_cycles (\d+) f0_cycles (\d+) f0_calls (\d+)', t)]
    p['points_left'] = [int(x) for x in re.findall(r'RND \d+ points left (\d+)', t)]
    p['rounds_n'] = len(p['rounds']); p['gt_span_sum_ms'] = sum(r['gt_span_ms'] for r in p['recorder']) if p['recorder'] else None
    return p
doc = B.header('TrueKNN (port)', BIN, [D + '/data/u100000.csv', D + '/data/u400000.csv'], 'Uniform random 3D points (u100000 and u400000), queries = points, dim 3, start radius 0.004, K 16 (args as the earlier 4090 runs). Timers: build, per round, "True KNN time" (sum of rounds), "Total time". The output file goes to /dev/shm/base1 and is deleted.')
doc['configs'] = []; os.makedirs('/dev/shm/base1', exist_ok=True)
for n in (100000, 400000):
    data = '%s/data/u%d.csv' % (D, n); runs = []
    for i in range(REPS):
        out = '/dev/shm/base1/tk_%d_%d_%d.txt' % (n, i, os.getpid()); cmd = [BIN, data, str(n), '3', '0.004', '16', out]
        x = B.run(cmd, D + '/tk/ark/b', B.env(), 300); x['parsed'] = parse(x['stdout']); runs.append(x)
        try: os.remove(out)
        except OSError: pass
    doc['configs'].append(dict(name='n%d_k16_r0.004' % n, cmd=' '.join(cmd[:-1]).replace(B.HOME, '~') + ' OUT', runs=runs, summary=B.summarize(runs, ['build_s', 'true_knn_s', 'total_s', 'rounds_n', 'gt_span_sum_ms']) | {'wall_s': B.stats([x['wall_s'] for x in runs if x['rc'] == 0])}))
print(B.write('trueknn', doc))
