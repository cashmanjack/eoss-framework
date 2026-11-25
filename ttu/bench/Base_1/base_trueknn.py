"""Baseline 4 (v2): TrueKNN (Nagarajan, Mandarapu, Kulkarni, ICS 2023), our port of Arkade's sample03, ~/thirdparty/div1/tk/ark/b (device buffer build, clock recorder always on).
usage: python base_trueknn.py [REPS]   -> notes/base1/trueknn_v2_<ts>.json. Change no code (timed runs use the build above).
Correctness (untimed): an observer copy built from the same source in /dev/shm/tkobs (only a dump of the final neighbour buffer added, notes/base1/tk_observer_dump.diff)
against an exact CPU kd tree K nearest. TrueKNN is known to return approximate lists (stop rule, F218/F227); the fraction exact is reported, not required to be 1."""
import re, sys, os, base_common as B, base_checks as C
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
D = B.HOME + '/thirdparty/div1'; BIN = D + '/tk/ark/b/sample03-trueknn'; OBS = '/dev/shm/tkobs/build/sample03-trueknn'
def parse(t):
    p = {}
    for k, pat in (('build_s', r'Build time: ([0-9.]+) seconds'), ('true_knn_s', r'True KNN time: ([0-9.]+) seconds'), ('total_s', r'Total time: ([0-9.]+)')):
        m = re.search(pat, t); p[k] = float(m.group(1)) if m else None
    p['rounds'] = [dict(round=int(a), time_s=float(b)) for a, b in re.findall(r'Round: (\d+) Time: ([0-9.]+) seconds', t)]
    p['recorder'] = [dict(round=int(m[0]), radius=float(m[1]), gt_span_ms=float(m[2]), total_warp_cycles=int(m[3]), f0_cycles=int(m[4]), f0_calls=int(m[5])) for m in re.findall(r'RND (\d+) radius ([0-9.]+) GT span_ms ([0-9.]+) CBK warps \d+ total_warp_cycles (\d+) f0_cycles (\d+) f0_calls (\d+)', t)]
    p['points_left'] = [int(x) for x in re.findall(r'RND \d+ points left (\d+)', t)]
    p['rounds_n'] = len(p['rounds']); p['gt_span_sum_ms'] = sum(r['gt_span_ms'] for r in p['recorder']) if p['recorder'] else None
    return p
doc = B.header('TrueKNN (port)', BIN, [D + '/data/u100000.csv', D + '/data/u400000.csv'], 'Uniform random 3D points (default_rng seeds 1 and 3, the recorded recipe; seed 1 checked against the file), queries = points, dim 3, start radius 0.004, K 16. Timers: build, per round, "True KNN time" (sum of rounds), "Total time". Output (time) file in /dev/shm/base1, deleted. Warmup process before each run.')
doc['configs'] = []; os.makedirs('/dev/shm/base1', exist_ok=True)
for n in (100000, 400000):
    data = '%s/data/u%d.csv' % (D, n); runs = []
    for i in range(REPS):
        out = '/dev/shm/base1/tk_%d_%d_%d.txt' % (n, i, os.getpid()); cmd = [BIN, data, str(n), '3', '0.004', '16', out]
        x = B.run(cmd, D + '/tk/ark/b', B.env(), 300); x['parsed'] = parse(x['stdout']); runs.append(x)
        try: os.remove(out)
        except OSError: pass
    fb = '/dev/shm/base1/tkfb_%d_%d.bin' % (n, os.getpid()); tf = fb + '.t'
    chk = B.check_run([OBS, data, str(n), '3', '0.004', '16', tf], '/dev/shm/tkobs/build', B.env(extra={'TK_FB': fb}), 600)
    chk['correctness'] = C.trueknn(fb, C.load_csv(data), 16) | dict(reference='CPU kd tree exact 16 nearest (float32 points widened to float64, self removed by index), exact = every sorted distance within 1e-5 relative')
    for f in (fb, tf):
        try: os.remove(f)
        except OSError: pass
    doc['configs'].append(dict(name='n%d_k16_r0.004' % n, cmd=' '.join(cmd[:-1]).replace(B.HOME, '~') + ' OUT', runs=runs, summary=B.summ(runs, ['build_s', 'true_knn_s', 'total_s', 'rounds_n', 'gt_span_sum_ms']), correctness=chk))
print(B.write('trueknn_v2', doc))
