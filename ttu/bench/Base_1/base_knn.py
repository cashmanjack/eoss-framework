"""Baseline 2 (v2): our growing radius kNN OptiX program (knn_host), ~/thirdparty/div1/knn build (original heap in global memory).
usage: python base_knn.py [REPS]   -> notes/base1/knn_v2_<ts>.json. Change no code.
Input: u200000.csv = default_rng(5).random((200000,3)), the recorded recipe of the earlier 4090 input (u2e5; the original file is gone, not byte compared).
Correctness (untimed, same binary): the program's Active count per round against a CPU kd tree count of queries with fewer than K points inside the radius (sphere and cube)."""
import re, sys, os, base_common as B, base_checks as C
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
D = B.HOME + '/thirdparty/div1/knn'; BIN = D + '/knn_host'; DATA = B.HOME + '/thirdparty/base_data/u200000.csv'
def parse(t):
    p = {}
    m = re.search(r'Total Search Time: (\d+)ms', t); p['total_search_ms'] = float(m.group(1)) if m else None
    m = re.search(r'Loaded (\d+) pts and (\d+) queries in (\d+)ms', t); p['load_ms'] = float(m.group(3)) if m else None
    p['rounds'] = [dict(active=int(a), radius=float(r)) for a, r in re.findall(r'\[Round \d+\] Active=(\d+), Radius=([0-9.]+)', t)]
    return p
doc = B.header('kNN (growing radius, own OptiX program)', BIN, [DATA], '200,000 uniform random 3D points, queries = points, K 16, start radius 0.004 doubling (the earlier 4090 setting). The program times only the search loop (no per round timers, no build timer). Warmup process before each run.')
cmd = [BIN, DATA, DATA, '16', '0.004']; runs = []
for i in range(REPS):
    x = B.run(cmd, D, B.env(), 120); x['parsed'] = parse(x['stdout']); runs.append(x)
chk = B.run(cmd, D, B.env(), 120, warm=False); pr = parse(chk.pop('stdout')); chk.pop('warmup', None); chk.pop('clock', None)
pts = C.load_csv(DATA); act = [r['active'] for r in pr['rounds']]; rad = [r['radius'] for r in pr['rounds']]
chk['round_counts'] = C.knn_rounds(pts, 16, rad, act) | dict(note='Active(round i+1) against CPU counts of queries with fewer than K points in the sphere / cube of radius_i. Not expected to match exactly: the program inserts every box candidate the hardware returns (boxes inflated, F116), so it can have more candidates than the exact cube; it should be at or below the cube count.')
OBS = '/dev/shm/knnobs/knn_host'; dp = '/dev/shm/base1/knn_%d' % os.getpid()
rr = B.check_run([OBS, DATA, DATA, '16', '0.004'], '/dev/shm/knnobs', B.env(extra={'KNN_DUMP': dp}), 300)
chk['correctness'] = C.knn_lists(dp, pts, 16, rad) | dict(reference='observer copy (same PTX, byte identical; host dumps each round\'s heaps) against the exact 16 nearest by CPU kd tree (self counted)')
for i in range(1, len(rad) + 1):
    try: os.remove('%s_r%d.bin' % (dp, i))
    except OSError: pass
doc['configs'] = [dict(name='k16_r0.004', cmd=' '.join(cmd).replace(B.HOME, '~'), runs=runs, summary=B.summ(runs, ['total_search_ms', 'load_ms']), correctness=chk)]
print(B.write('knn_v2', doc))
