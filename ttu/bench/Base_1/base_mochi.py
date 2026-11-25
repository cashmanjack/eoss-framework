"""Baseline 6 (v2): Mochi (Mandarapu, James, Kulkarni; DCD on RT cores), UNPATCHED build ~/Mochi-DCD-on-RT/build/sample23-3Dparticlesim (RENDER 0, COUNT_COLLISIONS 0, NUM_PARTICLES 100000, BOX_WIDTH 1.0, sm_89).
usage: python base_mochi.py [REPS]   -> notes/base1/mochi_v2_<ts>.json. Change no code (timed runs use the stock build above).
Correctness (untimed): an observer copy of the same source with COUNT_COLLISIONS=1 (a compile flag) and one dump added (/dev/shm/mochiobs, notes/base1/mochi_observer_dump.diff) writes
the particle state before the last collision launch of a frame and the program's collision count for that launch; the CPU counts the same pairs with the device test. Frames 0, 5 and 9."""
import re, sys, os, base_common as B, base_checks as C
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
BD = B.HOME + '/Mochi-DCD-on-RT/build'; BIN = BD + '/sample23-3Dparticlesim'; DATA = B.HOME + '/Mochi-DCD-on-RT/datasets/100k_particles.txt'; OBS = '/dev/shm/mochiobs/build/sample23-3Dparticlesim'
def parse(t):
    L = [l.strip() for l in t.splitlines() if l.strip()]; p = {}
    try:
        i = L.index('100000'); v = [float(x) for x in L[i + 5:i + 9]]
        p = dict(build_ms=v[0], collision_ms=v[1], update_ms=v[2], total_ms=v[3])
    except (ValueError, IndexError): pass
    p['frames_done'] = len(re.findall(r'frame \d+ done', t))
    return p
doc = B.header('Mochi', BIN, [DATA], '100,000 particles (100k_particles.txt, the repo file, 400 sub steps per frame), 10 frames. Program timers (host, milliseconds): build = BVH refit per sub step summed (first build excluded or included as the program counts it), collision = the collision launch, update = the position update kernel; total = their sum. Warmup process before each run.')
cmd = [BIN, DATA, '10']; runs = []
for i in range(REPS):
    x = B.run(cmd, BD, B.env(), 300); x['parsed'] = parse(x['stdout']); runs.append(x)
chk = {'frames': []}; os.makedirs('/dev/shm/base1', exist_ok=True)
for fr in (0, 5, 9):
    fo = '/dev/shm/base1/mochi_%d_%d.bin' % (fr, os.getpid())
    r = B.check_run([OBS, DATA, '10'], os.path.dirname(OBS), B.env(extra={'MOCHI_DUMP': fo, 'MOCHI_DUMP_FRAME': str(fr)}), 600)
    c = C.mochi(fo) if os.path.exists(fo) else dict(error='no dump written', rc=r['rc'], tail=r['stdout_tail'])
    c['frame'] = fr; chk['frames'].append(c)
    if os.path.exists(fo): os.remove(fo)
chk['reference'] = 'CPU: all (query, hit) pairs with the device test (2 r_hit > d, r_q + r_hit > d, q < hit or 2 r_q < d) on the dumped particle state of the last sub step of the frame, against the program\'s col_count sum for that launch'
doc['configs'] = [dict(name='100k_10frames', cmd=' '.join(cmd).replace(B.HOME, '~'), runs=runs, summary=B.summ(runs, ['build_ms', 'collision_ms', 'update_ms', 'total_ms', 'frames_done']), correctness=chk)]
print(B.write('mochi_v2', doc))
