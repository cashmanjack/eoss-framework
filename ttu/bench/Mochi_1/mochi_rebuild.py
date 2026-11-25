"""Mochi_1: periodic BVH rebuild (predictions: model/predictions_mochi_rebuild.md, committed before this ran).
usage: python mochi_rebuild.py time REPS    -> timed runs, rounds interleaved over the intervals, raw json in notes/mochi1 (mode x, timestamped)
       python mochi_rebuild.py check        -> observer runs (COUNT_COLLISIONS=1), every dump of every interval compared with the CPU, raw json in notes/mochi1
Timed binary: /dev/shm/mochi_rb/b0 (COUNT 0), observer: /dev/shm/mochi_rb/b1 (COUNT 1), both the same patch (notes/mochi_rebuild_patch_20261125.diff).
Stock binary (unpatched): ~/Mochi-DCD-on-RT/build. No TTU code. GPU by PID idle check (base_common)."""
import sys, os, re, json, glob, shutil, datetime
sys.path.insert(0, os.path.expanduser('~/eoss/ttu/bench/Base_1'))
import base_common as B, base_checks as C
OUT = B.HOME + '/eoss/notes/mochi1'; os.makedirs(OUT, exist_ok=True)
DATA = B.HOME + '/Mochi-DCD-on-RT/datasets/100k_particles.txt'
STOCK = B.HOME + '/Mochi-DCD-on-RT/build/sample23-3Dparticlesim'
PT = '/dev/shm/mochi_rb/b0/sample23-3Dparticlesim'; PO = '/dev/shm/mochi_rb/b1/sample23-3Dparticlesim'
INTERVALS = [0, 4000, 400, 100, 25, 10, 1]

def wr(name, doc):
    p = '%s/%s_%s.json' % (OUT, name, datetime.datetime.now().strftime('%Y%m%dT%H%M%S'))
    with open(p, 'x') as f: json.dump(doc, f, indent=1)
    return p

def parse(t, err):
    L = [l.strip() for l in t.splitlines() if l.strip()]; p = {}
    try:
        i = L.index('100000'); v = [float(x) for x in L[i + 5:i + 9]]
        p = dict(build_ms=v[0], collision_ms=v[1], update_ms=v[2], total_ms=v[3])
    except (ValueError, IndexError): pass
    p['frames_done'] = len(re.findall(r'frame \d+ done', t))
    m = re.search(r'rebuilds (\d+) every (\d+)', err or ''); p['rebuilds'] = int(m.group(1)) if m else None
    return p

def timed(reps):
    bd = os.path.dirname(STOCK)
    arms = [('stock', STOCK, 0)] + [('patched_N%d' % n, PT, n) for n in INTERVALS]
    runs = {a[0]: [] for a in arms}
    for r in range(reps):
        for name, binp, n in arms:
            e = B.env(extra={'MOCHI_REBUILD_EVERY': str(n)} if binp == PT else None)
            x = B.run([binp, DATA, '10'], bd if binp == STOCK else os.path.dirname(binp), e, 600)
            x['parsed'] = parse(x['stdout'], x['stderr']); x['stdout'] = x['stdout'][-300:]; runs[name].append(x)
            print(r, name, x['rc'], x['parsed'], flush=True)
    keys = ['build_ms', 'collision_ms', 'update_ms', 'total_ms', 'frames_done', 'rebuilds']
    doc = dict(date=datetime.datetime.now().isoformat(timespec='seconds'), gpu=B.gpu_info(), data=B.file_info(DATA),
               binaries=dict(stock=B.file_info(STOCK), patched_timed=B.file_info(PT)), intervals=INTERVALS, reps=reps,
               note='N = rebuild every N sub steps (global counter, 400 sub steps per frame), N 0 = never; patched arms include owlBuildSBT after a rebuild in the build term',
               arms={k: dict(runs=v, summary=B.summ(v, keys)) for k, v in runs.items()})
    print(wr('mochi_rebuild_time', doc))

def check():
    res = {}
    for n in INTERVALS:
        d = '/dev/shm/mochi_rb/dump_N%d' % n; shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
        r = B.check_run([PO, DATA, '10'], os.path.dirname(PO), B.env(extra={'MOCHI_REBUILD_EVERY': str(n), 'MOCHI_DUMP_DIR': d, 'MOCHI_DUMP_EVERY': '16'}), 1800)
        files = sorted(glob.glob(d + '/f*_s*.bin')); per = []
        for f in files:
            c = C.mochi(f); m = re.search(r'f(\d+)_s(\d+)', f); c['frame'] = int(m.group(1)); c['sub_step'] = int(m.group(2)); per.append(c); os.remove(f)
        frames = sorted({c['frame'] for c in per})
        res['N%d' % n] = dict(rc=r['rc'], stderr_tail=r['stderr_tail'], dumps=len(per), frames_covered=frames, all_equal=bool(per) and all(c['equal'] for c in per),
                              n_unequal=sum(1 for c in per if not c['equal']), last_substep_of_every_frame_checked=all(any(c['frame'] == f and c['sub_step'] == 399 for c in per) for f in range(10)),
                              max_abs_diff=max((abs(c['program_collisions'] - c['cpu_collisions']) for c in per), default=None),
                              by_frame={f: dict(program=[c['program_collisions'] for c in per if c['frame'] == f], cpu=[c['cpu_collisions'] for c in per if c['frame'] == f]) for f in frames})
        print('N', n, {k: v for k, v in res['N%d' % n].items() if k != 'by_frame'}, flush=True)
        shutil.rmtree(d, ignore_errors=True)
    print(wr('mochi_rebuild_check', dict(date=datetime.datetime.now().isoformat(timespec='seconds'), gpu=B.gpu_info(), dumps_every_substeps=16, results=res, reference='CPU pair count with the device test on each dumped particle state (base_checks.mochi)')))

if __name__ == '__main__':
    if sys.argv[1] == 'time': timed(int(sys.argv[2]))
    else: check()
