"""BITTU compute mode validation on the GPU that gpucfg selects (written 2026-11-18 for the RTX 6000 Ada; same method as F119 and F122 on the 4070 Ti).

    validate_gpu.py prepare WORK      CPU only: host, ptx, words, stage 0 and stage 1 --clock builds (check_seq ALL PASS on both), the NVBit tool
    validate_gpu.py run WORK OUTDIR   GPU: the validated sequences only, under run.py's gate (idle GPU, check_seq marker, timeout -k 2 30), smallest launch first,
                                      STOP at the first fault, hang, refused check or unexpected output
Run with ~/eoss/.venv/bin/python after `source ~/cuda128.sh`; WORK must be /dev/shm/t9r (timing_fix.py uses that path for the builds). The patched builds
contain NVIDIA code: they stay in /dev/shm and are deleted by `validate_gpu.py clean WORK`. Raw outputs go to OUTDIR with timestamped names, never overwritten.

Run order (one warp, tri1 scene, ray hit):
  1 stage 0 plain, 1 launch           (no TTU instruction)
  2 stage 0 twin under the tool, empty probe pair, 1 launch
  3 stage 1 plain, 1 launch           (first TTU run: one warp, one launch)
  4 stage 1 under the tool, 1 launch
  5 checks: analyze.py on both stage 1 outputs (hit, t against the CPU reference, lanes agree), the tool's report (decode, instance count, masks)
  6 timing_fix.main: scenes depth 1, 2, 3, 5 launches each (F122 method)
"""
import json, os, re, subprocess, sys, time, hashlib
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
S3 = os.path.join(ROOT, 'ttu', 'harness', 'setup3'); VENV = os.path.expanduser('~/eoss/.venv/bin/python'); PY = '/usr/bin/python3'
OPTIX = os.path.expanduser('~/optix_sdk/include'); NVCC = os.path.expanduser('~/opt/cuda-12.8/bin/nvcc')


def sh(c, **kw):
    return subprocess.run(c, capture_output=True, text=True, **kw)


def die(msg, p=None):
    sys.exit('STOP: ' + msg + (('\n' + p.stdout[-2500:] + p.stderr[-1500:]) if p is not None else ''))


def words_txt(path):
    sys.path.insert(0, os.path.join(ROOT, 'notes', 'setup3')); import store_words
    p0, a, b = store_words.store_words(0, 0xFF)        # case "none, mask 255" (our OptiX runs), P0 true path
    open(path, 'w').write(' '.join('%08x' % x for x in a + b) + '\n')


def prepare(work):
    os.makedirs(work, exist_ok=True)
    c = sh([NVCC, '-O2', '-arch=sm_89', '-I' + OPTIX, '-o', work + '/setup3', S3 + '/setup3.cu', '-lcuda', '-lnvidia-ml']); c.returncode and die('host build', c)
    c = sh([NVCC, '-O2', '-arch=sm_89', '-ptx', '-I' + OPTIX, '-o', work + '/setup3_dev.ptx', S3 + '/setup3_dev.cu']); c.returncode and die('ptx build', c)
    words_txt(work + '/words.txt')
    for st in (0, 1):
        out = '%s/g_s%dc' % (work, st)
        c = sh([VENV, S3 + '/build_patch.py', '--stage', str(st), '--bin', work + '/setup3', '--clock', '--out', out]); c.returncode and die('build_patch stage %d' % st, c)
        k = sh([VENV, S3 + '/check_seq.py', out])
        if 'ALL PASS' not in (k.stdout.strip().splitlines() or [''])[-1]: die('check_seq stage %d' % st, k)
        print('stage %d build %s check_seq ALL PASS sha256 %s' % (st, out, hashlib.sha256(open(out, 'rb').read()).hexdigest()[:8]), flush=True)
    c = sh([HERE + '/build.sh', '/dev/shm/bittu']); c.returncode and die('tool build', c); print(c.stdout.strip())


def gate(cmd, env=None):
    p = sh(cmd, env=env)
    if p.returncode or 'ttu_one: no error' not in p.stdout or 'exit 0' not in p.stdout: die('run failed: ' + ' '.join(cmd[:4]), p)
    return p.stdout


def run(work, outdir):
    os.makedirs(outdir, exist_ok=True); ts = time.strftime('%Y%m%dT%H%M%S'); B0, B1 = work + '/g_s0c', work + '/g_s1c'
    for b in (B0, B1):
        if not os.path.exists(b + '.checked'): die('no check_seq marker for ' + b)
    ex = ['--scene', 'tri1', '--warps', '1', '--repeat', '1']; o = lambda n: os.path.join(outdir, 'bittu6000_%s_%s' % (n, ts)); log = []
    env = dict(os.environ, TTU_GPU=os.environ.get('TTU_GPU', '1'), BITTU_PRINT='0'); env.pop('CUDA_VISIBLE_DEVICES', None)
    steps = [('1 stage 0 plain', [PY, S3 + '/run.py', B0, 'hit', 'zero', o('s0.json')] + ex, env),
             ('2 stage 0 twin under the tool (empty probe pair)', [VENV, HERE + '/run_bittu.py', B0, 'hit', 'zero', o('t0.json'), o('t0.txt')] + ex, dict(env, BITTU_LIKE=B1)),
             ('3 stage 1 plain (first TTU run)', [PY, S3 + '/run.py', B1, 'hit', 'zero', o('s1.json')] + ex, env),
             ('4 stage 1 under the tool', [VENV, HERE + '/run_bittu.py', B1, 'hit', 'zero', o('t1.json'), o('t1.txt')] + ex, env)]
    for name, cmd, e in steps:
        t = gate(cmd, e); log.append((name, 'exit 0, ttu_one: no error')); print(name, 'ok', flush=True)
    for tag in ('s1', 't1'):
        a = sh([VENV, S3 + '/analyze.py', o(tag + '.json'), '1']); print('analyze', tag, a.stdout.strip().splitlines()[-3:], flush=True)
        if a.returncode: die('analyze ' + tag, a)
    rep = open(o('t1.txt')).read(); print(rep[:3000]); open(o('check_summary.txt'), 'x').write(rep)
    sys.path.insert(0, HERE); import timing_fix
    tf_dir = os.path.join(outdir, 'timing_fix_' + ts); timing_fix.main(tf_dir)


if __name__ == '__main__':
    cmd, work = sys.argv[1], sys.argv[2]
    if cmd == 'prepare': prepare(work)
    elif cmd == 'run': run(work, sys.argv[3])
    elif cmd == 'clean': sh(['rm', '-rf', work, '/dev/shm/bittu'])
    else: sys.exit(__doc__)
