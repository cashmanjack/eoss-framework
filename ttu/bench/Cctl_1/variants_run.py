"""TTUCCTL variants, stage 1 (approved by Jack 2026-11-24, including V4). One variant at a time, one warp, run.py's gate (marker by hash, idle GPU) under timeout -k 2 30.
  variants_run.py OUTDIR V2_predoff [V3_srcrel0 ...]
Before each run: nvidia-smi on both GPUs, only run if no compute process (anyone's) and no other user's process; else wait up to 10 minutes, then stop.
Per variant: one launch at 1,024 triangles; if it ran correctly, 3 x 30 launches at 1,024 and 2,048. Outcome classes: ran correctly | wrong result |
illegal instruction or other fault | timeout or hang. After any non-correct outcome a health check (baseline cctl stage 0 launch, 30 s) runs; if it fails or a
process is left, STOP everything. Raw output is written with mode x (never overwritten)."""
import json, os, statistics as st, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..')); S3 = os.path.join(ROOT, 'ttu', 'harness', 'setup3')
D = '/dev/shm/cctl1'; VENV = os.path.expanduser('~/eoss/.venv/bin/python'); ENV = dict(os.environ, TTU_GPU='1'); ENV.pop('CUDA_VISIBLE_DEVICES', None)
def sh(c, t=None, **kw):
    try: return subprocess.run(c, capture_output=True, text=True, timeout=t, **kw), False
    except subprocess.TimeoutExpired as e: return e, True
def idle():
    for _ in range(60):
        a = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,name', '--format=csv,noheader'], capture_output=True, text=True).stdout.strip()
        g = subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout
        procs = [l for l in g.splitlines() if l.startswith('|') and ' C ' in l or ' C+G ' in l]
        others = ''   # no process matching by name (CLAUDE.md, 2026-11-24): the compute process list of nvidia-smi above is the check
        if not a and not procs and not others: return True
        print('GPU busy, waiting:', a, procs, others, flush=True); time.sleep(10)
    sys.exit('STOP: GPUs not idle for 10 minutes')
def launch(binp, N, rep, stage_name):
    idle(); out = '%s/out/v_%d.json' % (D, os.getpid()); os.makedirs(D + '/out', exist_ok=True)
    cmd = [VENV, os.path.join(S3, 'run.py'), binp, 'hit', 'zero', out, '--scene', 'par:%d:s' % N, '--warps', '1', '--blocks', '1', '--sms', '0,0', '--repeat', str(rep)]
    p, to = sh(['timeout', '-k', '2', '30'] + cmd, 45, env=ENV)
    if to: return 'timeout or hang', 'python wrapper timeout (45 s)', None
    t = p.stdout + p.stderr
    if 'Permission denied' in t or 'No such file' in t: sys.exit('STOP harness error, no launch happened (not an outcome): ' + t[-300:])
    if p.returncode in (124, 137) or 'Terminated' in t: return 'timeout or hang', t[-600:], None
    if 'ttu_one: no error' in t and 'exit 0' in t:
        if ' results same every launch: 1' not in t: return 'wrong result', t[-600:], None
        L = json.load(open(out)); os.remove(out)
        if not all(l['ran'] == 1 for l in L['launches']): return 'wrong result', 'not exactly one TTU block: %s' % [l['ran'] for l in L['launches']][:5], None
        return 'ran correctly', t[-300:], [l['warp_clk'][0][1] - l['warp_clk'][0][0] for l in L['launches']]
    return 'illegal instruction or other fault', t[-900:], None
def analyze_ok(rep):   # one launch json is deleted above; correctness of the registers is checked by analyze.py on a separate single launch
    return True
def a_left():
    return subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], capture_output=True, text=True).stdout.strip()
def health():
    idle(); p, to = sh(['timeout', '-k', '2', '30', VENV, os.path.join(S3, 'run.py'), D + '/cctl_s0', 'hit', 'zero', D + '/out/h_%d.json' % os.getpid(), '--scene', 'par:1024:s', '--warps', '1', '--blocks', '1', '--sms', '0,0', '--repeat', '1'], 45, env=ENV)
    left = a_left()
    ok = (not to) and 'exit 0' in (p.stdout + p.stderr) and not left
    return ok, left
def main(outd, names):
    os.makedirs(outd, exist_ok=True); fn = '%s/variants_%s.json' % (outd, time.strftime('%Y%m%dT%H%M%S')); res = {}
    for n in names:
        binp = '%s/%s_s1' % (D, n); print('=== %s' % n, flush=True); r = {}
        cls, tail, c = launch(binp, 1024, 1, n); r['one_launch'] = {'class': cls, 'tail': tail, 'cycles': c}; print(n, 'one launch at N 1024:', cls, c, flush=True)
        if cls == 'ran correctly':
            out = '%s/out/an_%d.json' % (D, os.getpid())
            p, to = sh(['timeout', '-k', '2', '30', VENV, os.path.join(S3, 'run.py'), binp, 'hit', 'zero', out, '--scene', 'par:1024:s', '--warps', '1', '--blocks', '1', '--sms', '0,0', '--repeat', '1'], 45, env=ENV)
            a = subprocess.run([VENV, os.path.join(S3, 'analyze.py'), out, '1'], capture_output=True, text=True); r['analyze'] = a.stdout.strip().splitlines()[-1:] ; print(n, 'analyze:', r['analyze'], flush=True)
            if a.returncode or 'PASS' not in a.stdout: r['one_launch']['class'] = cls = 'wrong result'
        if cls == 'ran correctly':
            for N in (1024, 2048):
                cs = {}
                for arm, b in (('single', D + '/single_s1'), ('twice', D + '/twice_s1'), ('cctl', D + '/cctl_s1'), (n, binp)):
                    allc = []
                    for rp in range(3):
                        c2, t2, cy = launch(b, N, 30, n)
                        if c2 != 'ran correctly': r['matrix_stop'] = {'arm': arm, 'N': N, 'class': c2, 'tail': t2}; cls = c2; break
                        allc += cy
                    if cls != 'ran correctly': break
                    cs[arm] = st.median(allc)
                if cls != 'ran correctly': break
                r['N%d' % N] = cs; print(n, N, {k: int(v) for k, v in cs.items()}, 'saved fraction %.3f (twice %.3f, cctl %.3f)' % (1 - cs[n] / cs['single'], 1 - cs['twice'] / cs['single'], 1 - cs['cctl'] / cs['single']), flush=True)
        r['class'] = cls
        if cls != 'ran correctly':
            ok, left = health(); r['health_after'] = {'ok': ok, 'left': left}; print(n, 'outcome', cls, '| health check after:', ok, left, flush=True)
            if not ok: res[n] = r; json.dump(res, open(fn, 'x')); sys.exit('STOP: HANG NOT CLEARED after %s (%s). Tell Jack immediately.' % (n, left))
        res[n] = r
    with open(fn, 'x') as f: json.dump(res, f)
    print('raw', fn)
if __name__ == '__main__': main(sys.argv[1], sys.argv[2:])
