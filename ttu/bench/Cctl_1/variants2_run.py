"""Catalogue items 1 to 7 stage 1 runner (approved by Jack 2026-11-24). One variant per call, in the order given.
  variants2_run.py OUTDIR NAME [NAME ...]        names from variants2.py
Per variant: preflight (nvidia-smi compute process PIDs on both GPUs and their owners, `who`; run only if no compute process of anyone is present, else wait up to 10 min), one launch
at 1,024 triangles under timeout -k 2 30, classify (ran correctly | wrong result | fault | timeout or hang), then for a correct one 3 x 30 launches at 1,024 and 2,048 for the arms
single, twice, cctl and the variant. After any outcome that is not "ran correctly" a health check runs (stage 0 launch, compute process list empty, our own child gone); if it fails
the runner stops everything and exits 3. NO process matching by name anywhere (CLAUDE.md, 2026-11-24): only our own child PID (subprocess handle) and nvidia-smi PIDs.
Raw results are written with mode x."""
import json, os, pwd, signal, statistics as st, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..')); S3 = os.path.join(ROOT, 'ttu', 'harness', 'setup3')
D = '/dev/shm/cctl1'; VENV = os.path.expanduser('~/eoss/.venv/bin/python'); ENV = dict(os.environ, TTU_GPU='1'); ENV.pop('CUDA_VISIBLE_DEVICES', None)
BASE = {'F0_fuse0': 'cctl', 'P1': 'cctl', 'P2': 'cctl', 'P3': 'cctl', 'O1': 'cctl', 'O2': 'cctl', 'O3': 'cctl', 'O4': 'cctl', 'W1': 'cctl', 'W2': 'cctl', 'S1': 'twice', 'S2': 'twice', 'X1': 'twice', 'D1': 'twice'}
def compute_pids():
    o = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory', '--format=csv,noheader'], capture_output=True, text=True).stdout.strip().splitlines()
    r = []
    for l in o:
        pid = int(l.split(',')[0])
        try: user = pwd.getpwuid(os.stat('/proc/%d' % pid).st_uid).pw_name
        except Exception: user = '?'
        r.append((pid, user))
    return r
def preflight(tag):
    for _ in range(60):
        p = compute_pids()
        if not p:
            who = sorted({l.split()[0] for l in subprocess.run(['who'], capture_output=True, text=True).stdout.splitlines()})
            if ' ' not in tag: print('  preflight %s: no GPU compute process on either GPU; logged in: %s' % (tag, ','.join(who)), flush=True)
            return
        print('  GPU busy (pid,user):', p, 'waiting', flush=True); time.sleep(10)
    sys.exit('STOP: GPUs not free of compute processes for 10 minutes')
def run_one(binp, N, rep, jsonout):
    """returns (class, text). Our own child only."""
    cmd = ['timeout', '-k', '2', '30', VENV, os.path.join(S3, 'run.py'), binp, 'hit', 'zero', jsonout, '--scene', 'par:%d:s' % N, '--warps', '1', '--blocks', '1', '--sms', '0,0', '--repeat', str(rep)]
    pr = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=ENV, start_new_session=True)
    try: out, _ = pr.communicate(timeout=50)
    except subprocess.TimeoutExpired:
        os.killpg(pr.pid, signal.SIGKILL); out, _ = pr.communicate(); return 'timeout or hang', 'outer timeout 50 s: ' + (out or '')[-500:]
    if 'Permission denied' in out or 'No such file' in out or 'refus' in out.lower(): sys.exit('STOP harness error, no launch (not an outcome): ' + out[-400:])
    if pr.returncode in (124, 137): return 'timeout or hang', out[-600:]
    if 'ttu_one: no error' in out and 'exit 0' in out:
        return ('ran correctly' if ' results same every launch: 1' in out else 'wrong result'), out[-400:]
    return 'fault', out[-900:]
def health():
    jo = '%s/out/h_%d.json' % (D, os.getpid()); c, t = run_one(D + '/cctl_s0', 1024, 1, jo)
    time.sleep(2); left = compute_pids()
    return c == 'ran correctly' and not left, (c, left)
def cycles(binp, N, rep, tag):
    jo = '%s/out/%s_%d.json' % (D, tag, os.getpid()); c, t = run_one(binp, N, rep, jo)
    if c != 'ran correctly': return c, t, None
    L = json.load(open(jo))['launches']; os.remove(jo)
    if not all(l['ran'] == 1 for l in L): return 'wrong result', 'not exactly one TTU block', None
    return c, '', [l['warp_clk'][0][1] - l['warp_clk'][0][0] for l in L]
def results_equal(binp, basep):
    """one launch each: analyze.py PASS on the variant, and its result registers equal to the base build's except the words that differ between two base launches (addresses)."""
    ja, jb, jc = ['%s/out/cmp_%s_%d.json' % (D, k, os.getpid()) for k in 'abc']
    for b, j in ((basep, ja), (basep, jb), (binp, jc)):
        preflight('cmp'); c, t = run_one(b, 1024, 1, j)
        if c != 'ran correctly': return False, 'launch for comparison: %s' % c
    a, b_, v = [json.load(open(j))['out'] for j in (ja, jb, jc)]
    nd = [(l, k) for l in range(len(a)) for k in range(len(a[l])) if a[l][k] != b_[l][k]]
    mask = set(k for l, k in nd) | {84, 85, 86, 87};   # columns 84 to 87 are the clock stamps (high words can agree by chance between two base launches: this misclassified P3 once)
    dif = [(l, k) for l in range(len(a)) for k in range(len(a[l])) if k not in mask and a[l][k] != v[l][k]]
    an = subprocess.run([VENV, os.path.join(S3, 'analyze.py'), jc, '1'], capture_output=True, text=True); ok = (an.returncode == 0 and 'PASS' in an.stdout)
    for j in (ja, jb, jc): os.remove(j)
    return (ok and not dif), 'analyze %s; %d positions differ from base (not counting %d address columns)' % ('PASS' if ok else 'FAIL', len(dif), len(mask))
def main(outd, names):
    os.makedirs(outd, exist_ok=True); os.makedirs(D + '/out', exist_ok=True); fn = '%s/variants2_%s.json' % (outd, time.strftime('%Y%m%dT%H%M%S')); res = {}
    for n in names:
        binp = '%s/%s_s1' % (D, n); base = BASE[n]; r = {'base': base}; print('=== %s (base %s)' % (n, base), flush=True)
        preflight(n); cls, tail, cy = cycles(binp, 1024, 1, 'one'); r['one_launch'] = {'class': cls, 'cycles': cy, 'tail': tail[-600:]}; print(n, 'one launch at 1,024:', cls, cy, flush=True)
        if cls == 'ran correctly':
            same, info = results_equal(binp, '%s/%s_s1' % (D, base)); r['results'] = {'equal': same, 'info': info}; print(n, 'results vs base:', same, info, flush=True)
            if not same: cls = 'wrong result'
        if cls == 'ran correctly':
            for N in (1024, 2048):
                cs = {}
                for arm, b in (('single', D + '/single_s1'), ('twice', D + '/twice_s1'), ('cctl', D + '/cctl_s1'), (n, binp)):
                    allc = []
                    for rp in range(3):
                        preflight('%s %d %s' % (n, N, arm)); c2, t2, c = cycles(b, N, 30, 'm')
                        if c2 != 'ran correctly': r['matrix_stop'] = {'arm': arm, 'N': N, 'class': c2, 'tail': t2[-600:]}; cls = c2; break
                        allc += c
                    if cls != 'ran correctly': break
                    cs[arm] = st.median(allc)
                if cls != 'ran correctly': break
                r['N%d' % N] = cs; print(n, N, {k: int(v) for k, v in cs.items()}, 'saved %.3f (twice %.3f, cctl %.3f); variant/base cycles %.4f' % (1 - cs[n] / cs['single'], 1 - cs['twice'] / cs['single'], 1 - cs['cctl'] / cs['single'], cs[n] / cs[base]), flush=True)
        r['class'] = cls
        if cls != 'ran correctly':
            ok, info = health(); r['health_after'] = {'ok': ok, 'info': str(info)}; print(n, 'OUTCOME', cls, '| health after:', ok, info, flush=True)
            if not ok: res[n] = r; json.dump(res, open(fn, 'x')); sys.exit('STOP: health check failed after %s: %s. Tell Jack immediately.' % (n, info))
        res[n] = r
    json.dump(res, open(fn, 'x')); print('raw', fn)
if __name__ == '__main__': main(sys.argv[1], sys.argv[2:])
