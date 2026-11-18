"""BITTU compute mode on the GPU gpucfg selects: many warps (GAS group) and the IAS group (written 2026-11-18, Phase 14; predictions model/priorworks/predictions_phase14.md).

    validate_gpu2.py prepare WORK_GAS WORK_IAS     CPU only: host, ptx, words, builds (check_seq ALL PASS on every build), the NVBit tool
    validate_gpu2.py run WORK_GAS WORK_IAS OUTDIR  GPU: checked groups only, run.py gate (idle GPU, marker, timeout -k 2 30), smallest launch first, STOP at the first problem
    validate_gpu2.py clean WORK_GAS WORK_IAS
Run with ~/eoss/.venv/bin/python after `source ~/cuda128.sh`. Order: GAS group with 2, 4, 8, 16 warps (one block, scene tri1, ray hit), then the IAS group, one warp, rays hit and miss.
Per case: stage 0 plain, stage 0 twin under the tool (empty probe pair), stage 1 plain, stage 1 under the tool, each first with 1 launch, then 5 launches (F122 method). Checks: instances = 32 x warps and full masks in every launch,
analyze.py on plain and instrumented stage 1 output, corrected group time = window(stage 1 under tool) - window(twin under tool) against the harness clock (stage 1 minus stage 0, median over warps).
The patched builds contain NVIDIA code: they stay in /dev/shm and are deleted by `clean`. Raw outputs: OUTDIR, timestamped, never overwritten."""
import json, os, re, statistics as st, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
S3 = ROOT + '/ttu/harness/setup3'; VENV = os.path.expanduser('~/eoss/.venv/bin/python'); PY = '/usr/bin/python3'
OPTIX = os.path.expanduser('~/optix_sdk/include'); NVCC = os.path.expanduser('~/opt/cuda-12.8/bin/nvcc'); IASDUMP = ROOT + '/notes/q3/ias/ias_a.bin'
sys.path.insert(0, HERE); import validate_gpu as v1


def sh(c, **kw): return subprocess.run(c, capture_output=True, text=True, **kw)
def stop(msg, p=None): sys.exit('STOP: ' + msg + (('\n' + p.stdout[-2500:] + p.stderr[-1500:]) if p is not None else ''))


def check(out, extra=()):
    k = sh([VENV, S3 + '/check_seq.py', out] + list(extra))
    if 'ALL PASS' not in (k.stdout.strip().splitlines() or [''])[-1]: stop('check_seq ' + out, k)
    print('check_seq ALL PASS', out, flush=True)


def prepare(wg, wi):
    for w in (wg, wi): os.makedirs(w, exist_ok=True)
    c = sh([NVCC, '-O2', '-arch=sm_89', '-I' + OPTIX, '-o', wg + '/setup3', S3 + '/setup3.cu', '-lcuda', '-lnvidia-ml']); c.returncode and stop('host', c)
    c = sh([NVCC, '-O2', '-arch=sm_89', '-ptx', '-I' + OPTIX, '-o', wg + '/setup3_dev.ptx', S3 + '/setup3_dev.cu']); c.returncode and stop('ptx', c)
    v1.words_txt(wg + '/words.txt')
    for f in ('setup3', 'setup3_dev.ptx'): sh(['cp', wg + '/' + f, wi + '/' + f])
    c = sh([VENV, S3 + '/ias_words.py', IASDUMP, wi + '/words.txt']); c.returncode and stop('ias_words', c)
    for st_ in (0, 1):
        o = '%s/m_s%d' % (wg, st_); c = sh([VENV, S3 + '/build_patch.py', '--stage', str(st_), '--bin', wg + '/setup3', '--clock', '--regs', '104', '--out', o]); c.returncode and stop('build_patch GAS', c); check(o)
        o = '%s/ias_s%dc' % (wi, st_); c = sh([VENV, S3 + '/build_patch.py', '--stage', str(st_), '--bin', wi + '/setup3', '--dump', IASDUMP, '--ias', '--clock', '--out', o]); c.returncode and stop('build_patch IAS', c); check(o, [IASDUMP])
    c = sh([HERE + '/build.sh', '/dev/shm/bittu']); c.returncode and stop('tool', c); print(c.stdout.strip())


def gate(cmd, env):
    p = sh(cmd, env=env)
    if p.returncode or 'ttu_one: no error' not in p.stdout or 'exit 0' not in p.stdout: stop('run failed ' + ' '.join(cmd[:5]), p)


def clk(j, w):
    L = json.load(open(j))['launches'][1:] or json.load(open(j))['launches']
    return st.median(st.median(e - s for s, e in l['warp_clk'][:w]) for l in L)


def win(txt):
    t = open(txt).read(); m = [int(x) for x in re.findall(r'median (\d+), probe overhead', t)]; o = [int(x) for x in re.findall(r'overhead o \(t0b - t0a\): median (\d+)', t)]
    return (st.median(m[1:] or m), st.median(o[1:] or o))


def case(tag, B0, B1, ex, w, outdir, ts, ray):
    env = dict(os.environ, TTU_GPU=os.environ.get('TTU_GPU', '1'), BITTU_PRINT='0'); env.pop('CUDA_VISIBLE_DEVICES', None); res = {}
    for rep in (1, 5):
        e = ex + ['--repeat', str(rep)]; o = lambda n: os.path.join(outdir, 'b6000m_%s_r%d_%s_%s' % (tag, rep, n, ts))
        gate([PY, S3 + '/run.py', B0, ray, 'zero', o('s0.json')] + e, env)
        gate([VENV, HERE + '/run_bittu.py', B0, ray, 'zero', o('t0.json'), o('t0.txt')] + e, dict(env, BITTU_LIKE=B1))
        gate([PY, S3 + '/run.py', B1, ray, 'zero', o('s1.json')] + e, env)
        gate([VENV, HERE + '/run_bittu.py', B1, ray, 'zero', o('t1.json'), o('t1.txt')] + e, env)
        for t in ('s1', 't1'):
            a = sh([VENV, S3 + '/analyze.py', o(t + '.json'), '1'] + (['--ias'] if False else []))
            if a.returncode or 'PASS' not in a.stdout: stop('analyze ' + t, a)
        rep_t = open(o('t1.txt')).read(); L = re.findall(r'launch \d+: (\d+) threads ran the group, (\d+) instances, (\d+) with a full warp mask, (\d+) with clocks out of order', rep_t)
        ok = bool(L) and all(int(a) == 32 * w and int(b) == 32 * w and int(c) == int(b) and int(d) == 0 for a, b, c, d in L)
        decode = re.search(r'(\d+) instructions: (decoded, complete fused group)', rep_t); fuse = re.search(r'TTUMACROFUSE (\d+)', rep_t); close = '.CLOSE (bit 74)' in rep_t
        w1, o1 = win(o('t1.txt')); w0, o0 = win(o('t0.txt')); h = clk(o('s1.json'), w) - clk(o('s0.json'), w)
        res[rep] = dict(counts_ok=ok, launches=len(L), decoded=bool(decode), fuse=fuse and int(fuse[1]), close_bit74_last=close, window_s1=w1, window_twin=w0, probe_cost=o1, harness_group=h, corrected=w1 - w0, corrected_over_harness=(w1 - w0) / h)
        print(tag, rep, json.dumps(res[rep]), flush=True)
        if not ok or not decode: stop('%s rep %d: counts or decode not as expected (see report)' % (tag, rep))
    return res


def run(wg, wi, outdir):
    os.makedirs(outdir, exist_ok=True); ts = time.strftime('%Y%m%dT%H%M%S'); allr = {}
    for b in (wg + '/m_s0', wg + '/m_s1', wi + '/ias_s0c', wi + '/ias_s1c'):
        if not os.path.exists(b + '.checked'): stop('no check marker ' + b)
    for w in (2, 4, 8, 16):
        allr['gas_w%d' % w] = case('gas_w%d' % w, wg + '/m_s0', wg + '/m_s1', ['--scene', 'tri1', '--warps', str(w)], w, outdir, ts, 'hit')
    for ray in ('hit', 'miss'):
        allr['ias_' + ray] = case('ias_' + ray, wi + '/ias_s0c', wi + '/ias_s1c', ['--ias', '--scene', 'tri1', '--warps', '1'], 1, outdir, ts, ray)
    json.dump(allr, open(os.path.join(outdir, 'summary_%s.json' % ts), 'x'), indent=1)


if __name__ == '__main__':
    c = sys.argv[1]
    if c == 'prepare': prepare(sys.argv[2], sys.argv[3])
    elif c == 'run': run(sys.argv[2], sys.argv[3], sys.argv[4])
    elif c == 'clean': sh(['rm', '-rf', sys.argv[2], sys.argv[3], '/dev/shm/bittu'])
