"""Cctl_1: TTUCCTL as a profiler (worked example) and TTUCCTL variants (prepare only). Predictions: model/predictions_cctl_profiler.md (committed first).

  cctl1.py prep            CPU only: host, ptx, words, the 3 builds x stage 0 and 1 (check_seq ALL PASS on all 6) in /dev/shm/cctl1
  cctl1.py stage0          GPU: stage 0 of single, twice, cctl (no TTU instruction), 30 launches each
  cctl1.py chk             GPU: ONE stage 1 launch per arm, N 16 then N 1024 (validated builds only)
  cctl1.py run OUTDIR REPS GPU: the matrix, sizes 1024 2048 4096, arms single twice cctl, REPS x 30 launches per cell, rotating order
  cctl1.py variants        CPU only: variant binaries V1..V4 (16 byte patch of the checked cctl stage 1 build), wrapper checks, check_seq failures listed
  cctl1.py clean           delete /dev/shm/cctl1 (patched builds hold NVIDIA code)
Safety as CLAUDE.md: run.py gate (idle GPU, check_seq marker, timeout -k 2 30), smallest launch first, STOP at the first bad exit or fault.
The variants have no .checked marker, so run.py refuses them: this script never runs a variant.
"""
import hashlib, json, os, shutil, statistics as st, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
S3 = os.path.join(ROOT, 'ttu', 'harness', 'setup3'); D = '/dev/shm/cctl1'; VENV = os.path.expanduser('~/eoss/.venv/bin/python')
OPTIX = os.path.expanduser('~/optix_sdk/include'); NVCC = os.path.expanduser('~/opt/cuda-12.8/bin/nvcc')
sys.path.insert(0, os.path.join(ROOT, 'ttu', 'config')); sys.path.insert(0, os.path.join(ROOT, 'notes', 'setup3'))
C = ['--clock', '--remap', '--compact-out', '--regs', '104', '--smfilter']
BUILDS = {'single': C, 'twice': C + ['--twice'], 'cctl': C + ['--twice', '--cctl']}
SIZES = [1024, 2048, 4096]
def sh(c, **kw): return subprocess.run(c, capture_output=True, text=True, **kw)
def die(m, p=None): sys.exit('STOP: ' + m + (('\n' + p.stdout[-1500:] + p.stderr[-1000:]) if p is not None else ''))
def sha(p): return hashlib.sha256(open(p, 'rb').read()).hexdigest()
def check(p):
    c = sh([VENV, os.path.join(S3, 'check_seq.py'), p]); ok = bool(c.stdout.strip()) and 'ALL PASS' in c.stdout.strip().splitlines()[-1]; return ok, c
def prep():
    os.makedirs(D, exist_ok=True); env = dict(os.environ)
    c = sh([NVCC, '-O2', '-arch=sm_89', '-I' + OPTIX, '-o', D + '/setup3', S3 + '/setup3.cu', '-lcuda', '-lnvidia-ml']); c.returncode and die('host build', c)
    c = sh([NVCC, '-O2', '-arch=sm_89', '-ptx', '-I' + OPTIX, '-o', D + '/setup3_dev.ptx', S3 + '/setup3_dev.cu']); c.returncode and die('ptx build', c)
    import store_words
    p0, a, b = store_words.store_words(0, 0xFF); open(D + '/words.txt', 'w').write(' '.join('%08x' % x for x in a + b) + '\n')
    for n, fl in BUILDS.items():
        for stg in (0, 1):
            out = '%s/%s_s%d' % (D, n, stg); p = sh([VENV, os.path.join(S3, 'build_patch.py'), '--stage', str(stg), '--bin', D + '/setup3', '--out', out] + fl); p.returncode and die('build %s %d' % (n, stg), p)
            ok, c = check(out); print('%-6s stage %d check_seq %s sha %s' % (n, stg, 'ALL PASS' if ok else 'FAIL', sha(out)[:8]), flush=True); ok or die('check_seq %s %d' % (n, stg), c)
def gpu_env():
    import gpucfg
    if os.environ.get('TTU_GPU') != '1': die('set TTU_GPU=1 (triton: GPU 1 only, CLAUDE.md); gpucfg alone picked GPU %d' % gpucfg.load().index)
    e = dict(os.environ); e.pop('CUDA_VISIBLE_DEVICES', None); return e, gpucfg.load().sm_count
def run(arm, N, stage, rep, tag):
    env, _ = gpu_env(); binp = '%s/%s_s%d' % (D, arm, stage); out = '%s/out/%s_%d.json' % (D, tag, os.getpid()); os.makedirs(D + '/out', exist_ok=True)
    cmd = [VENV, os.path.join(S3, 'run.py'), binp, 'hit', 'zero', out, '--scene', 'par:%d:s' % N, '--warps', '1', '--blocks', '1', '--sms', '0,0', '--repeat', str(rep)]
    o = sh(cmd, env=env); t = o.stdout + o.stderr
    if 'ttu_one: no error' not in t or 'exit 0' not in t or ' results same every launch: 1' not in t: die('%s N %d stage %d' % (arm, N, stage), o)
    L = json.load(open(out))['launches']; os.remove(out)
    if not all(l['ran'] == 1 for l in L): die('not exactly one TTU block %s N %d: %s' % (arm, N, [l['ran'] for l in L][:6]))
    return [l['warp_clk'][0][1] - l['warp_clk'][0][0] for l in L]
def stage0():
    for a in BUILDS: run(a, 1024, 0, 30, 's0'); print('stage 0', a, '30 launches ok', flush=True)
def chk():
    for a in BUILDS: print('stage 1 check', a, 'N 16', int(st.median(run(a, 16, 1, 1, 'c'))), flush=True)
    for a in BUILDS: print('stage 1 check', a, 'N 1024', int(st.median(run(a, 1024, 1, 1, 'c'))), flush=True)
def matrix(outd, reps):
    os.makedirs(outd, exist_ok=True); fn = '%s/cctl1_%s.json' % (outd, time.strftime('%Y%m%dT%H%M%S'))
    if os.path.exists(fn): die('exists ' + fn)
    cells = [(a, N) for N in SIZES for a in BUILDS]; res = {}
    for r in range(int(reps)):
        for k in range(len(cells)):
            a, N = cells[(k + r * 3) % len(cells)]; res.setdefault('%s|%d' % (a, N), []).append(run(a, N, 1, 30, 'cell'))
        print('rep', r, 'done', flush=True)
    with open(fn, 'x') as f: json.dump(res, f)
    print('raw', fn)
    for N in SIZES:
        med = {a: st.median([c for x in res['%s|%d' % (a, N)] for c in x]) for a in BUILDS}
        per = [1 - st.median(res['twice|%d' % N][r]) / st.median(res['cctl|%d' % N][r]) for r in range(int(reps))]
        print('N %5d single %d twice(keep) %d cctl(flush) %d  flush/single %.4f  C = 1 - keep/flush %.3f  per rep %s' % (N, med['single'], med['twice'], med['cctl'], med['cctl'] / med['single'], 1 - med['twice'] / med['cctl'], ' '.join('%.3f' % x for x in per)))
def variants():
    sys.path.insert(0, ROOT); sys.path.insert(0, S3)
    import build_patch, struct
    base = D + '/cctl_s1'; ok, c = check(base); ok or die('baseline check_seq', c)
    B = open(base, 'rb').read(); cc = build_patch.ttuccttl_bytes(); assert B.count(cc) == 1, 'TTUCCTL bytes not found exactly once'; off = B.index(cc)
    lo, hi = struct.unpack('<QQ', cc); print('baseline TTUCCTL at file offset 0x%x: lo %016x hi %016x' % (off, lo, hi))
    def pred(lo, p, neg): return (lo & ~(0xF << 12)) | ((p | (8 if neg else 0)) << 12)   # Pg bits 14:12, Pg_not bit 15
    V = {'V1_ivall0': (lo, hi & ~(1 << (80 - 64)), 'ivall (bit 80) = 0'),
         'V2_predoff': (pred(lo, 7, True), hi, 'predicate @!PT (Pg 7, not 1), never executes'),
         'V3_srcrel0': (lo, (hi & ~(7 << (113 - 64))) | (0 << (113 - 64)), 'src_rel_sb (115:113) = 0 (was 7)'),
         'V4_dstwr0': (lo, (hi & ~(7 << (110 - 64))) | (0 << (110 - 64)), 'dst_wr_sb (112:110) = 0 (was 7): the paper reports a FAULT; never run')}
    s0 = D + '/cctl_s0'; rep = []
    for n, (l2, h2, what) in V.items():
        nb = struct.pack('<QQ', l2, h2); out = '%s/%s_s1' % (D, n)
        with open(out, 'wb') as f: f.write(B[:off] + nb + B[off + 16:])
        shutil.copy(base + '.layout.json', out + '.layout.json')
        if os.path.exists(out + '.checked'): os.remove(out + '.checked')
        A = open(out, 'rb').read(); d = [i for i in range(len(B)) if A[i] != B[i]]
        d_ok = bool(d) and min(d) >= off and max(d) < off + 16 and len(A) == len(B)
        x = (struct.unpack('<QQ', nb)[0] ^ lo, struct.unpack('<QQ', nb)[1] ^ hi); bits = [i for i in range(64) if x[0] >> i & 1] + [64 + i for i in range(64) if x[1] >> i & 1]
        fails = []
        ck, c = check(out)
        if not ck: fails = [l.split('FAIL')[0].strip() for l in c.stdout.splitlines() if ' FAIL' in l and not l.startswith('FAILED')]
        print('%s: %s; differs from baseline only in bytes %s (inside the one TTUCCTL: %s); changed bits %s; check_seq %s; failing rules %s' % (n, what, '0x%x..0x%x' % (min(d), max(d)), d_ok, bits, 'ALL PASS' if ck else 'FAILS (expected, see plan)', fails))
        d_ok or die('variant differs outside the TTUCCTL: ' + n); rep.append((n, bits, ck, fails))
        EXP = ('sha256 matches build', 'cctl: exactly one TTUCCTL, ivall bit 80 set, unpredicated', 'cctl: bytes equal the EoSS encoder output for TTUCCTL.IVALL')
        print('  only the expected rules fail (sha of the baseline, the TTUCCTL bytes and its predicate/ivall rule): %s' % all(f in EXP for f in fails))
        if ck: print('  note: check_seq passed a variant (marker written): deleting it'); os.remove(out + '.checked')
    print('stage 0 of the variants is the baseline stage 0 binary (the slot is a NOP): sha %s' % sha(s0)[:8])
def clean():
    shutil.rmtree(D, ignore_errors=True); print('removed', D)
if __name__ == '__main__':
    a = sys.argv[1:]; {'prep': prep, 'stage0': stage0, 'chk': chk, 'variants': variants, 'clean': clean}.get(a[0], lambda: matrix(a[1], a[2]))() if a else print(__doc__)
