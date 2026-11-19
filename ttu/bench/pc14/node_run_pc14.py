"""PC14 (RTX 5090, SM120) port of Node_4090/node_run.py, 2026-11-19. Predictions model/predictions_5090_batch4.md. Needs PC14_TTU_APPROVED, TTU_GPU=0; every run goes through run.py (pc14_gpucheck.sh immediately before, timeout 30). Raw per run JSON kept in OUTDIR/raw.
Node state on the 4090 (predictions model/predictions_phase10f.md, item 4). node_run.py OUTDIR PHASE   PHASE: build | stage0 | chk | cap REPS | arms REPS
Replicates F150 (capacity curve: single, twice, cctl), F159 (fences), F161/F162 (TTUCCTL on the sibling / own SM), F174 (own L1 stream) with the SAME checked group; arm (h) (TTUCCTL alone on the sibling, build --sibccctl) is only in the NODE_H=1 rerun (after the B15 change of 2026-11-18); it was NOT in the first run: check_seq's B15 rule rejects that stage 1 build (rule written for --ownccctl, 2026-11-18 finding), and relaxing the checker was refused, so F161 is skipped here; flags reconstructed from the build_patch assertions (not recorded in a script on triton).
Safety: builds need check_seq ALL PASS (run.py refuses otherwise); stage 0 of every build before any stage 1; smallest launch first; run.py's timeout 30; first idle GPU; STOP at the first bad exit, a launch where not exactly one TTU block ran, a sibling timeout or results differing between launches. Raw JSON OUTDIR/node_<phase>_<TS>.json (mode x)."""
import json, os, statistics as st, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); S = os.path.join(HERE, '..', '..', 'harness', 'setup3'); sys.path.insert(0, os.path.join(HERE, '..', '..', 'config'))
import gpucfg  # noqa: E402
NSM = gpucfg.load().sm_count; D = '/dev/shm/t9sm120/n'; VENV = sys.executable; BASE = D + '/setup3'
DUMP = os.path.join(HERE, '..', '..', '..', 'notes', 'setup4_sm120', 'dump_sm120_fn-0x20000_len0x20000.bin'); RAW = None
C = ['--clock', '--remap', '--compact-out', '--regs', '104', '--smfilter']
BUILDS = {'single': C, 'twice': C + ['--twice'], 'cctl': C + ['--twice', '--cctl'], 'sib': C + ['--twice', '--sibling'], 'sibown': C + ['--twice', '--sibling', '--ownfence'],
          'sibO': C + ['--twice', '--sibling', '--ownccctl'], 'sibh': C + ['--twice', '--sibling', '--sibccctl']}
# arm: (build, sibling spec, blocks, sms, env)
ARMS = {'single': ('single', None, 1, '0,0', {}), 'twice': ('twice', None, 1, '0,0', {}), 'cctl': ('cctl', None, 1, '0,0', {}),
        'b1': ('sib', '1:4096:0:0:2', 1, '0,0', {}), 'e_cta': ('sib', '1:4096:0:0:4', 1, '0,0', {}), 'c1_gpu': ('sib', '1:4096:0:0:3', 1, '0,0', {}),
        'f_own': ('sibown', '1:4096:0:0:2', 1, '0,0', {}), 'g_both': ('sibown', '1:4096:0:0:3', 1, '0,0', {}),
        'i_ownccctl': ('sibO', '1:4096:0:0:2', 1, '0,0', {}), 'h_sibccctl': ('sibh', '1:4096:0:0:5', NSM, '0,1', {}),
        'own8_stream': ('sib', '0:1024:1:0:8', NSM, '0,0', {}), 'own9_wait': ('sib', '0:1024:1:0:9', NSM, '0,0', {'SIB_DNS': '70000'})}
if os.environ.get('NODE_H'):   # F161 rerun after the B15 change (Jack, 2026-11-18): only single, twice, b1 (sib) and h, as in notes/q22/fence_h.py
    ARMS = {k: ARMS[k] for k in ('single', 'twice', 'b1', 'h_sibccctl')}; BUILDS = {k: BUILDS[k] for k in ('single', 'twice', 'sib', 'sibh')}
CAP_N = [16, 64, 256, 1024, 2048, 3072, 4096, 6144, 8192, 16384, 65536]; CAP_ARMS = ['single', 'twice', 'cctl']; MAIN_N = [1024, 2048]
def sh(c, **kw): return subprocess.run(c, capture_output=True, text=True, **kw)
def build(stage, name):
    out = '%s/%s_s%d' % (D, name, stage); p = sh([VENV, os.path.join(S, 'build_patch.py'), '--stage', str(stage), '--sm', '120', '--bin', BASE, '--out', out] + BUILDS[name])
    if p.returncode: sys.exit('STOP build %s %d\n%s%s' % (name, stage, p.stdout[-800:], p.stderr[-800:]))
    c = sh([VENV, os.path.join(S, 'check_seq.py'), out, DUMP])
    if not c.stdout.strip() or 'ALL PASS' not in c.stdout.splitlines()[-1]: sys.exit('STOP check_seq %s %d\n%s%s' % (name, stage, c.stdout[-1200:], c.stderr[-500:]))
def run(arm, N, stage, rep, tag):
    b, sib, blocks, sms, env = ARMS[arm]; binp = '%s/%s_s%d' % (D, b, stage); out = '%s/out/%s_%d.json' % (D, tag, os.getpid())
    cmd = [VENV, os.path.join(S, 'run.py'), binp, 'hit', 'zero', out, '--scene', 'par:%d:s' % N, '--warps', '1', '--blocks', str(blocks), '--sms', sms, '--repeat', str(rep), '--sm120'] + (['--sibling', sib] if sib else [])
    e = dict(os.environ, TTU_GPU=str(gpucfg.load().index), PC14_TTU_APPROVED='cesar_jack_2026-11-19', **env); e.pop('CUDA_VISIBLE_DEVICES', None); e.pop('SIB_SM2', None)
    o = sh(cmd, env=e); t = o.stdout + o.stderr
    if 'ttu_one: no error' not in t or 'exit 0' not in t or ' results same every launch: 1' not in t: sys.exit('STOP %s N %d stage %d\n%s' % (arm, N, stage, t[-900:]))
    L = json.load(open(out))['launches']; os.makedirs(RAW, exist_ok=True); __import__("shutil").move(out, '%s/%s_%s_N%d_s%d_%d.json' % (RAW, tag, arm, N, stage, int(time.time() * 1000)))
    if not all(l['ran'] == 1 for l in L): sys.exit('STOP not exactly one TTU block %s N %d: %s' % (arm, N, [l['ran'] for l in L][:6]))
    if sib and any(l.get('sib', {}).get('timedout') for l in L): sys.exit('STOP sibling timeout %s N %d' % (arm, N))
    return [l['warp_clk'][0][1] - l['warp_clk'][0][0] for l in L]
def main(outd, phase, reps=3):
    global RAW; RAW = outd + '/raw'; os.makedirs(outd, exist_ok=True); os.makedirs(D, exist_ok=True); os.makedirs(D + '/out', exist_ok=True); reps = int(reps)
    if phase == 'build':
        for f in ('setup3', 'setup3_dev.ptx', 'words.txt'): sh(['cp', '/dev/shm/t9sm120/' + f, D + '/' + f])
        for n in BUILDS:
            for stg in (0, 1): build(stg, n)
        print('built and checked', list(BUILDS)); return
    fn = '%s/node_%s_%s.json' % (outd, phase, time.strftime('%Y%m%dT%H%M%S'))
    if os.path.exists(fn): sys.exit('exists')
    res = {}
    if phase == 'stage0':
        for a in ARMS: c = run(a, 1024, 0, 2, 's0'); print('stage 0', a, 'ok', flush=True)
        return
    if phase == 'chk':   # one stage 1 launch per arm, smallest scene first (N 1,024 is the smallest used by the arms; cap N 16 first for the three capacity arms)
        for a in [x for x in CAP_ARMS if x in ARMS]: print('stage 1 check', a, 'N 16', run(a, 16, 1, 1, 'c'), flush=True)
        for a in ARMS: print('stage 1 check', a, 'N 1024', run(a, 1024, 1, 1, 'c'), flush=True)
        return
    cells = [(a, N) for N in (CAP_N if phase == 'cap' else MAIN_N) for a in (CAP_ARMS if phase == 'cap' else ARMS)]
    for r in range(reps):
        for k in range(len(cells)):
            a, N = cells[(k + r * 3) % len(cells)]   # rotate the start each repetition
            res.setdefault('%s|%d' % (a, N), []).append(run(a, N, 1, 30, 'cell'))
        print('rep', r, 'done', flush=True); json.dump(res, open(fn + '.tmp', 'w'))
    os.rename(fn + '.tmp', fn)
    for N in (CAP_N if phase == 'cap' else MAIN_N):
        row = {a: st.median([c for x in res['%s|%d' % (a, N)] for c in x]) for a in (CAP_ARMS if phase == 'cap' else ARMS)}
        print('N %6d' % N, ' '.join('%s %d (s %.2f)' % (a, v, 1 - v / row['single']) for a, v in row.items() if a != 'single'), 'single', int(row['single']))
if __name__ == '__main__': main(*sys.argv[1:])
