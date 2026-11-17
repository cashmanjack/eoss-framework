"""Sched_2 redone (predictions: model/predictions_sched2.md, committed before any run). ALU warps INSIDE the TTU block (build_patch --aluwarps), one block of W warps: one TTU warp (warp T) and W-1 ALU warps
(each: delay loop, clock, 47 iterations of 192 independent FFMA + 3 loop instructions, clock). Scene par:16:s, ray hit, 1 block, 30 launches (first dropped) per run, 5 repetitions, arm order rotated.
    run2.py WORK build               build the checked binaries in /dev/shm/s2 (stage 0 and 1 of every shape), needs /dev/shm/s2/setup3, setup3_dev.ptx, words.txt
    run2.py WORK stage0              every arm's stage 0 build once (no TTU instruction), 1 launch each, prints the ALU windows
    run2.py WORK first               the smallest stage 1 launch: W = 4, T = 0, 1 launch
    run2.py WORK ncu [stage]         counters (smsp__inst_executed, FFMA pipe) for W = 13, T = 0, one launch, stage 0 (default) or 1 (TMPDIR=NCU_TMP)
    run2.py WORK runs                the matrix, stops at the first failure
Raw JSON per run: WORK/s2b_<arm>_r<rep>_<timestamp>.json (refuses to overwrite). Needs TTU_GPU, CUDA_DEVICE_ORDER=PCI_BUS_ID and CUDA_VISIBLE_DEVICES unset."""
import json, os, statistics as st, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); S3 = os.path.join(HERE, '..', '..', 'harness', 'setup3'); D = '/dev/shm/s2'; TS = time.strftime('%Y%m%dT%H%M%S'); SC = 'par:16:s'
# arm name: (W, T); T = the TTU warp's index (0 = lowest id, W-1 = highest id)
ARMS = {'alone': (1, 0), 'lo4': (4, 0), 'lo5': (5, 0), 'lo9': (9, 0), 'lo13': (13, 0), 'lo29': (29, 0), 'hi5': (5, 4), 'hi13': (13, 12), 'hi29': (29, 28)}
SHAPES = sorted({a[1] for a in ARMS.values()})
BF = ['--clock', '--sync', '--remap', '--compact-out', '--regs', '64', '--max-out', '61']


def binp(T, st_):
    return '%s/b_t%d_s%d' % (D, T, st_)


def build():
    env = dict(os.environ); env['PATH'] = os.path.expanduser('~/opt/cuda-12.8/bin') + ':' + env['PATH']
    py = os.path.expanduser('~/eoss/.venv/bin/python')
    for T in SHAPES:
        for stg in (0, 1):
            out = binp(T, stg)
            if os.path.exists(out): print('exists', out); continue
            subprocess.run([py, os.path.join(S3, 'build_patch.py'), '--stage', str(stg), '--bin', D + '/setup3', '--out', out] + BF + ['--aluwarps', '--aluiters', '47', '--aluunroll', '192', '--aludelay', '16', '--ttuwarp', str(T)], check=True, env=env, capture_output=True)
            r = subprocess.run([py, os.path.join(S3, 'check_seq.py'), out], env=env, capture_output=True, text=True)
            print(out, r.stdout.strip().splitlines()[-1][:40]); assert 'ALL PASS' in r.stdout, r.stdout[-1500:]


def go(binary, out, W, repeat, extra=(), env_extra=None):
    if os.path.exists(out): sys.exit('exists ' + out)
    env = dict(os.environ, **(env_extra or {}))
    p = subprocess.run([sys.executable, os.path.join(S3, 'run.py'), binary, 'hit', 'zero', out, '--scene', SC, '--warps', str(W), '--repeat', str(repeat)] + list(extra), capture_output=True, text=True, env=env)
    ok = 'exit 0' in p.stdout and 'ttu_one: no error' in p.stdout and '1 block(s) ran' in p.stdout and 'GPU' in p.stdout
    print('   ', os.path.basename(out), 'OK' if ok else 'FAILED', flush=True)
    if not ok: sys.exit('STOP:\n' + p.stdout[-2500:] + p.stderr[-1500:])
    return json.load(open(out))


def windows(j, T):   # per launch: TTU warp window, ALU warp windows (start, end) relative to the TTU warp's start
    out = []
    for l in j['launches']:
        wc = l['warp_clk']; s0 = wc[T][0]
        out.append(dict(ttu=wc[T][1] - wc[T][0], alu=[(w[0] - s0, w[1] - s0) for i, w in enumerate(wc) if i != T]))
    return out


def lat(j, T): return [l['warp_clk'][T][1] - l['warp_clk'][T][0] for l in j['launches'][1:]]


if __name__ == '__main__':
    work, step = sys.argv[1], sys.argv[2]; os.makedirs(work, exist_ok=True)
    if step == 'build': build()
    elif step == 'stage0':
        for a, (W, T) in ARMS.items():
            if a == 'alone': continue
            j = go(binp(T, 0), '%s/s2b_stage0_%s_%s.json' % (work, a, TS), W, 1)
            w = windows(j, T)[-1]; print('     %s: warp T window %d cycles; ALU windows (start, end) rel. to T start, first 3: %s; n ALU %d' % (a, w['ttu'], w['alu'][:3], len(w['alu'])))
    elif step == 'first':
        go(binp(0, 1), '%s/s2b_first_lo4_%s.json' % (work, TS), 4, 1)
    elif step == 'ncu':
        stg = int(sys.argv[3]) if len(sys.argv) > 3 else 0; W, T = ARMS['lo13']; csv = '%s/s2b_ncu_stage%d_%s.csv' % (work, stg, TS)
        go(binp(T, stg), '%s/s2b_ncu_stage%d_%s.json' % (work, stg, TS), W, 1, ['--ncu', 'smsp__inst_executed.sum,sm__inst_executed_pipe_fma.sum,sm__inst_executed_pipe_alu.sum,smsp__issue_active.sum,sm__cycles_elapsed.max', '--ncu-out', csv], dict(TMPDIR=os.environ.get('NCU_TMP', '/dev/shm/hs3/ncutmp')))
        print(open(csv).read()[-1500:])
    else:
        res = {a: [] for a in ARMS}; names = list(ARMS)
        for rep in range(5):
            order = names[rep % len(names):] + names[:rep % len(names)]
            for a in order:
                W, T = ARMS[a]; j = go(binp(T, 1), '%s/s2b_%s_r%d_%s.json' % (work, a, rep, TS), W, 30); res[a].append(st.median(lat(j, T))); print('      median TTU warp latency %.1f cycles' % res[a][-1], flush=True)
        for n, v in res.items(): print(n, [round(x, 1) for x in v], 'median of medians %.1f' % st.median(v))
