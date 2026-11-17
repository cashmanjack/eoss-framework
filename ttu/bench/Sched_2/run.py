"""Sched_2 runner (predictions: model/predictions_sched.md). One TTU warp (checked Unit_1 builds t1r_s0 = no TTU instruction, t1r_s1 = the checked group, both check_seq ALL PASS) with and without the harness's ALU compute kernel
(--compute alu:all: 2 blocks of 384 threads on every SM = 6 always ready warps per scheduler) on a second stream. Scene par:16:s, ray hit, 1 warp, 30 launches (first dropped) per run, 5 repetitions, arm order rotated.
    run.py WORK [first|stage0|runs]      first: the one smallest TTU launch (stage 1, 1 warp, 1 launch, no compute); stage0: stage 0 with compute (placement and sanity); runs: the matrix, stops at the first failure
Raw JSON per run: WORK/s2_<arm>_r<rep>_<timestamp>.json (refuses to overwrite). Needs TTU_GPU, CUDA_DEVICE_ORDER=PCI_BUS_ID and CUDA_VISIBLE_DEVICES unset (gpucfg)."""
import json, os, statistics as st, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); S3 = os.path.join(HERE, '..', '..', 'harness', 'setup3')
B0, B1 = '/dev/shm/t9u/t1r_s0', '/dev/shm/t9u/t1r_s1'; TS = time.strftime('%Y%m%dT%H%M%S'); SC = 'par:16:s'
ARMS = [('s1_alone', B1, []), ('s1_alu', B1, ['--compute', 'alu:all']), ('s0_alone', B0, []), ('s0_alu', B0, ['--compute', 'alu:all'])]


def go(binp, out, extra, repeat):
    if os.path.exists(out): sys.exit('exists ' + out)
    p = subprocess.run([sys.executable, os.path.join(S3, 'run.py'), binp, 'hit', 'zero', out, '--scene', SC, '--warps', '1', '--repeat', str(repeat)] + extra, capture_output=True, text=True)
    ok = 'exit 0' in p.stdout and 'ttu_one: no error' in p.stdout and '1 block(s) ran' in p.stdout and 'GPU' in p.stdout
    print('   ', os.path.basename(out), 'OK' if ok else 'FAILED', flush=True)
    if not ok: sys.exit('STOP:\n' + p.stdout[-2500:] + p.stderr[-1500:])
    return json.load(open(out))


def lat(j): return [l['warp_clk'][0][1] - l['warp_clk'][0][0] for l in j['launches'][1:]]


if __name__ == '__main__':
    work, step = sys.argv[1], sys.argv[2]; os.makedirs(work, exist_ok=True)
    if step == 'first': go(B1, '%s/s2_first_s1_alone_%s.json' % (work, TS), [], 1)
    elif step == 'stage0': j = go(B0, '%s/s2_stage0_alu_%s.json' % (work, TS), ['--compute', 'alu:all'], 5); print('latencies', lat(j))
    else:
        res = {a[0]: [] for a in ARMS}
        for rep in range(5):
            order = ARMS[rep % 4:] + ARMS[:rep % 4]
            for name, b, ex in order:
                j = go(b, '%s/s2_%s_r%d_%s.json' % (work, name, rep, TS), ex, 30); res[name].append(st.median(lat(j))); print('      median latency %.1f cycles' % res[name][-1], flush=True)
        for n, v in res.items(): print(n, [round(x, 1) for x in v], 'median of medians %.1f' % st.median(v))
