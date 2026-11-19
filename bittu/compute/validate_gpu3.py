"""BITTU compute mode, probes on ONE warp while the other warps run without records (written 2026-11-19, Phase 15; predictions model/priorworks/predictions_phase15.md, item 1).

    validate_gpu3.py prepare WORK_GAS         CPU only: host, ptx, words, GAS builds (check_seq ALL PASS), the NVBit tool (with BITTU_WARP)
    validate_gpu3.py run WORK_GAS OUTDIR      GPU: checked GAS group only, run.py gate, smallest launch first, STOP at the first problem
    validate_gpu3.py clean WORK_GAS
Run with ~/eoss/.venv/bin/python after `source ~/cuda128.sh`.
BITTU_WARP=w makes the injected functions return at once for every warp except w (the other warps still pay the call and the register save and restore: NVBit cannot call per warp).
Cases: 1 block, GAS group, scene tri1, ray hit, 2, 4, 8, 16 warps, probed warp 0 and probed warp w-1. Per case (F122 method, same as validate_gpu2.py): stage 0 plain, stage 0 twin under the tool
(empty probe pair), stage 1 plain, stage 1 under the tool, each with 1 launch, then 5 launches. Corrected window = window(stage 1 under tool) - window(twin under tool) of the probed warp,
against the harness clock of the same warp (stage 1 minus stage 0, plain, median over launches 2 to 5). Also reported: the harness group time of the NON probed warps in the instrumented run
against the plain run (what the stubs of the other warps cost). The patched builds contain NVIDIA code: they stay in /dev/shm and are deleted by `clean`. Raw outputs: OUTDIR, timestamped, never overwritten."""
import json, os, re, statistics as st, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import validate_gpu2 as v2
from validate_gpu2 import sh, stop, check, gate, win, S3, VENV, PY, NVCC, OPTIX
v1 = v2.v1


def prepare(wg):
    os.makedirs(wg, exist_ok=True)
    c = sh([NVCC, '-O2', '-arch=sm_89', '-I' + OPTIX, '-o', wg + '/setup3', S3 + '/setup3.cu', '-lcuda', '-lnvidia-ml']); c.returncode and stop('host', c)
    c = sh([NVCC, '-O2', '-arch=sm_89', '-ptx', '-I' + OPTIX, '-o', wg + '/setup3_dev.ptx', S3 + '/setup3_dev.cu']); c.returncode and stop('ptx', c)
    v1.words_txt(wg + '/words.txt')
    for st_ in (0, 1):
        o = '%s/m_s%d' % (wg, st_); c = sh([VENV, S3 + '/build_patch.py', '--stage', str(st_), '--bin', wg + '/setup3', '--clock', '--regs', '104', '--out', o]); c.returncode and stop('build_patch GAS', c); check(o)
    c = sh([HERE + '/build.sh', '/dev/shm/bittu']); c.returncode and stop('tool', c); print(c.stdout.strip())


def per_warp(j, w):
    """median over launches 2.. of (end - start) per warp, list of w values"""
    d = json.load(open(j)); L = d['launches'][1:] or d['launches']
    return [st.median(l['warp_clk'][k][1] - l['warp_clk'][k][0] for l in L) for k in range(w)]


def case(tag, B0, B1, w, tw, outdir, ts):
    env = dict(os.environ, TTU_GPU=os.environ.get('TTU_GPU', '1'), BITTU_PRINT='0', BITTU_WARP=str(tw)); env.pop('CUDA_VISIBLE_DEVICES', None); res = {}
    ex = ['--scene', 'tri1', '--warps', str(w)]
    for rep in (1, 5):
        e = ex + ['--repeat', str(rep)]; o = lambda n: os.path.join(outdir, 'b6000w_%s_r%d_%s_%s' % (tag, rep, n, ts))
        envp = {k: v for k, v in env.items() if k != 'BITTU_WARP'}
        gate([PY, S3 + '/run.py', B0, 'hit', 'zero', o('s0.json')] + e, envp)
        gate([VENV, HERE + '/run_bittu.py', B0, 'hit', 'zero', o('t0.json'), o('t0.txt')] + e, dict(env, BITTU_LIKE=B1))
        gate([PY, S3 + '/run.py', B1, 'hit', 'zero', o('s1.json')] + e, envp)
        gate([VENV, HERE + '/run_bittu.py', B1, 'hit', 'zero', o('t1.json'), o('t1.txt')] + e, env)
        for t in ('s1', 't1'):
            a = sh([VENV, S3 + '/analyze.py', o(t + '.json'), '1'])
            if a.returncode or 'PASS' not in a.stdout: stop('analyze ' + t, a)
        rep_t = open(o('t1.txt')).read(); L = re.findall(r'launch \d+: (\d+) threads ran the group, (\d+) instances, (\d+) with a full warp mask, (\d+) with clocks out of order', rep_t)
        ok = bool(L) and all(int(a) == 32 and int(b) == 32 and int(c) == 32 and int(d) == 0 for a, b, c, d in L)
        decode = re.search(r'(\d+) instructions: (decoded, complete fused group)', rep_t); close = '.CLOSE (bit 74)' in rep_t
        w1, o1 = win(o('t1.txt')); w0, o0 = win(o('t0.txt'))
        s1, s0, t1, t0 = (per_warp(o(n + '.json'), w) for n in ('s1', 's0', 't1', 't0'))
        h = s1[tw] - s0[tw]; others = [k for k in range(w) if k != tw]
        res[rep] = dict(counts_ok=ok, launches=len(L), decoded=bool(decode), close_bit74_last=close, window_s1=w1, window_twin=w0, probe_cost=o1, harness_group_probed_warp=h, corrected=w1 - w0, corrected_over_harness=(w1 - w0) / h,
                        harness_group_all_warps_median=st.median(a - b for a, b in zip(s1, s0)), nonprobed_group_plain=st.median(s1[k] - s0[k] for k in others), nonprobed_group_instrumented=st.median(t1[k] - t0[k] for k in others),
                        probed_group_instrumented_harness=t1[tw] - t0[tw])
        print(tag, rep, json.dumps(res[rep]), flush=True)
        if not ok or not decode: stop('%s rep %d: counts or decode not as expected (see report)' % (tag, rep))
        if rep == 5:   # per launch ratios of the 5 launch run (window per launch from the report vs the plain harness median): spread
            pass
    return res


def run(wg, outdir):
    os.makedirs(outdir, exist_ok=True); ts = time.strftime('%Y%m%dT%H%M%S'); allr = {}
    for b in (wg + '/m_s0', wg + '/m_s1'):
        if not os.path.exists(b + '.checked'): stop('no check marker ' + b)
    for w in (2, 4, 8, 16):
        for tw in (0, w - 1):
            allr['gas_w%d_p%d' % (w, tw)] = case('gas_w%d_p%d' % (w, tw), wg + '/m_s0', wg + '/m_s1', w, tw, outdir, ts)
    json.dump(allr, open(os.path.join(outdir, 'summary_w_%s.json' % ts), 'x'), indent=1)


if __name__ == '__main__':
    c = sys.argv[1]
    if c == 'prepare': prepare(sys.argv[2])
    elif c == 'run': run(sys.argv[2], sys.argv[3])
    elif c == 'clean': sh(['rm', '-rf', sys.argv[2], '/dev/shm/bittu'])
