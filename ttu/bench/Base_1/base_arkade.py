"""Baseline 3 (v2): Arkade (Mandarapu et al., ICS 2024), sample01-knn at commit 45b9425, ~/thirdparty/div1/ark/build_fixed (KN 50, our DEVOUT switch, clock recorder).
usage: python base_arkade.py [REPS]   -> notes/base1/arkade_v2_<ts>.json. Change no code.
Inputs: ark2e5.txt = default_rng(11).random((200000,2)) '%.7f' (100,000 points + 100,000 queries; recorded recipe of the earlier 4090 input; its first 110,000 lines equal ark_a1e5.txt, checked) and ark_a1e5.txt (100,000 points + 10,000 queries).
Configs per input: DEVOUT=1 (device output, as in earlier tables) with the recorder on and off (CB_OFF=1), and the stock pinned host output (recorder off).
Correctness (untimed, ARK_OUT dump of the final neighbour buffer): per query the K nearest points inside the radius by CPU kd tree (2D, z = 0)."""
import re, sys, os, base_common as B, base_checks as C
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
D = B.HOME + '/thirdparty/div1'; BIN = D + '/ark/build_fixed/sample01-knn'; BD = B.HOME + '/thirdparty/base_data'
INPUTS = (('ark2e5_100kq', BD + '/ark2e5.txt', 100000, 100000), ('ark_a1e5_10kq', D + '/data/ark_a1e5.txt', 100000, 10000))
def parse(t):
    p = {}; nums = [l for l in t.splitlines() if re.fullmatch(r'[0-9.]+', l.strip())]
    if len(nums) >= 6: p['build_s'] = float(nums[4]); p['first_launch_host_s'] = float(nums[5])
    m = re.search(r'KERNEL queries (\d+) ms ([0-9.]+)', t)
    if m: p['queries'] = int(m.group(1)); p['kernel_ms'] = float(m.group(2))
    m = re.search(r'GT span_ms ([0-9.]+)', t)
    if m: p['gt_span_ms'] = float(m.group(1))
    m = re.search(r'CBK warps (\d+) total_warp_cycles (\d+) trace_warp_cycles (\d+) f0_cycles (\d+) f0_calls (\d+) f1_cycles (\d+) f1_calls (\d+)', t)
    if m: p.update(warps=int(m[1]), total_warp_cycles=int(m[2]), trace_warp_cycles=int(m[3]), f0_cycles=int(m[4]), f0_calls=int(m[5]), f1_cycles=int(m[6]), f1_calls=int(m[7]))
    return p
doc = B.header('Arkade', BIN, [i[1] for i in INPUTS], 'Radius 0.005, K 50. Timers in the program: BVH build (host), first launch (host, asynchronous: only the enqueue), second launch timed with CUDA events (kernel time). No copy or result time is printed. Warmup process before each run.')
doc['configs'] = []
for iname, f, npt, nq in INPUTS:
    for name, ex in (('devout_recorder_on', {'DEVOUT': '1'}), ('devout_recorder_off', {'DEVOUT': '1', 'CB_OFF': '1'}), ('stock_pinned_recorder_off', {'CB_OFF': '1'})):
        cmd = [BIN, f, str(npt), str(nq), '0.005']; runs = []
        for i in range(REPS):
            x = B.run(cmd, D + '/ark/build_fixed', B.env(extra=ex), 120); x['parsed'] = parse(x['stdout']); runs.append(x)
        c = dict(name=iname + '_' + name, env=ex, cmd=' '.join(cmd).replace(B.HOME, '~'), runs=runs,
                 summary=B.summ(runs, ['build_s', 'first_launch_host_s', 'kernel_ms', 'gt_span_ms', 'total_warp_cycles', 'f0_calls']))
        if name == 'devout_recorder_off':
            out = '/dev/shm/base1/ark_out_%s_%d.bin' % (iname, os.getpid()); os.makedirs('/dev/shm/base1', exist_ok=True)
            chk = B.check_run(cmd, D + '/ark/build_fixed', B.env(extra=dict(ex, ARK_OUT=out)), 300)
            allp = C.load_csv(f)
            chk['correctness'] = C.arkade(out, allp[:npt, :2], allp[npt:npt + nq, :2], 0.005, 50) | dict(reference='CPU kd tree, 2D: the K=50 nearest points within radius 0.005 per query (float64), index sets by distance multiset (rtol 1e-5)')
            os.remove(out); c['correctness'] = chk
        doc['configs'].append(c)
print(B.write('arkade_v2', doc))
