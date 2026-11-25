"""Baseline 3: Arkade (Mandarapu et al., ICS 2024), sample01-knn at commit 45b9425 as built in ~/thirdparty/div1/ark/build_fixed (KN 50, our output buffer switch DEVOUT, clock recorder).
usage: python base_arkade.py [REPS]   -> notes/base1/arkade_<ts>.json. Change no code.
Configs: DEVOUT=1 (device output buffer, as in all earlier tables) with recorder on and off (CB_OFF=1), and the stock host pinned output (no DEVOUT) with recorder off."""
import re, sys, base_common as B
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
D = B.HOME + '/thirdparty/div1'; BIN = D + '/ark/build_fixed/sample01-knn'; DATA = D + '/data/ark_a1e5.txt'
def parse(t):
    p = {}; nums = [l for l in t.splitlines() if re.fullmatch(r'[0-9.]+', l.strip())]
    if len(nums) >= 6: p['build_s'] = float(nums[4]); p['first_launch_host_s'] = float(nums[5])   # lines: radius, K, npoints, nqueries, build, first launch (async enqueue, not a kernel time)
    m = re.search(r'KERNEL queries (\d+) ms ([0-9.]+)', t)
    if m: p['queries'] = int(m.group(1)); p['kernel_ms'] = float(m.group(2))
    m = re.search(r'GT span_ms ([0-9.]+)', t)
    if m: p['gt_span_ms'] = float(m.group(1))
    m = re.search(r'CBK warps (\d+) total_warp_cycles (\d+) trace_warp_cycles (\d+) f0_cycles (\d+) f0_calls (\d+) f1_cycles (\d+) f1_calls (\d+)', t)
    if m: p.update(warps=int(m[1]), total_warp_cycles=int(m[2]), trace_warp_cycles=int(m[3]), f0_cycles=int(m[4]), f0_calls=int(m[5]), f1_cycles=int(m[6]), f1_calls=int(m[7]))
    return p
doc = B.header('Arkade', BIN, [DATA], 'ark_a1e5.txt = 100,000 uniform random 2D points (z = 0) followed by 10,000 query points; radius 0.005, K 50, 10,000 queries. The earlier 4090 tables also used a 100,000 query file (ark2e5.txt) that no longer exists. Timers in the program: BVH build (host), first launch (host, asynchronous, only the enqueue), second launch with CUDA events (kernel time). No host to device copy or result copy time is printed.')
doc['configs'] = []
for name, ex in (('devout_recorder_on', {'DEVOUT': '1'}), ('devout_recorder_off', {'DEVOUT': '1', 'CB_OFF': '1'}), ('stock_pinned_recorder_off', {'CB_OFF': '1'})):
    cmd = [BIN, DATA, '100000', '10000', '0.005']; runs = []
    for i in range(REPS):
        x = B.run(cmd, D + '/ark/build_fixed', B.env(extra=ex), 120); x['parsed'] = parse(x['stdout']); runs.append(x)
    doc['configs'].append(dict(name=name, env=ex, cmd=' '.join(cmd).replace(B.HOME, '~'), runs=runs, summary=B.summarize(runs, ['build_s', 'first_launch_host_s', 'kernel_ms', 'gt_span_ms', 'total_warp_cycles', 'f0_calls']) | {'wall_s': B.stats([x['wall_s'] for x in runs if x['rc'] == 0])}))
print(B.write('arkade', doc))
