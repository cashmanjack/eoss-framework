"""Baseline 5 (v2): RT-BarnesHut (Nagarajan et al., SC 2024), branch BarnesHutRT of OWLRayTracing, ~/OWLRayTracing/build/rtbarneshut (stock, built 2026-06-22, PRINT_ARTIFACT true).
usage: python base_barneshut.py [REPS]   -> notes/base1/barneshut_v2_<ts>.json. Change no code (timed runs use the stock binary).
The program hardcodes CUDA device 1, so both GPUs are visible; only ordinal 1 is used (the warmup runs on the same GPU).
The program's own "RT force" timer does not wait for the launch (no synchronisation before the end timestamp), so it is NOT a kernel time: GPU launch time is taken from nsys (cuda kernel summary) in untimed runs.
Correctness (untimed): an observer copy of the same source (/dev/shm/bhobs, one dump line, notes/base1/bh_observer_dump.diff) against a float64 direct sum on 2,000 bodies (scalar force G m_i sum m_j / r^2, G 0.1)."""
import re, sys, os, json, subprocess, base_common as B, base_checks as C
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
BIN = B.HOME + '/OWLRayTracing/build/rtbarneshut'; OBS = '/dev/shm/bhobs/build/rtbarneshut'; DD = B.HOME + '/thirdparty/base_data'
def parse(t):
    p = {}
    for k, pat in (('preprocessing_s', r'Preprocessing Time: ([0-9.]+)'), ('rt_force_s', r'RT Cores Force Calculations time: ([0-9.]+)'), ('execution_s', r'Execution time: ([0-9.]+)')):
        m = re.search(pat, t); p[k] = float(m.group(1)) if m else None
    return p
def nsys_kernel_ms(cmd, cwd, environ):
    """GPU time of the optixLaunch kernel(s) from nsys, untimed run."""
    B.idle_check(); base = '/dev/shm/base1/nsys_%d' % os.getpid()
    p = subprocess.run(['timeout', '-k', '2', '600', 'nsys', 'profile', '-o', base, '--force-overwrite=false', '--stats=false', '-t', 'cuda'] + cmd, cwd=cwd, env=environ, capture_output=True, text=True)
    s = subprocess.run(['nsys', 'stats', '-r', 'cuda_gpu_kern_sum', '-f', 'csv', base + '.nsys-rep'], capture_output=True, text=True).stdout
    for f in (base + '.nsys-rep', base + '.sqlite'):
        try: os.remove(f)
        except OSError: pass
    return s
doc = B.header('RT-BarnesHut', BIN, [DD + '/bh2_g100k.txt', DD + '/bh2_g1M.txt'], 'Gaussian disc (x,y sigma 100, z sigma 20, mass uniform 1 to 100, seed 42, ttu/bench/Base_1/gen_inputs.py, header N,1,0.025,0.05,0.5 as printed by the earlier runs). The earlier g100k and g1M files are gone; the body recipe is the one in Jack\'s generate_synthetic1M.py and is NOT confirmed to be the one used for them. One force step per run. Printed preprocessing = tree to DFS + auto ropes + intersection setup; sort and tree build are CPU and not in the printed figures. Warmup process before each run (the GPU is idle during the seconds of CPU tree building, so the recorded clock shows whether it dropped).')
doc['configs'] = []; os.makedirs('/dev/shm/base1', exist_ok=True)
for name, f in (('g100k', 'bh2_g100k.txt'), ('g1M', 'bh2_g1M.txt')):
    cmd = [BIN, 'treelogy', DD + '/' + f]; runs = []; E = B.env(cuda_visible='0,1')
    for i in range(REPS):
        x = B.run(cmd, B.HOME + '/OWLRayTracing/build', E, 600); x['parsed'] = parse(x['stdout']); runs.append(x)
    fo = '/dev/shm/base1/bhf_%s_%d.bin' % (name, os.getpid())
    chk = B.check_run([OBS, 'treelogy', DD + '/' + f], '/dev/shm/bhobs/build', dict(E, BH_FORCES=fo), 900)
    chk['correctness'] = C.barneshut(fo) | dict(reference='float64 direct sum on 2,000 random bodies; the program is a Barnes-Hut approximation (opening angle 0.5), so a few percent difference is expected, not zero')
    os.remove(fo)
    ns = nsys_kernel_ms(cmd, B.HOME + '/OWLRayTracing/build', E)
    doc['configs'].append(dict(name=name, cmd=' '.join(cmd).replace(B.HOME, '~'), runs=runs, summary=B.summ(runs, ['preprocessing_s', 'rt_force_s', 'execution_s']), correctness=chk, nsys_cuda_gpu_kern_sum_csv=ns))
print(B.write('barneshut_v2', doc))
