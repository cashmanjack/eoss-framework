"""Run a patched setup3 binary on the first idle GPU, only if check_seq.py passed on this exact file.

    run.py BIN RAYS FILL OUT.json [--no-ttu] [--repeat N] [--ncu METRICS --ncu-out CSV [--ncu-skip K]] [--inject TOOL.so]

--ncu: run the binary under Nsight Compute (same timeout), profiling only ttu_one, caches and
clocks left alone (--cache-control none --clock-control none), one CSV row per ttu_one launch.

Refuses if: BIN.checked is missing or holds another sha256, any process is on
the chosen GPU, or the driver has no validated TTU group (ttu/config/profiles.json). Runs under `timeout -k 2 30` (one process, all repeats inside it).
"""
import hashlib, os, subprocess, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'config'))
import gpucfg  # noqa: E402  (one config per GPU: index, driver, OptiX library, SM count)


def smi(*a):
    return subprocess.run(['nvidia-smi'] + list(a), capture_output=True, text=True).stdout.strip()


def main():
    binp, rays, fill, out = sys.argv[1:5]
    extra = sys.argv[5:]
    ncu = []
    inject = None
    if '--inject' in extra:   # an NVBit tool (BITTU): CUDA_INJECTION64_PATH; BITTU_* settings come from the caller's environment
        i = extra.index('--inject'); inject = os.path.abspath(extra[i + 1]); del extra[i:i + 2]
    if '--ncu' in extra:
        i = extra.index('--ncu'); metrics = extra[i + 1]; del extra[i:i + 2]
        j = extra.index('--ncu-out'); csv = extra[j + 1]; del extra[j:j + 2]
        skip = []
        if '--ncu-instances' in extra:   # per unit values (e.g. one L1 per SM) instead of only the sum
            extra.remove('--ncu-instances'); skip += ['--print-metric-instances', 'values']
        if '--ncu-skip' in extra:   # profile only after the first K ttu_one launches (e.g. pass 0 of --passes 2)
            k = extra.index('--ncu-skip'); skip = ['--launch-skip', str(int(extra[k + 1]))]; del extra[k:k + 2]
        ncu = [os.path.expanduser('~/opt/cuda-12.8/bin/ncu'), '--target-processes', 'all', '-k', 'ttu_one',
               '--cache-control', 'none', '--clock-control', 'none', '--metrics', metrics, '--csv', '--page', 'raw',
               '--log-file', csv] + skip
    sha = hashlib.sha256(open(binp, 'rb').read()).hexdigest()
    chk = binp + '.checked'
    if not os.path.exists(chk) or open(chk).read().strip() != sha:
        sys.exit('refused: check_seq.py has not passed on this binary (%s)' % sha)
    cfg = gpucfg.load()   # first idle GPU (TTU_GPU overrides); refuses a driver without a validated TTU group
    cfg.check_idle()
    GPU = cfg.index
    env = cfg.env(dict(os.environ, **({'CUDA_INJECTION64_PATH': inject} if inject else {})))
    ptx = os.path.join(os.path.dirname(binp), 'setup3_dev.ptx')
    cmd = ['timeout', '-k', '2', '30'] + ncu + [binp, ptx, os.path.join(os.path.dirname(binp), 'words.txt'), rays, fill, out] + extra
    print('sha256', sha)
    r = subprocess.run(cmd, env=env)
    print('exit', r.returncode)
    print('GPU %d after:' % GPU, smi('-i', str(GPU), '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader'),
          '| procs:', smi('-i', str(GPU), '--query-compute-apps=pid', '--format=csv,noheader') or 'none')
    sys.exit(r.returncode)


if __name__ == '__main__':
    main()
