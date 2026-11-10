"""Run a patched setup3 binary on GPU 1, only if check_seq.py passed on this exact file.

    run.py BIN RAYS FILL OUT.json [--no-ttu] [--repeat N] [--ncu METRICS --ncu-out CSV [--ncu-skip K]]

--ncu: run the binary under Nsight Compute (same timeout), profiling only ttu_one, caches and
clocks left alone (--cache-control none --clock-control none), one CSV row per ttu_one launch.

Refuses if: BIN.checked is missing or holds another sha256, any process is on
GPU 1, or the driver is not 580.178.04. Runs under `timeout -k 2 30` (one process, all repeats inside it).
"""
import hashlib, os, subprocess, sys

DRIVER = '580.178.04'
OPTIX_LIB = os.path.expanduser('~/opt/nvoptix-580.178.04')


def smi(*a):
    return subprocess.run(['nvidia-smi'] + list(a), capture_output=True, text=True).stdout.strip()


def main():
    binp, rays, fill, out = sys.argv[1:5]
    extra = sys.argv[5:]
    ncu = []
    if '--ncu' in extra:
        i = extra.index('--ncu'); metrics = extra[i + 1]; del extra[i:i + 2]
        j = extra.index('--ncu-out'); csv = extra[j + 1]; del extra[j:j + 2]
        skip = []
        if '--ncu-skip' in extra:   # profile only after the first K ttu_one launches (e.g. pass 0 of --passes 2)
            k = extra.index('--ncu-skip'); skip = ['--launch-skip', str(int(extra[k + 1]))]; del extra[k:k + 2]
        ncu = [os.path.expanduser('~/opt/cuda-12.8/bin/ncu'), '--target-processes', 'all', '-k', 'ttu_one',
               '--cache-control', 'none', '--clock-control', 'none', '--metrics', metrics, '--csv', '--page', 'raw',
               '--log-file', csv] + skip
    sha = hashlib.sha256(open(binp, 'rb').read()).hexdigest()
    chk = binp + '.checked'
    if not os.path.exists(chk) or open(chk).read().strip() != sha:
        sys.exit('refused: check_seq.py has not passed on this binary (%s)' % sha)
    if smi('--query-gpu=driver_version', '--format=csv,noheader').split('\n')[0] != DRIVER:
        sys.exit('refused: driver changed')
    procs = smi('-i', '1', '--query-compute-apps=pid', '--format=csv,noheader')
    if procs:
        sys.exit('refused: GPU 1 busy: %s' % procs)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='1',
               LD_LIBRARY_PATH=OPTIX_LIB + ':' + os.environ.get('LD_LIBRARY_PATH', ''))
    ptx = os.path.join(os.path.dirname(binp), 'setup3_dev.ptx')
    cmd = ['timeout', '-k', '2', '30'] + ncu + [binp, ptx, os.path.join(os.path.dirname(binp), 'words.txt'), rays, fill, out] + extra
    print('sha256', sha)
    r = subprocess.run(cmd, env=env)
    print('exit', r.returncode)
    print('GPU 1 after:', smi('-i', '1', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader'),
          '| procs:', smi('-i', '1', '--query-compute-apps=pid', '--format=csv,noheader') or 'none')
    sys.exit(r.returncode)


if __name__ == '__main__':
    main()
