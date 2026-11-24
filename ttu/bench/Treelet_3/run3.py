"""Treelet_3 runner. OptiX only, no TTU code. tethys GPU 1. Idle check by nvidia-smi (compute apps and utilization), never by process names.
    run3.py prep|run|tlsize WORK SCENE SEED PART TAG ... (arguments after PART as in treelet3.cu; binary and PTX from /dev/shm/t3)
Raw outputs are opened with mode x by the host; this script adds a timestamp to every run file name."""
import os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); GC = os.path.join(HERE, '..', 'Phase10f', 'tethys_gpucheck.sh')
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES='1', CUDA_DEVICE_ORDER='PCI_BUS_ID', LD_LIBRARY_PATH=os.path.expanduser('~/opt/nvoptix-580.178.04') + ':' + os.path.expanduser('~/opt/cuda-12.8/lib64') + ':' + os.environ.get('LD_LIBRARY_PATH', ''))
BIN = '/dev/shm/t3/treelet3'; PTXO = '/dev/shm/t3/treelet3_old_dev.ptx'; PTXN = '/dev/shm/t3/treelet3_dev.ptx'


def idle():
    for _ in range(30):
        if subprocess.run(['bash', GC], capture_output=True).returncode == 0: return
        time.sleep(1)
    sys.exit('GPU busy')


def go(mode, work, scene, seed, part, rest, tmo=int(os.environ.get('T3_TMO', 900)), pre=()):
    idle(); t = time.time(); p = subprocess.run(list(pre) + ['timeout', '-k', '2', str(tmo), BIN, mode, PTXO, PTXN, work, scene, str(seed), part] + rest, env=ENV, capture_output=True, text=True)
    print(mode, 'rc', p.returncode, '%.1fs' % (time.time() - t), '\n'.join(p.stdout.strip().splitlines()[-12:]), p.stderr[-600:], flush=True)
    if p.returncode: sys.exit('STOP')


if __name__ == '__main__':
    mode, work, scene, seed, part = sys.argv[1:6]; rest = sys.argv[6:]; os.makedirs(work, exist_ok=True)
    if mode == 'run':   # rest: TAG WPS ROUNDS ARMS...  -> OUT name inserted
        tag, wps, rounds = rest[:3]; out = '%s/run_%s_%s_w%s_%s.json' % (work, tag, part.replace(':', ''), wps, time.strftime('%Y%m%dT%H%M%S')); rest = [tag, wps, rounds, out] + rest[3:]
    go(mode, work, scene, seed, part, rest)
