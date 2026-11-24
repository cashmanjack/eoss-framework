"""Rehosting runs (R1, R2): one launch of a checked build, with the pre run checks Jack ordered (2026-11-24).

    r_run.py BUILD RAY OUTDIR LABEL [host args, e.g. --scene box1 --warps 1 --repeat 3]

Before every run: `who` and the nvidia-smi compute process list of GPU 1 are printed and logged; the run is refused when any
process (anyone's) is on the GPU. The raw json goes to OUTDIR/LABEL_<timestamp>.json, opened with mode x (never overwritten).
The launch itself is ttu/harness/setup3/run.py (needs BUILD.checked, runs under timeout -k 2 30, one process).
"""
import os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, '..', 'harness', 'setup3', 'run.py')
VENV = os.path.expanduser('~/eoss/.venv/bin/python')


def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()


def main():
    build, ray, outdir, label = sys.argv[1:5]
    extra = sys.argv[5:]
    env = dict(os.environ, TTU_GPU='1')   # not CUDA_VISIBLE_DEVICES (gpucfg sets it itself; both together asserts)
    print('who:\n' + sh(['who']))
    apps = sh(['nvidia-smi', '--query-compute-apps=pid,name,gpu_uuid', '--format=csv,noheader'])
    util = sh(['nvidia-smi', '-i', '1', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader'])
    print('nvidia-smi compute apps (all GPUs): %s | GPU 1: %s' % (apps or 'none', util))
    if apps:
        sys.exit('refused: a GPU process exists (%s)' % apps)
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, '%s_%s.json' % (label, time.strftime('%Y%m%dT%H%M%S')))
    open(out, 'x').close()   # mode x: refuses an existing name; the host then writes into it
    r = subprocess.run([VENV, RUN, build, ray, 'zero', out] + extra, env=env)
    print('r_run exit', r.returncode, 'raw file', out)
    print('after: compute apps:', sh(['nvidia-smi', '--query-compute-apps=pid,name', '--format=csv,noheader']) or 'none')
    sys.exit(r.returncode)


if __name__ == '__main__':
    main()
