"""One run of a rehosted RTNN kernel (R3 to R5): the same preconditions as ttu/harness/setup3/run.py.

    rtnn_run.py BUILD OUTDIR LABEL [host options]       e.g.  rtnn_run.py /dev/shm/t9r/k_s1 notes/rehost r3_w1 --mode real --pts 4096 --r 0.08 --warps 1

Refuses unless: BUILD.checked holds this file's sha256 (check_seq passed on this exact file); `who` and nvidia-smi are logged and no process of anyone is on any GPU;
gpucfg accepts the driver. Runs `timeout -k 2 30` on the GPU gpucfg selects (TTU_GPU=1 here). The raw json is OUTDIR/LABEL_<timestamp>.json, created with mode x.
Stage 0 builds run with --mode inject (fake candidates, the group is NOPs); stage 1 builds with --mode real."""
import hashlib, json, os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'config'))
import gpucfg


def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()


def main():
    build, outdir, label = sys.argv[1:4]; extra = sys.argv[4:]
    sha = hashlib.sha256(open(build, 'rb').read()).hexdigest()
    chk = build + '.checked'
    if not os.path.exists(chk) or open(chk).read().strip() != sha:
        sys.exit('refused: check_seq.py has not passed on this binary (%s)' % sha)
    meta = json.load(open(build + '.layout.json'))
    print('who:\n' + sh(['who']))
    apps = sh(['nvidia-smi', '--query-compute-apps=pid,name,gpu_uuid', '--format=csv,noheader'])
    print('nvidia-smi compute apps (all GPUs): %s' % (apps or 'none'))
    if apps: sys.exit('refused: a GPU process exists (%s)' % apps)
    os.environ['TTU_GPU'] = os.environ.get('TTU_GPU', '1')
    cfg = gpucfg.load(); cfg.check_idle()
    env = cfg.env(dict(os.environ))
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, '%s_%s.json' % (label, time.strftime('%Y%m%dT%H%M%S') + '_%06d' % (int(time.time() * 1e6) % 1000000)))
    open(out, 'x').close()
    d = os.path.dirname(os.path.abspath(build))
    gasf = os.path.join(outdir, 'gas_%s_%s.bin' % (label, time.strftime('%Y%m%dT%H%M%S') + '_%06d' % (int(time.time() * 1e6) % 1000000)))
    open(gasf, 'x').close(); extra = extra + ['--dump-gas', gasf]   # the GAS of every scene we run is saved (private)
    cmd = ['timeout', '-k', '2', '30', build, os.path.join(d, 'rtnn_dev.ptx'), os.path.join(d, 'words.txt'), out] + extra
    print('sha256', sha, '| stage', meta['stage'], '| rounds', meta['rounds'], '| inline', meta['inline'], '|', ' '.join(extra))
    r = subprocess.run(cmd, env=env)
    print('exit', r.returncode, '| raw file', out)
    print('GPU %d after:' % cfg.index, sh(['nvidia-smi', '-i', str(cfg.index), '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader']), '| procs:', sh(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader']) or 'none')
    sys.exit(r.returncode)


if __name__ == '__main__':
    main()
