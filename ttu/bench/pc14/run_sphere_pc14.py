"""Sphere_1 runner (predictions: model/predictions_spheres.md). OptiX only, no TTU code. run.py WORK [d1|d2|smoke]
Needs /dev/shm/sp1/{sphere, sphere_dev.ptx, sphere_cnt_dev.ptx}; one GPU via TTU_GPU (gpucfg's rule; we set CUDA_DEVICE_ORDER); LD_LIBRARY_PATH to ~/opt/nvoptix-<driver> when the system library is missing.
Raw files: WORK/<tag>_<TS>.json (+ .bin per ray results), opened with mode x (the host refuses to overwrite)."""
import os, subprocess, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'config'))
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES='0', CUDA_DEVICE_ORDER='PCI_BUS_ID')   # pc14: the GPU 0 idle test is notes/pc14_gpucheck.sh (the Steam helper counts as a compute process for gpucfg)
GC = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'notes', 'pc14_gpucheck.sh')
W = sys.argv[1]; WHICH = sys.argv[2] if len(sys.argv) > 2 else 'd1'; os.makedirs(W, exist_ok=True); TS = time.strftime('%Y%m%dT%H%M%S'); D = '/dev/shm/sp1'
ARMS = ['sph', 'custom', 'tri0', 'tri1']
NS = (1, 4, 16, 64, 256)


def run(tag, ptx, args, tmo=30):
    out = '%s/%s_%s.json' % (W, tag, TS); assert not os.path.exists(out)
    if subprocess.run(['bash', GC], capture_output=True).returncode: sys.exit('GPU busy (pc14_gpucheck)')
    p = subprocess.run(['timeout', '-k', '2', str(tmo), D + '/sphere', D + '/' + ptx, out] + args, env=ENV, capture_output=True, text=True)
    print(tag, 'rc', p.returncode, p.stdout.strip()[-60:], p.stderr[-300:], flush=True)
    if p.returncode: sys.exit('STOP')


if WHICH == 'smoke':
    for k in ARMS: run('smoke_%s' % k, 'sphere_dev.ptx', ['kind=' + k, 'scene=line', 'N=4', 'r=0.05', 'verify=%s/smoke_%s_%s.bin' % (W, k, TS), 'nver=4096', 'sat=65536', 'reps=3', 'ncal=1', 'warp=/dev/null'])
if WHICH == 'd1':     # controlled candidates: N spheres along a line, the axis rays cross every box and hit none
    for n in (1, 4, 16, 64, 256):
        ids = '%s/ids_%s.txt' % (W, TS); open(ids, 'w').write('7\n')
        run('d1_count_n%d' % n, 'sphere_cnt_dev.ptx', ['kind=custom', 'scene=line', 'N=%d' % n, 'r=0.05', 'verify=%s/d1_count_n%d_%s.bin' % (W, n, TS), 'nver=4096'])
        for k in ARMS:
            run('d1_%s_n%d' % (k, n), 'sphere_dev.ptx', ['kind=' + k, 'scene=line', 'N=%d' % n, 'r=0.05', 'verify=%s/d1_%s_n%d_%s.bin' % (W, k, n, TS), 'nver=4096', 'sat=4194304', 'reps=10', 'ncal=3', 'warp=' + ids, 'wreps=12'])
if WHICH == 'd1b':    # near miss line: centres 0.72 r off the axis in x and y (axis 1.018 r from every centre), jitter 0.005 r (nearest approach at least 1.011 r), zero hits expected
    for n in (1, 4, 16, 64, 256):
        ids = '%s/ids_%s.txt' % (W, TS); open(ids, 'w').write('7\n')
        for k in ARMS:
            run('d1b_%s_n%d' % (k, n), 'sphere_dev.ptx', ['kind=' + k, 'scene=line', 'off=0.72', 'jit=0.005', 'N=%d' % n, 'r=0.05', 'verify=%s/d1b_%s_n%d_%s.bin' % (W, k, n, TS), 'nver=4096', 'sat=4194304', 'reps=10', 'ncal=3', 'warp=' + ids, 'wreps=12'])
if WHICH in ('ncu', 'ncub'):    # warp instructions per candidate: ncu on one saturation launch, D1 N = 16, 64, 256 (the first launch, a warm up, is skipped)
    for n in (16, 64, 256):
        for k in ARMS:
            out = '%s/%s_%s_n%d_%s.csv' % (W, WHICH, k, n, TS); assert not os.path.exists(out)
            if subprocess.run(['bash', GC], capture_output=True).returncode: sys.exit('GPU busy (pc14_gpucheck)')
            env = dict(ENV, TMPDIR=os.environ.get('NCU_TMP', '/dev/shm/sp1/ncutmp'))
            p = subprocess.run(['timeout', '-k', '2', '120', 'ncu', '--csv', '-k', 'regex:optixLaunch', '--launch-skip', '1', '--launch-count', '1', '--metrics', 'smsp__inst_executed.sum,sm__cycles_elapsed.max,lts__t_sectors.sum', D + '/sphere', D + '/sphere_dev.ptx', '%s/ncujson_%s_n%d_%s.json' % (W, k, n, TS), 'kind=' + k, 'scene=line', 'N=%d' % n, 'r=0.05', 'sat=1048576', 'ncu=1'] + (['off=0.72', 'jit=0.005'] if WHICH == 'ncub' else []), env=env, capture_output=True, text=True)
            open(out, 'x').write(p.stdout); print('ncu', k, n, 'rc', p.returncode, p.stderr[-200:], flush=True)
            if p.returncode: sys.exit('STOP')
if WHICH == 'd2':     # random rays on the same random point set, three radii
    for r in ('0.02', '0.04', '0.08'):
        run('d2_count_r%s' % r, 'sphere_cnt_dev.ptx', ['kind=custom', 'N=20000', 'r=' + r, 'verify=%s/d2_count_r%s_%s.bin' % (W, r, TS), 'nver=1048576'])
        for k in ARMS:
            run('d2_%s_r%s' % (k, r), 'sphere_dev.ptx', ['kind=' + k, 'N=20000', 'r=' + r, 'verify=%s/d2_%s_r%s_%s.bin' % (W, k, r, TS), 'nver=1048576', 'sat=4194304', 'reps=10', 'ncal=3'])
if WHICH == 'stack':   # B: N stacked one triangle candidates, identical lanes (jit 0), one warp and saturation
    for n in (1, 4, 16, 64, 256, 1024, 4096):
        ids = '%s/ids_%s.txt' % (W, TS); open(ids, 'w').write('7\n')
        run('stack_n%d' % n, 'sphere_dev.ptx', ['kind=stack', 'scene=line', 'jit=0', 'N=%d' % n, 'r=0.05', 'verify=%s/stack_n%d_%s.bin' % (W, n, TS), 'nver=4096', 'sat=4194304', 'reps=10', 'ncal=3', 'warp=' + ids, 'wreps=12'])
if WHICH == 'd2e':    # post hoc (after F427's agreement check): custom primitive with the entry root only, same random scene as d2
    for r in ('0.02', '0.04', '0.08'):
        run('d2e_custom_r%s' % r, 'sphere_entry_dev.ptx', ['kind=custom', 'N=20000', 'r=' + r, 'verify=%s/d2e_custom_r%s_%s.bin' % (W, r, TS), 'nver=1048576', 'sat=4194304', 'reps=10', 'ncal=3'])
if WHICH == 'ncue':   # ncu of the entry only custom arm, D1 N = 16, 64, 256
    for n in (16, 64, 256):
        out = '%s/ncue_custom_n%d_%s.csv' % (W, n, TS); assert not os.path.exists(out)
        if subprocess.run(['bash', GC], capture_output=True).returncode: sys.exit('GPU busy (pc14_gpucheck)')
        env = dict(ENV, TMPDIR=os.environ.get('NCU_TMP', '/dev/shm/sp1/ncutmp'))
        p = subprocess.run(['timeout', '-k', '2', '120', 'ncu', '--csv', '-k', 'regex:optixLaunch', '--launch-skip', '1', '--launch-count', '1', '--metrics', 'smsp__inst_executed.sum,sm__cycles_elapsed.max,lts__t_sectors.sum', D + '/sphere', D + '/sphere_entry_dev.ptx', '%s/ncuejson_custom_n%d_%s.json' % (W, n, TS), 'kind=custom', 'scene=line', 'N=%d' % n, 'r=0.05', 'sat=1048576', 'ncu=1'], env=env, capture_output=True, text=True)
        open(out, 'x').write(p.stdout); print('ncu custom entry', n, 'rc', p.returncode, flush=True)
        if p.returncode: sys.exit('STOP')
