"""Ovh_1 runner (predictions: model/predictions_ovh.md). OptiX only. run.py WORK   (WORK = private raw data dir, files get a timestamp, never overwritten)
Needs /dev/shm/ovh1/{ovh, ovh_dev.ptx, ovh_empty_dev.ptx}; LD_LIBRARY_PATH to the extracted OptiX library when the system one is missing; one GPU via CUDA_VISIBLE_DEVICES."""
import os, subprocess, sys, time
W = sys.argv[1]; os.makedirs(W, exist_ok=True); TS = time.strftime('%Y%m%dT%H%M%S')
D = '/dev/shm/ovh1'
for scene in ('tri1', 'soup'):
    for ptx, nrep, tag in (('ovh_empty_dev.ptx', '1', 'empty'), ('ovh_dev.ptx', '0,1,3', 'trace')):
        out = '%s/%s_%s_%s.json' % (W, scene, tag, TS)
        assert not os.path.exists(out)
        smi = subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu', '--format=csv,noheader'], capture_output=True, text=True).stdout
        p = subprocess.run(['timeout', '-k', '2', '120', D + '/ovh', D + '/' + ptx, out, 'scene=' + scene, 'nrep=' + nrep], capture_output=True, text=True)
        print(scene, tag, 'rc', p.returncode, p.stdout.strip(), p.stderr[-300:], flush=True)
        if p.returncode: sys.exit('STOP')
