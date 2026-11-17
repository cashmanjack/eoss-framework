"""Cache_1 / Cache_2 runner (plain CUDA, no TTU; predictions model/predictions_cache.md). /usr/bin/python3 run.py WORK MODE...   (binary /dev/shm/cache1 from cache1.cu, GPU 1 idle, timeout 600)
Raw text per mode with a timestamp name, never overwritten."""
import os, subprocess, sys, time
work = sys.argv[1]; os.makedirs(work, exist_ok=True); ts = time.strftime('%Y%m%dT%H%M%S')
for mode in sys.argv[2:]:
    out = '%s/cache1_%s_%s.txt' % (work, mode, ts)
    p = subprocess.run(['timeout', '-k', '2', '600', '/dev/shm/cache1', mode], capture_output=True, text=True, env=dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES='1'))
    with open(out, 'x') as f: f.write(p.stdout)
    print(mode, 'exit', p.returncode, len(p.stdout.splitlines()), 'lines', out, p.stderr[-200:], flush=True)
    if p.returncode: sys.exit('STOP')
