"""Omm_1 interleaved runner: run2.py WORK TAG key=value ...  (all variants in one process, round robin launches). Needs /dev/shm/om1/{omm2, omm_dev.ptx}. Raw WORK/TAG_<TS>.json, never overwritten."""
import os, subprocess, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'config'))
import gpucfg
CFG = gpucfg.load(); ENV = CFG.env(); W, TAG = sys.argv[1], sys.argv[2]; D = os.environ.get('OMM_DIR', '/dev/shm/om1'); os.makedirs(W, exist_ok=True)
base = os.path.join(W, '%s_%s' % (TAG, time.strftime('%Y%m%dT%H%M%S')))
if os.path.exists(base + '.json'): sys.exit('refuse to overwrite ' + base)
CFG.check_idle()
p = subprocess.run(['timeout', '-k', '2', '600', D + '/omm2', D + '/omm_dev.ptx', base + '.json'] + sys.argv[3:], env=ENV, capture_output=True, text=True)
print(TAG, 'rc', p.returncode, p.stdout.strip()[-70:], p.stderr[-300:], flush=True)
sys.exit(p.returncode)
