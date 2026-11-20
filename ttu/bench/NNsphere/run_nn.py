"""NNsphere runner (pc14). run_nn.py WORK TAG ARM DATA K QN REPS [timeout] [dump]  -- gpucheck first, timeout -k 2, raw json and dumps timestamped (host opens with mode x). Radii from the radii_*.json given by env NN_RADII."""
import json, os, subprocess, sys, time
W, TAG, ARM, DATA, K, QN, REPS = sys.argv[1:8]; TMO = sys.argv[8] if len(sys.argv) > 8 else '30'; DUMP = len(sys.argv) > 9 and sys.argv[9] == 'dump'
R = json.load(open(os.environ['NN_RADII']))[DATA]['r'][K]; TS = time.strftime('%Y%m%dT%H%M%S'); os.makedirs(W, exist_ok=True)
GC = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'notes', 'pc14_gpucheck.sh')
if subprocess.run(['bash', GC], capture_output=True).returncode: sys.exit('GPU busy (pc14_gpucheck)')
out = '%s/%s_%s_%s_k%s_q%s_%s.json' % (W, TAG, ARM, DATA, K, QN, TS)
args = ['timeout', '-k', '2', TMO, '/dev/shm/nn/nn', ARM, ('/dev/shm/nn/nn_dev_count.ptx' if os.environ.get('NN_COUNT') == '1' else '/dev/shm/nn/nn_dev_k%s.ptx' % K), out, 'data=/dev/shm/nn/data/%s.bin' % DATA, 'n=1000000', 'K=' + ('1' if os.environ.get('NN_COUNT') == '1' else K), 'r=%.9g' % R, 'qn=' + QN, 'reps=' + REPS]
if os.environ.get('NN_SORT') == '1': args.append('sort=1'); TAG2 = 'S'
if DUMP: args.append('dump=/dev/shm/nn/out/%s_%s_%s_k%s_q%s_%s' % (TAG, ARM, DATA, K, QN, TS))
os.makedirs('/dev/shm/nn/out', exist_ok=True)
env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', CUDA_DEVICE_ORDER='PCI_BUS_ID')
t0 = time.time(); p = subprocess.run(args, env=env, capture_output=True, text=True)
print(ARM, DATA, 'K', K, 'Q', QN, 'rc', p.returncode, '%.1fs' % (time.time() - t0), p.stdout.strip()[-80:], p.stderr[-400:], flush=True)
if p.returncode: sys.exit('STOP')
print(out)
