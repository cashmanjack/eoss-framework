"""Load_1: loaded point query test (OptiX only). python3 load_run.py OUTDIR   (GPU 1 on triton). Raw json timestamped, mode x; stops on the first failure."""
import json, os, subprocess, sys, time
D = '/dev/shm/ld'; NQS = [32, 1024, 4544, 9088, 18176, 36352, 100000, 409600, 1638400]
env = dict(os.environ, CUDA_VISIBLE_DEVICES='1', CUDA_DEVICE_ORDER='PCI_BUS_ID')
out = sys.argv[1]; os.makedirs(out, exist_ok=True); ts = time.strftime('%Y%m%dT%H%M%S')
busy = subprocess.run(['nvidia-smi', '--id=1', '--query-compute-apps=pid', '--format=csv,noheader'], capture_output=True, text=True).stdout.strip()
if busy: sys.exit('GPU 1 has compute processes: ' + busy)
res = []
for order in (0, 1):
    for nq in NQS:
        rec = '%s/warprec_o%d_n%d_%s.bin' % (out, order, nq, ts)
        r = subprocess.run(['timeout', '-k', '2', '30', D + '/load', D + '/dev.ptx', str(order), str(nq), '20', rec], capture_output=True, text=True, env=env)
        if r.returncode: sys.exit('FAULT order %d nq %d rc %d\n%s\n%s' % (order, nq, r.returncode, r.stdout[-500:], r.stderr[-800:]))
        j = json.loads(r.stdout.strip().splitlines()[-1]); res.append(j)
        L = [x for x in j['launches'] if not x['cold']]; sp = sorted(x['span_ns'] for x in L)[len(L) // 2]
        print('order %d nq %8d span median %9.2f us  cold %9.2f us  calls/query %.3f hits %d  warp cyc mean %9.0f' % (order, nq, sp / 1e3, j['launches'][0]['span_ns'] / 1e3, L[0]['calls'] / nq, L[0]['hits'], sum(x['warp_cyc_mean'] for x in L) / len(L)), flush=True)
with open('%s/load1_%s.json' % (out, ts), 'x') as f: json.dump(res, f)
