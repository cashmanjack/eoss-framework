"""Load_1 refit arm: python3 load2_run.py OUTDIR  (GPU 1). 32 query smoke run (1 step) first, then 100,000 queries, 7 steps. Raw json timestamped, mode x."""
import json, os, subprocess, sys, time
D = '/dev/shm/ld'; env = dict(os.environ, CUDA_VISIBLE_DEVICES='1', CUDA_DEVICE_ORDER='PCI_BUS_ID'); out = sys.argv[1]; mode = sys.argv[2] if len(sys.argv) > 2 else '0'; os.makedirs(out, exist_ok=True); ts = time.strftime('%Y%m%dT%H%M%S')
busy = subprocess.run(['nvidia-smi', '--id=1', '--query-compute-apps=pid', '--format=csv,noheader'], capture_output=True, text=True).stdout.strip()
if busy: sys.exit('GPU 1 has compute processes: ' + busy)
res = []
for nq, steps in ((32, 1), (100000, 7)):  # mode 1 (scramble): the 7 steps use f = 0, 0.5, 1, 2, 4, 8, 16 cells
    r = subprocess.run(['timeout', '-k', '2', '30', D + '/load2', D + '/dev.ptx', str(nq), '20', str(steps), mode], capture_output=True, text=True, env=env)
    if r.returncode: sys.exit('FAULT nq %d rc %d\n%s\n%s' % (nq, r.returncode, r.stdout[-500:], r.stderr[-800:]))
    for l in r.stdout.strip().splitlines():
        j = json.loads(l); L = [x for x in j['launches'] if not x['cold']]; sp = sorted(x['span_ns'] for x in L)[len(L) // 2]; res.append(j)
        print('nq %6d step %d f %.2f tree %s span %8.2f us  calls/query %.3f hits %d warp cyc %8.0f' % (nq, j['step'], j['f'], j['tree'], sp / 1e3, L[0]['calls'] / nq, L[0]['hits'], sum(x['warp_cyc_mean'] for x in L) / len(L)), flush=True)
with open('%s/refit_m%s_%s.json' % (out, mode, ts), 'x') as f: json.dump(res, f)
