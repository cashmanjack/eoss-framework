"""Load_1 program arm: python3 load3_run.py OUTDIR (GPU 1). Arms: PROG 0 trivial / 1 Mochi program x GAS / instance level. 32 query smoke first. Raw json timestamped, mode x."""
import json, os, subprocess, sys, time
D = '/dev/shm/ld'; DS = os.path.expanduser('~/Mochi-DCD-on-RT/datasets/100k_particles.txt'); env = dict(os.environ, CUDA_VISIBLE_DEVICES='1', CUDA_DEVICE_ORDER='PCI_BUS_ID'); out = sys.argv[1]; os.makedirs(out, exist_ok=True); ts = time.strftime('%Y%m%dT%H%M%S')
busy = subprocess.run(['nvidia-smi', '--id=1', '--query-compute-apps=pid', '--format=csv,noheader'], capture_output=True, text=True).stdout.strip()
if busy: sys.exit('GPU 1 has compute processes: ' + busy)
res = []
for nq in (32, 100000):
    for prog, ias in ((0, 0), (1, 0), (0, 1), (1, 1)):
        r = subprocess.run(['timeout', '-k', '2', '30', D + '/load3', D + '/dev3_p%d.ptx' % prog, DS, str(nq), '20', str(ias)], capture_output=True, text=True, env=env)
        if r.returncode: sys.exit('FAULT nq %d prog %d ias %d rc %d\n%s\n%s' % (nq, prog, ias, r.returncode, r.stdout[-500:], r.stderr[-800:]))
        j = json.loads(r.stdout.strip().splitlines()[-1]); j['prog'] = prog; res.append(j); L = [x for x in j['launches'] if not x['cold']]; sp = sorted(x['span_ns'] for x in L)[len(L) // 2]
        print('nq %6d prog %d ias %d span median %8.2f us (cold %.2f) calls/query %.3f warp cyc %8.0f' % (nq, prog, ias, sp / 1e3, j['launches'][0]['span_ns'] / 1e3, L[0]['calls'] / nq, sum(x['warp_cyc_mean'] for x in L) / len(L)), flush=True)
with open('%s/prog_%s.json' % (out, ts), 'x') as f: json.dump(res, f)
