"""Load_1 sequence arm: python3 load3_seq_run.py OUTDIR (GPU 1). SEQ 0, 1, 2 with Mochi program and instance level; 32 query smoke first. Raw json timestamped, mode x."""
import json, os, subprocess, sys, time
D = '/dev/shm/ld'; DS = os.path.expanduser('~/Mochi-DCD-on-RT/datasets/100k_particles.txt'); env = dict(os.environ, CUDA_VISIBLE_DEVICES='1', CUDA_DEVICE_ORDER='PCI_BUS_ID'); out = sys.argv[1]; os.makedirs(out, exist_ok=True); ts = time.strftime('%Y%m%dT%H%M%S')
busy = subprocess.run(['nvidia-smi', '--id=1', '--query-compute-apps=pid', '--format=csv,noheader'], capture_output=True, text=True).stdout.strip()
if busy: sys.exit('GPU 1 has compute processes: ' + busy)
res = []
for nq, rep, warm in ((32, 3, 2), (100000, 400, 200)):
    for seq in (0, 1, 2):
        r = subprocess.run(['timeout', '-k', '2', '30', D + '/load3', D + '/dev3_p1.ptx', DS, str(nq), str(rep), '1', str(seq), str(warm)], capture_output=True, text=True, env=env)
        if r.returncode: sys.exit('FAULT nq %d seq %d rc %d\n%s\n%s' % (nq, seq, r.returncode, r.stdout[-500:], r.stderr[-800:]))
        j = json.loads(r.stdout.strip().splitlines()[-1]); res.append(j); L = [x for x in j['launches'] if not x['cold']]; sp = sorted(x['span_ns'] for x in L)[len(L) // 2]
        m = lambda k: sorted(x[k] for x in L)[len(L) // 2]
        print('nq %6d seq %d span median %8.2f us (min %.2f max %.2f) warp cycles p10 %.0f med %.0f p90 %.0f max %.0f calls/query %.3f' % (nq, seq, sp / 1e3, min(x['span_ns'] for x in L) / 1e3, max(x['span_ns'] for x in L) / 1e3, m('wc_p10'), m('wc_med'), m('wc_p90'), m('wc_max'), L[0]['calls'] / nq), flush=True)
with open('%s/seq_%s.json' % (out, ts), 'x') as f: json.dump(res, f)
