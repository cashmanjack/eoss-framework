"""Mochi options arm (model/FORM_options_arm.md). python3 load4_run.py OUTDIR (GPU 1, OptiX only, no TTU code). Smoke launch of 32 queries per arm first, then 100,000 queries, 400 timed after 200 warmups, IAS on, SEQ 0.
Two rounds, arm order rotated. Stops at the first non zero exit. Raw json timestamped, mode x."""
import json, os, subprocess, sys, time
D = '/dev/shm/ld4'; OLD = '/dev/shm/ld/dev3_p1.ptx'; DS = os.path.expanduser('~/Mochi-DCD-on-RT/datasets/100k_particles.txt')
ARMS = [('base', OLD, 0), ('refac', D + '/b_p1.ptx', 0), ('F1_attr2', OLD, 1), ('F2_sphere', OLD, 2), ('F3_depth2', OLD, 4), ('F4_stack', OLD, 8), ('F5_sbt', D + '/sbt_p1.ptx', 16), ('F6_sm52', D + '/m52_p1.ptx', 0), ('ALL', D + '/all_p1.ptx', 31)]
out = sys.argv[1]; os.makedirs(out, exist_ok=True); ts = time.strftime('%Y%m%dT%H%M%S'); res = []
busy = subprocess.run(['nvidia-smi', '--id=1', '--query-compute-apps=pid', '--format=csv,noheader'], capture_output=True, text=True).stdout.strip()
if busy: sys.exit('GPU 1 has compute processes: ' + busy)
def run(name, ptx, flip, nq, rep, warm):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='1', CUDA_DEVICE_ORDER='PCI_BUS_ID', FLIP=str(flip))
    r = subprocess.run(['timeout', '-k', '2', '30', D + '/load4', ptx, DS, str(nq), str(rep), '1', '0', str(warm)], capture_output=True, text=True, env=env)
    if r.returncode: sys.exit('FAULT %s nq %d rc %d\n%s\n%s' % (name, nq, r.returncode, r.stdout[-400:], r.stderr[-800:]))
    j = json.loads(r.stdout.strip().splitlines()[-1]); j['arm'] = name; j['flip'] = flip; return j
for name, ptx, flip in ARMS: run(name, ptx, flip, 32, 3, 2); print('smoke ok', name, flush=True)
for rnd in range(2):
    for k in range(len(ARMS)):
        name, ptx, flip = ARMS[(k + rnd * 4) % len(ARMS)]; j = run(name, ptx, flip, 100000, 400, 200); j['round'] = rnd; res.append(j)
        L = [x for x in j['launches'] if not x['cold']]; m = lambda key: sorted(x[key] for x in L)[len(L) // 2]
        print('round %d %-10s span median %7.2f us (min %.2f max %.2f) warp cyc p10 %6.0f med %6.0f p90 %6.0f max %6.0f p90/med %.2f calls/q %.3f' % (rnd, name, m('span_ns') / 1e3, min(x['span_ns'] for x in L) / 1e3, max(x['span_ns'] for x in L) / 1e3, m('wc_p10'), m('wc_med'), m('wc_p90'), m('wc_max'), m('wc_p90') / m('wc_med'), L[0]['calls'] / 100000), flush=True)
        with open('%s/options_%s.json' % (out, ts), 'x' if (rnd == 0 and k == 0) else 'w') as f: json.dump(res, f)
print('raw', out, ts)
