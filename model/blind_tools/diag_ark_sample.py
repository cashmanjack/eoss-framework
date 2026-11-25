"""DIAGNOSTIC on an already measured dataset (dry run only): how do the Arkade callback cycles (f0), calls and kernel time of the probe build change with the number of queries in the launch?
    diag_ark_sample.py DATA NAME --ns 250,500,1000,2500,5000,10000
Uses the CP-ARK inputs of RULE.md (seed 7 queries, radius rule). Writes model/blind/dry_run_NAME/diag_ark_sample_<ts>.json (never overwritten). Not part of the blind procedure."""
import os, sys, json, time, re, argparse, statistics, numpy as np
sys.argv = [sys.argv[0], 'calibrate', 'diag'] + sys.argv[1:]
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ap0 = argparse.ArgumentParser(); ap0.add_argument('step'); ap0.add_argument('name'); ap0.add_argument('data'); ap0.add_argument('dname'); ap0.add_argument('--ns', default='250,500,1000,2500,5000,10000'); ap0.add_argument('--gpu', default='1')
a0 = ap0.parse_args()
sys.argv = [sys.argv[0], 'calibrate', a0.dname, '--data', a0.data, '--dry-run', '--no-commit', '--no-push', '--gpu', a0.gpu]
import types
P = types.ModuleType('bp'); P.__file__ = os.path.join(HERE, 'blind_primary.py')   # run the tool's definitions without its last line (the step dispatch); the tool itself is not modified
exec(compile(''.join(open(P.__file__).read().rstrip().splitlines(True)[:-1]), P.__file__, 'exec'), P.__dict__)
pts = P.load(a0.data); N = len(pts); P.warm_ready(); p2, tree, Q, r0, R = P.ark_inputs(pts, N); out = dict(label='DIAGNOSTIC (dry run dataset)', data=os.path.basename(a0.data), N=N, radius=R, rows={})
for ns in [int(x) for x in a0.ns.split(',')]:
    sf = '/dev/shm/diag_ark_%d.txt' % os.getpid(); P.ark_file(p2, p2[Q[:ns]], sf); rs = []
    for i in range(3):
        x = P.run([P.A.ark_probe, sf, str(N), str(ns), repr(R)], os.path.dirname(P.A.ark_probe), {'DEVOUT': '1'}, 300); txt = P.strip(x); x.update(P.ark_parse(txt))
        rs.append(dict(kernel_ms=x['kernel_ms'], f0_cycles=x['f0_cycles'], f0_calls=x['f0_calls'], full_clock=x['clock']['full_clock']))
    os.remove(sf); m = lambda k: statistics.median([r[k] for r in rs if r[k] is not None])
    out['rows'][str(ns)] = dict(runs=rs, kernel_ms=m('kernel_ms'), f0_cycles=m('f0_cycles'), f0_calls=m('f0_calls'), cycles_per_call=m('f0_cycles') / m('f0_calls'), cycles_per_query=m('f0_cycles') / ns)
    print('ns %d: kernel %.4f ms, f0 %.4g cycles, calls %.0f, cycles/call %.0f, cycles/query %.0f' % (ns, m('kernel_ms'), m('f0_cycles'), m('f0_calls'), out['rows'][str(ns)]['cycles_per_call'], out['rows'][str(ns)]['cycles_per_query']), flush=True)
f = os.path.join(P.bc.REPO, 'model', 'blind', 'dry_run_' + a0.dname, 'diag_ark_sample_%s.json' % time.strftime('%Y%m%dT%H%M%S'))
with open(f, 'x') as fh: json.dump(out, fh, indent=1)
print(f)
