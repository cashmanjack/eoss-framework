"""Reuse distance histograms of the node fetch stream per grouping depth (queue 5).
    reuse_run.py WORK LOG2_RAYS         (OptiX plain run per scene for the hits and the GAS; no TTU code; replay and stack distances on the CPU: model/replay/reuse.cpp)
Scenes 1e5 and 3e6 triangles, rays make_rays seed 2026 (the Finding 80/86/89 ray set), launch orders: random (file order), oracle depth 1..4, full key (stable sort by group id).
Output WORK/reuse_<scene>_<key>.txt (histograms) and WORK/reuse_meta.json. reuse_model.py turns them into LRU predictions.
"""
import json, os, subprocess, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'Memory_4'))
import morton_run as mr  # noqa: E402
M = mr.M; REP = os.path.join(HERE, '..', '..', '..', 'model', 'replay'); T = '/dev/shm/cashman3-tmp'


def main(work, lg):
    os.makedirs(work, exist_ok=True); n = 1 << int(lg); meta = {}
    for name, nt in (('1e5', 100000), ('3e6', 3000000)):
        R = mr.prep.make_rays(n); base = M + '/u_rays.bin'; R.tofile(base); ref = M + '/u_ref'
        mr.host(nt, base, ref, ['plain'], ['--dump-gas', ref + '.gas', '--dump-tris', ref + '.tris'])
        canon = np.fromfile(ref + '.0.bin', np.uint32).reshape(-1, 8); os.remove(ref + '.0.bin')
        tris = np.fromfile(ref + '.tris', np.float32).reshape(-1, 3, 3)
        W, kids, root = mr.prep.tree(open(ref + '.gas', 'rb').read())
        codes = mr.oracle_codes(W, kids, root, R, canon, tris)
        subprocess.run(['/usr/bin/python3', os.path.join(REP, 'make_table.py'), ref + '.gas', ref + '.tris', M + '/u_table.bin'], env=dict(os.environ, ASSIGN=T + '/assign'), check=True)
        meta[name] = {}
        for key in ('random', 'd1', 'd2', 'd3', 'd4', 'full'):
            if key == 'random': order = np.arange(n); ng = n
            else: gid, ng = codes[key]; order = np.argsort(gid, kind='stable')
            order.astype(np.int32).tofile(M + '/u_order.bin')
            out = os.path.join(work, 'reuse_%s_%s.txt' % (name, key))
            subprocess.run(['nice', '-n', '10', T + '/reuse', M + '/u_table.bin', base, str(n), M + '/u_order.bin', out], check=True)
            meta[name][key] = dict(groups=int(ng)); print(name, key, 'done', flush=True)
        for f in os.listdir(M):
            if f.startswith('u_'): os.remove(os.path.join(M, f))
    json.dump(meta, open(os.path.join(work, 'reuse_meta.json'), 'w'), indent=1)


if __name__ == '__main__': main(*sys.argv[1:3])
