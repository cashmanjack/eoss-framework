"""Depth 4 grouping on the 3e6 scene (and 1e5 for reference): speed of the sorted launch and node working set per SM (one L1, ~60 KB) and per TPC (two SMs, ~120 KB).
    depth4_wset.py WORK LOG2_RAYS     OptiX only (no TTU code); replay on the CPU (model/replay/wset.cpp compiled to wset2).
Resident set of one SM = distinct nodes (128 B) fetched by the rays of WRES consecutive-by-142 warps (SM s holds warps P+s+142j) from a launch in group sorted order;
a TPC holds SMs 2t and 2t+1 (assumed adjacent ids). WRES 13 (ncu average) and 48 (limit). Snapshots at 1/3 and 2/3 of the launch.
"""
import json, os, subprocess, sys, statistics as st
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'Memory_4'))
import morton_run as mr  # noqa: E402
M = mr.M; REP = os.path.join(HERE, '..', '..', '..', 'model', 'replay'); T = '/dev/shm/cashman3-tmp'

def main(work, lg):
    os.makedirs(work, exist_ok=True); n = 1 << int(lg); out = {}
    for name, nt in (('3e6', 3000000), ('1e5', 100000)):
        R = mr.prep.make_rays(n); base = M + '/w_rays.bin'; R.tofile(base); ref = M + '/w_ref'
        mr.host(nt, base, ref, ['plain'], ['--dump-gas', ref + '.gas', '--dump-tris', ref + '.tris'])
        canon = np.fromfile(ref + '.0.bin', np.uint32).reshape(-1, 8); os.remove(ref + '.0.bin')
        tris = np.fromfile(ref + '.tris', np.float32).reshape(-1, 3, 3)
        W, kids, root = mr.prep.tree(open(ref + '.gas', 'rb').read())
        codes = mr.oracle_codes(W, kids, root, R, canon, tris)
        subprocess.run(['/usr/bin/python3', os.path.join(REP, 'make_table.py'), ref + '.gas', ref + '.tris', M + '/w_table.bin'], env=dict(os.environ, ASSIGN=T + '/assign'), check=True)
        nodes_total = None; out[name] = {}
        # timing: random, depth 1..4, full, 10 launches each, results checked against the random order result
        t = {}
        for key in ('random', 'd1', 'd2', 'd3', 'd4', 'full'):
            order = np.arange(n) if key == 'random' else mr.order_by(codes[key][0])
            f = M + '/w_sorted.bin'; R[order].tofile(f)
            r = mr.host(nt, f, M + '/w_o', ['plain'])[0]; o = np.fromfile(M + '/w_o.0.bin', np.uint32).reshape(-1, 8); os.remove(M + '/w_o.0.bin')
            t[key] = dict(ms=st.median(r['ms']), wrong=int((o != canon[order]).any(1).sum()))
        for key in t: t[key]['speedup'] = t['random']['ms'] / t[key]['ms']
        out[name]['timing'] = t; print(name, 'timing', json.dumps(t), flush=True)
        for d in (1, 2, 3, 4):
            gid, ng = codes['d%d' % d]; order = np.argsort(gid, kind='stable'); gid.astype(np.int32).tofile(M + '/w_grp.bin')
            nw = n // 32; res = {}
            for wres in (13, 48):
                sets = []   # set id lists: SM sets then TPC sets
                setid_sm = -np.ones(n, np.int32); setid_tpc = -np.ones(n, np.int32)
                for sn, P in enumerate((nw // 3, 2 * nw // 3)):
                    for s_ in range(142):
                        for j in range(wres):
                            w_ = P + s_ + 142 * j
                            if (w_ + 1) * 32 > n: continue
                            r_ = order[w_ * 32:(w_ + 1) * 32]; setid_sm[r_] = sn * 142 + s_; setid_tpc[r_] = sn * 71 + s_ // 2
                row = {}
                for lab, sid, ns_ in (('sm', setid_sm, 284), ('tpc', setid_tpc, 142)):
                    sid.tofile(M + '/w_set.bin')
                    subprocess.run([T + '/wset2', M + '/w_table.bin', base, str(n), '0', '-', M + '/w_grp.bin', str(ng), M + '/w_out.txt', M + '/w_set.bin', str(ns_), M + '/w_setout.txt'], check=True)
                    kb = np.loadtxt(M + '/w_setout.txt', comments='#', ndmin=2)[:, 1] * 128 / 1024
                    row[lab] = dict(mean_kb=float(kb.mean()), min_kb=float(kb.min()), max_kb=float(kb.max()))
                    nodes_total = int(open(M + '/w_out.txt').readline().split()[2])
                res[wres] = row
            out[name][d] = dict(groups=int(ng), rays_per_group=float(n / ng), resident=res)
            print(name, 'depth', d, json.dumps(out[name][d]), flush=True)
        out[name]['nodes_kb_total'] = nodes_total * 128 / 1024
    json.dump(out, open(os.path.join(work, 'depth4_wset_%s.json' % lg), 'w'), indent=1)
    for f in os.listdir(M):
        if f.startswith('w_'): os.remove(os.path.join(M, f))

if __name__ == '__main__': main(*sys.argv[1:3])
