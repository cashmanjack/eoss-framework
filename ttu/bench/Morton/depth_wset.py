"""Grouping depth (oracle key prefix) vs the node cache: per group node working set (CPU replay over the decoded BVH) and groups resident per SM.
    depth_wset.py WORK LOG2_RAYS         (one GPU plain OptiX run per scene for the hits and the GAS; no TTU code; replay on the CPU)
Groups: depth 1, 2, 3 ancestor of the hit subtree (prep/oracle_codes of morton_run). Per group, over all its rays (root started traversals, as OptiX does):
  nodes_union = distinct internal nodes (128 bytes) fetched by any ray of the group; nodes_hot = fetched by at least 5% of the group's rays; leaf slots entered.
Residency (simulation): rays launched in group sorted order, 32 per warp, warp w goes to SM w mod 142, an SM holds WRES warps (13: ncu achieved 13.4 for OptiX at 1.27M rays; 48 = hardware limit);
  resident groups per SM = distinct groups among its first WRES warps, averaged over launch positions (window start every 142*WRES warps).
"""
import json, os, subprocess, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'Memory_4'))
import morton_run as mr  # noqa: E402
M = mr.M; REP = os.path.join(HERE, '..', '..', '..', 'model', 'replay'); T = '/dev/shm/cashman3-tmp'


def main(work, lg):
    os.makedirs(work, exist_ok=True); n = 1 << int(lg); out = {}
    for name, nt in (('1e5', 100000), ('3e6', 3000000)):
        R = mr.prep.make_rays(n); base = M + '/w_rays.bin'; R.tofile(base); ref = M + '/w_ref'
        res = mr.host(nt, base, ref, ['plain'], ['--dump-gas', ref + '.gas', '--dump-tris', ref + '.tris'])
        canon = np.fromfile(ref + '.0.bin', np.uint32).reshape(-1, 8); os.remove(ref + '.0.bin')
        tris = np.fromfile(ref + '.tris', np.float32).reshape(-1, 3, 3)
        W, kids, root = mr.prep.tree(open(ref + '.gas', 'rb').read())
        codes = mr.oracle_codes(W, kids, root, R, canon, tris)
        subprocess.run([sys.executable if False else '/usr/bin/python3', os.path.join(REP, 'make_table.py'), ref + '.gas', ref + '.tris', M + '/w_table.bin'], env=dict(os.environ, ASSIGN=T + '/assign'), check=True)
        out[name] = {}
        for d in (1, 2, 3):
            gid, ng = codes['d%d' % d]; order = np.argsort(gid, kind='stable'); np.ascontiguousarray(gid.astype(np.int32)).tofile(M + '/w_grp.bin')
            # resident sets: two snapshots of the launch (at 1/3 and 2/3), SM s holds warps P + s + 142 j, j < 13 (13 warps: ncu achieved for OptiX); set id = snapshot * 142 + s
            nw = n // 32; setid = -np.ones(n, np.int32)
            for sn, P in enumerate((nw // 3, 2 * nw // 3)):
                for s_ in range(142):
                    for j in range(13):
                        w_ = P + s_ + 142 * j; setid[order[w_ * 32:(w_ + 1) * 32]] = sn * 142 + s_
            setid.tofile(M + '/w_set.bin')
            subprocess.run([T + '/wset2', M + '/w_table.bin', base, str(n), '0', '-', M + '/w_grp.bin', str(ng), M + '/w_out.txt', M + '/w_set.bin', '284', M + '/w_setout.txt'], check=True)
            sets = np.loadtxt(M + '/w_setout.txt', comments='#', ndmin=2)[:, 1] * 128 / 1024
            rows = np.loadtxt(M + '/w_out.txt', comments='#', ndmin=2)
            gs = gid[order].astype(np.int64)                 # group of the k-th launched ray
            wg = gs[: (n // 32) * 32].reshape(-1, 32)[:, 0]  # group of each warp (first lane; groups are contiguous runs)
            res = {}
            for wres in (13, 48):
                per = 142 * wres; cnt = []
                for s0 in range(0, len(wg) - per + 1, per):
                    blk = wg[s0:s0 + per].reshape(wres, 142)            # row j = j-th warp of each SM
                    cnt.append(np.mean([len(set(blk[:, s])) for s in range(142)]))
                res[wres] = float(np.mean(cnt))
            # working set of the average group, and per SM sums (upper bound: shared upper nodes counted once per group)
            u = rows[:, 2] * 128 / 1024; h = rows[:, 3] * 128 / 1024
            out[name][d] = dict(groups=int(ng), rays_per_group=float(n / ng), nodes_total=int(open(M + '/w_out.txt').readline().split()[2]),
                                union_kb_mean=float(u.mean()), union_kb_max=float(u.max()), hot_kb_mean=float(h.mean()), hot_kb_max=float(h.max()),
                                leafslots_mean=float(rows[:, 4].mean()), resident=res, sm_hot_kb_13=float(h.mean() * res[13]), sm_union_kb_13=float(u.mean() * res[13]), resident_set_kb_mean=float(sets.mean()), resident_set_kb_max=float(sets.max()), resident_set_kb_min=float(sets.min()))
            print(name, d, json.dumps(out[name][d]), flush=True)
    json.dump(out, open(os.path.join(work, 'depth_wset_%s.json' % lg), 'w'), indent=1)
    for f in os.listdir(M):
        if f.startswith('w_'): os.remove(os.path.join(M, f))


if __name__ == '__main__': main(*sys.argv[1:3])
