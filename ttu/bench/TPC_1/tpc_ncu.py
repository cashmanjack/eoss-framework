"""Option (b), ncu part (queue 4): the RT unit's L1 sector counts and hit rate with the checked traversal group, RT work alone (ncu serializes kernels, so no sibling compute here).

    tpc_ncu.py WORK        (needs /dev/shm/t9/setup3, setup3_dev.ptx, words.txt)
RT sectors in L1 = l1tex__t_sectors - l1tex__t_sectors_pipe_lsu (Memory_5 method); RT hits = lookup_hit - lsu lookup_hit. Summed over all 142 L1s (ncu gives no per instance value).
Runs: soup:100000 random rays and tri1 coherent, G 71 (RT on the even SMs) and G 142, and G 71 with 32 KB dynamic shared memory reserved on the RT block (shrinks the L1 of the even SMs only; the odd SMs' L1s stay as they are).
Order: stage 0 and stage 1 tiny first (as tpc1_run), then the profiled runs (R 16 outer rounds: 0.7 million rays per launch at G 71, launch 2 of 2 profiled). Every profiled run also compares every ray with OptiX.
"""
import csv, json, os, sys, shutil
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tpc1_run as T  # noqa: E402
from tpc1_run import build, D, LANES, S3, sh  # noqa: E402

M = ['l1tex__t_sectors.sum', 'l1tex__t_sectors_pipe_lsu.sum', 'l1tex__t_sectors_lookup_hit.sum', 'l1tex__t_sectors_pipe_lsu_lookup_hit.sum', 'gpu__time_duration.sum']


def run(binp, rays, out, scene, G, R, extra, ncu_csv=None):
    cmd = [sys.executable, os.path.join(S3, 'run.py'), binp, 'file:' + rays, 'zero', out, '--scene', scene, '--warps', str(T.W), '--blockrays', '--slim', '--loop', '--blocks', str(G), '--outer', str(R), '--repeat', '3', '--quick'] + extra
    if ncu_csv: cmd += ['--ncu', ','.join(M), '--ncu-out', ncu_csv, '--ncu-skip', '1']
    p = sh(cmd)
    if not ('exit 0' in p.stdout and ('no error' in p.stdout or ncu_csv)): sys.exit('FAILED %s\n%s\n%s' % (out, p.stdout[-2500:], p.stderr[-1500:]))
    return p.stdout


def parse(csvf):
    rows = list(csv.reader(open(csvf))); i = next(k for k, r in enumerate(rows) if r and r[0] == 'ID'); h = rows[i]; v = {}
    for r in rows[i + 2:]:
        if len(r) == len(h) and r[h.index('Kernel Name')].startswith('ttu_one'):
            for m in M: v[m] = float(r[h.index(m)].replace(',', ''))
    return v


def main(work):
    os.makedirs(work, exist_ok=True); os.makedirs(D, exist_ok=True); R = 16
    shutil.copy('/dev/shm/t9/words.txt', D); shutil.copy('/dev/shm/t9/setup3_dev.ptx', D)
    b = {}
    for G in (71, 142):
        for stg in (0, 1): b[(stg, G)] = '%s/s%d_g%d' % (D, stg, G); build(stg, G, R, b[(stg, G)])
    build(1, 71, 2, D + '/s1_tiny'); build(0, 71, 2, D + '/s0_tiny')
    res = []
    for scene, kind in (('soup:100000', 'rnd'), ('tri1', 'coh')):
        tag = scene.replace(':', ''); n_max = 142 * LANES * R; rays = '%s/rays_%s.bin' % (D, tag)
        (T.make_rays(n_max) if kind == 'rnd' else np.repeat(np.array([[0, 0, -1, 0, 0, 0, 1, 1e16]], np.float32), n_max, axis=0)).tofile(rays)
        small = '%s/rays_small_%s.bin' % (D, tag); np.fromfile(rays, np.float32, count=71 * LANES * 2 * 8).tofile(small)
        run(D + '/s0_tiny', small, '%s/s0_tiny_%s.json' % (work, tag), scene, 71, 2, ['--dump-tris', D + '/t.bin'])
        run(D + '/s1_tiny', small, '%s/s1_tiny_%s.json' % (work, tag), scene, 71, 2, [])
        print(tag, 'tiny stage 0 and 1 ok', flush=True)
        for name, G, ex in (('alone71', 71, []), ('alone142', 142, []), ('alone71_smem32k', 71, ['--smem', '32768'])):
            rf = '%s/rays_g%d_%s.bin' % (D, G, tag); np.fromfile(rays, np.float32, count=G * LANES * R * 8).tofile(rf)
            if ex: run(b[(0, G)], rf, '%s/s0_%s_%s.json' % (work, name, tag), scene, G, R, ex)   # stage 0 with the reservation first
            csvf = '%s/%s_%s.csv' % (work, name, tag); fin = D + '/fin.bin'; ref = D + '/ref.bin'
            run(b[(1, G)], rf, '%s/%s_%s.json' % (work, name, tag), scene, G, R, ex + ['--optix-repeat', '1', '--dump-final', fin, '--dump-ref', ref], csvf)
            a = np.fromfile(fin, np.uint32).reshape(-1, 14); r = np.fromfile(ref, np.uint32).reshape(-1, 8)
            hit = a[:, 0] != 0; rh = r[:, 0] != 0; both = hit & rh
            bad = int((hit != rh).sum()) + int((both & (np.abs(a[:, 4].view(np.float32)) != r[:, 1].view(np.float32))).sum()) + int((both & ((a[:, 3] & 0x1fffffff).astype(np.int64) != r[:, 2].astype(np.int64))).sum()); unst = int((a[:, 13] == 0x77777777).sum())
            v = parse(csvf); rt = v[M[0]] - v[M[1]]; rth = v[M[2]] - v[M[3]]; n = G * LANES * R
            row = dict(scene=tag, config=name, G=G, rays=n, wrong=bad, unstored=unst, rt_sectors=rt, rt_hits=rth, rt_hit_rate=rth / rt if rt else None, rt_sectors_per_ray=rt / n, lsu_sectors=v[M[1]], dur_ns=v[M[4]], raw=v)
            res.append(row)
            print('  %-18s G %3d wrong %d unstored %d | RT L1 sectors/ray %.2f, hit rate %.3f | lsu sectors/ray %.2f | %.1f ms' % (name, G, bad, unst, row['rt_sectors_per_ray'], row['rt_hit_rate'] or 0, v[M[1]] / n, v[M[4]] / 1e6), flush=True)
            if bad or unst: sys.exit('WRONG RESULT: stop')
            for f in (fin, ref, rf):
                if os.path.exists(f): os.remove(f)
        os.remove(rays)
    json.dump(res, open(os.path.join(work, 'tpc_ncu.json'), 'w'), indent=1)
    shutil.rmtree(D)


if __name__ == '__main__': main(sys.argv[1])
