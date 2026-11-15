"""Measured per ray counts (ncu on the OptiX launch, RTX 4070 Ti) for the validation points of the performance model, per launch order. OptiX only, no TTU code.
    term_ncu.py random WORK TAG SCENE ORDER...       TAG s3e5|s15e5|bistro, rays and orders from /dev/shm/rt/ck (random_check.py predict)
    term_ncu.py bounce WORK SCENE_N LOG2 ORDER...    bounce rays as bounce_oracle.py (regenerated, same seed)
Per ray (rays = launch size): RT L1 sectors = l1tex__t_sectors - l1tex__t_sectors_pipe_lsu (F95: the RT unit's own requests), L1 fills = l1tex__m_xbar2l1tex_read_sectors (includes the ray fetch,
about 1 sector per ray), L2 sectors = lts__t_sectors, DRAM bytes = dram__bytes_read, kernel time = gpu__time_duration. One launch under ncu (the second of 3), caches and clocks left alone."""
import csv, json, os, subprocess, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'Memory_4'))
import morton_run as mr, netsort_run as ns  # noqa: E402
M = mr.M; NCU = os.path.expanduser('~/opt/cuda-12.8/bin/ncu')
MET = 'l1tex__t_sectors.sum,l1tex__t_sectors_pipe_lsu.sum,l1tex__m_xbar2l1tex_read_sectors.sum,lts__t_sectors.sum,dram__bytes_read.sum,gpu__time_duration.sum'
SCENE = {'s3e5': '300000', 's15e5': '1500000', 'bistro': 'file:/dev/shm/scenes/tris/bistro_ext.tris'}


def ncu_launch(scene, rays_file, n_rays, csvf):
    mr.gpu_free()
    cmd = ['timeout', '-k', '2', '240', NCU, '--target-processes', 'all', '--launch-skip', '1', '--launch-count', '1', '--cache-control', 'none', '--clock-control', 'none', '--metrics', MET, '--csv', '--page', 'raw',
           '--log-file', csvf, M + '/morton', M + '/morton_dev.ptx', str(scene), rays_file, M + '/t_out', '3', 'plain']
    r = subprocess.run(cmd, env=mr.ENV, capture_output=True, text=True)
    if r.returncode: sys.exit('ncu run failed %d\n%s' % (r.returncode, (r.stdout + r.stderr)[-1500:]))
    hdr = None; row = None
    for l in open(csvf):
        if l.startswith('"ID"'): hdr = next(csv.reader([l]))
        elif hdr and l.startswith('"') and row is None: row = dict(zip(hdr, next(csv.reader([l]))))
    row = dict(zip(hdr, next(csv.reader([l for l in open(csvf) if l.startswith('"') and not l.startswith('"ID"')][1:2]))))
    v = {k: float(row[k].replace(',', '')) for k in MET.split(',')}
    for f in os.listdir(M):
        if f.startswith('t_out'): os.remove(os.path.join(M, f))
    return dict(rt_l1_sectors=(v['l1tex__t_sectors.sum'] - v['l1tex__t_sectors_pipe_lsu.sum']) / n_rays, l1_fills=v['l1tex__m_xbar2l1tex_read_sectors.sum'] / n_rays,
                l2_sectors=v['lts__t_sectors.sum'] / n_rays, dram_bytes=v['dram__bytes_read.sum'] / n_rays, ms=v['gpu__time_duration.sum'] / 1e6)


def random_sets(work, tag, orders):
    z = np.load('/dev/shm/rt/ck/ck_%s.npz' % tag); R = z['R']; out = {}
    for o in orders:
        perm = z[o.replace(' ', '_')].astype(np.int64); f = M + '/t_rays.bin'; R[perm].tofile(f)
        out[o] = ncu_launch(SCENE[tag], f, len(R), os.path.join(work, 'ncu_%s_%s.csv' % (tag, o.replace(' ', '_')))); print(tag, o, {k: round(x, 2) for k, x in out[o].items()}, flush=True)
    os.remove(M + '/t_rays.bin'); return out


def bounce_sets(work, scene_n, lg, orders):
    n = 1 << int(lg); R = ns.make(int(scene_n), n, 'bounce'); N = len(R); base = M + '/t_base.bin'; R.tofile(base); ref = M + '/t_ref'
    mr.host(int(scene_n), base, ref, ['plain'], ['--dump-gas', ref + '.gas', '--dump-tris', ref + '.tris'])
    canon = np.fromfile(ref + '.0.bin', np.uint32).reshape(-1, 8); tris = np.fromfile(ref + '.tris', np.float32).reshape(-1, 3, 3)
    W, kids, root = mr.prep.tree(open(ref + '.gas', 'rb').read()); codes = mr.oracle_codes(W, kids, root, R, canon, tris); rng = np.random.default_rng(7)
    allo = {'natural': np.arange(N), 'shuffled': rng.permutation(N), 'morton-o k4': mr.order_by(mr.code_o(R, 4)), 'morton-o k5': mr.order_by(mr.code_o(R, 5))}
    for d in (1, 2, 3): allo['oracle d%d' % d] = mr.order_by(codes['d%d' % d][0])
    allo['oracle full'] = mr.order_by(codes['full'][0]); out = {}
    for o in orders:
        f = M + '/t_rays.bin'; R[allo[o]].tofile(f)
        out[o] = ncu_launch(int(scene_n), f, N, os.path.join(work, 'ncu_b%s_%s.csv' % (scene_n, o.replace(' ', '_')))); print('bounce', scene_n, o, {k: round(x, 2) for k, x in out[o].items()}, flush=True)
    for f in os.listdir(M):
        if f.startswith('t_'): os.remove(os.path.join(M, f))
    return out


if __name__ == '__main__':
    a = sys.argv[1:]; os.makedirs(a[1], exist_ok=True)
    if a[0] == 'random': res = random_sets(a[1], a[2], a[4:]); name = a[2]
    else: res = bounce_sets(a[1], a[2], a[3], a[4:]); name = 'bounce%s' % a[2]
    json.dump(res, open(os.path.join(a[1], 'ncu_%s.json' % name), 'w'), indent=1)
