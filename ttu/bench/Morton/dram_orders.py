"""Rungholt DRAM bytes per ray order with ncu (OptiX only, no TTU code; 2026-11-17, tethys). Predictions: model/predictions_rungholt_ncu.md (committed before the ncu step).

    dram_orders.py build WORK SCENE.tris      same random rays (seed 2026, 2^23) and same orders as random_check.py predict (F205), WITHOUT the replay; saves WORK/ck.npz
    dram_orders.py ncu WORK SCENE.tris TMPDIR    per order: write the rays, ncu on the morton host (repeat 3), the second optixLaunch only (the first is the host's untimed warm up), raw CSV timestamped, mode x
Env as the other Morton scripts: MORTON_DIR (host + ptx), TTU_GPU, CUDA_DEVICE_ORDER=PCI_BUS_ID, CUDA_VISIBLE_DEVICES unset."""
import json, os, subprocess, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, '..', '..', '..')
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'Memory_4'))
import morton_run as mr, random_check as rc  # noqa: E402
M = mr.M
METRICS = 'dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum,lts__t_sectors.sum,lts__t_sector_hit_rate.pct,dram__throughput.avg.pct_of_peak_sustained_elapsed'
NAMES = ['natural', 'morton-o k5', 'oracle d1', 'oracle d2', 'oracle d3', 'oracle d4', 'oracle full']


def build(work, scene):
    os.makedirs(work, exist_ok=True); n = 1 << 23
    R = rc.make_rays(scene, n); base = M + '/d_rays.bin'; R.tofile(base); ref = M + '/d_ref'
    mr.host(scene, base, ref, ['plain'], ['--dump-gas', ref + '.gas', '--dump-tris', ref + '.tris'])
    canon = np.fromfile(ref + '.0.bin', np.uint32).reshape(-1, 8); tris = np.fromfile(ref + '.tris', np.float32).reshape(-1, 3, 3)
    print('%d rays, %.1f%% hit, %d triangles, GAS %.1f MiB' % (len(R), 100 * (canon[:, 0] != 0).mean(), len(tris), os.path.getsize(ref + '.gas') / 2 ** 20), flush=True)
    Wn, kids, root = mr.prep.tree(open(ref + '.gas', 'rb').read()); codes = mr.oracle_codes(Wn, kids, root, R, canon, tris)
    Rn = rc.unit_origin(R, scene); orders = {'natural': np.arange(len(R)), 'morton-o k5': mr.order_by(mr.code_o(Rn, 5))}
    for d in ('d1', 'd2', 'd3', 'd4', 'full'): orders['oracle ' + d] = mr.order_by(codes[d][0])
    fn = os.path.join(work, 'ck.npz')
    if os.path.exists(fn): sys.exit('exists ' + fn)
    np.savez(fn, R=R, canon=canon, **{k.replace(' ', '_'): v.astype(np.int32) for k, v in orders.items()})
    for f in os.listdir(M):
        if f.startswith('d_'): os.remove(os.path.join(M, f))


def run_ncu(work, scene, tmp):
    z = np.load(os.path.join(work, 'ck.npz')); R = z['R']; ts = time.strftime('%Y%m%dT%H%M%S'); os.makedirs(tmp, exist_ok=True)
    for name in NAMES:
        perm = z[name.replace(' ', '_')].astype(np.int64); f = M + '/d_rays.bin'; R[perm].tofile(f); mr.gpu_free()
        out = os.path.join(work, 'ncu_%s_%s.csv' % (name.replace(' ', '_'), ts))
        p = subprocess.run(['timeout', '-k', '2', '300', os.path.expanduser('~/opt/cuda-12.8/bin/ncu'), '--csv', '-k', 'regex:optixLaunch', '--launch-skip', '1', '--launch-count', '1', '--cache-control', 'none', '--clock-control', 'none',
                            '--metrics', METRICS, M + '/morton', M + '/morton_dev.ptx', scene, f, M + '/d_out', '3', 'plain'], env=dict(mr.ENV, TMPDIR=tmp), capture_output=True, text=True)
        with open(out, 'x') as fh: fh.write(p.stdout)
        print(name, 'rc', p.returncode, p.stderr[-200:].strip(), flush=True)
        if os.path.exists(M + '/d_out.0.bin'): os.remove(M + '/d_out.0.bin')
    os.remove(M + '/d_rays.bin')


if __name__ == '__main__':
    a = sys.argv[1:]
    build(a[1], a[2]) if a[0] == 'build' else run_ncu(a[1], a[2], a[3])
