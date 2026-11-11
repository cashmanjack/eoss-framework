"""Replay validation: CPU traversal replay vs measured queries per ray (resume loop kernel, existing checked group) and per element costs.

    validate.py WORK [measure]       measure: run the GPU part (TTU kernel, existing builds only) and keep the per ray rounds; without it only the CPU part runs on saved data.
Scenes soup:100000 and soup:3000000, G = 71 blocks, R = 2 rounds of rays (127,232 rays), rays = prep.make_rays(n).
Replay variants: order dfs / pq, query rule A (every hit return with any pending entry resumes) / B (only pending entries not culled by the new best).
Per element costs: Latency_3/4 numbers (node 343, leaf 113, triangle 52 cycles; fixed 740 per query), Memory_3 bytes (176 per query + 128 per node + 43 per triangle).
"""
import json, os, subprocess, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', '..')
sys.path.insert(0, os.path.join(ROOT, 'ttu', 'bench', 'Throughput_6L')); sys.path.insert(0, os.path.join(ROOT, 'ttu', 'bench', 'Memory_4')); sys.path.insert(0, os.path.join(ROOT, 'ttu', 'bench', 'Morton')); sys.path.insert(0, HERE)
M = '/dev/shm/m2'
SCENES = [('1e5', 'soup:100000', 100000), ('3e6', 'soup:3000000', 3000000)]
G, R = 71, 2


def measure(work):
    import long_run as T
    for name, scene, n_tri in SCENES:
        T.run(work, scene, 'rnd', G, R, tag='replay_%s' % name, keep=os.path.join(work, 'replay_%s' % name))


def dump_tables(work):
    import morton_run as MR
    for name, scene, n_tri in SCENES:
        rays = M + '/r_tiny.bin'; np.zeros((1024, 8), np.float32).tofile(rays)
        MR.host(n_tri, rays, M + '/tiny', ['plain'], ['--dump-gas', M + '/g_%s.gas' % name, '--dump-tris', M + '/g_%s.tris' % name])
        env = dict(os.environ, ASSIGN=M + '/assign')
        subprocess.run([sys.executable, os.path.join(HERE, 'make_table.py'), M + '/g_%s.gas' % name, M + '/g_%s.tris' % name, M + '/table_%s.bin' % name], env=env, check=True)


def compare(work):
    import prep
    n = G * 896 * R
    rays = prep.make_rays(n); rays.tofile(M + '/rays_val.bin')
    for name, scene, n_tri in SCENES:
        fin = np.fromfile(os.path.join(work, 'replay_%s.fin' % name), np.uint32).reshape(-1, 14)
        ref = np.fromfile(os.path.join(work, 'replay_%s.ref' % name), np.uint32).reshape(-1, 8)
        qm = 32 - fin[:, 13].astype(np.int64)   # measured queries (rounds) per ray
        for order in (0, 1):
            out = M + '/val_%s_%d.bin' % (name, order); pq3 = M + '/val_%s_%d.q3' % (name, order)
            subprocess.run([M + '/replay', M + '/table_%s.bin' % name, M + '/rays_val.bin', str(n), str(order), out, pq3], check=True)
            o = np.fromfile(out, np.int32).reshape(-1, 8); q3 = np.fromfile(pq3, np.int32).reshape(-1, 8, 3)
            rh = ref[:, 0] != 0; rt = ref[:, 1].view(np.float32)
            hit_ok = ((o[:, 5] >= 0) == rh) & (~rh | (np.abs(o[:, 6].view(np.float32) - rt) <= 1e-5 * np.maximum(1, rt)))
            print('\n%s %s: replay hit vs OptiX: %.2f%% of rays equal' % (scene, ['dfs', 'pq'][order], 100 * hit_ok.mean()))
            print('  mean per ray: nodes %.2f leaves %.2f triangles %.2f hits %.2f' % tuple(o[:, [0, 1, 2, 7]].mean(0)))
            for vn, qp in (('A', o[:, 3]), ('B', o[:, 4])):
                ok = hit_ok
                print('  queries per ray rule %s: replay %.3f vs measured %.3f; equal on %.1f%% of rays, |diff| <= 1 on %.1f%%, correlation %.3f' % (
                    vn, qp.mean(), qm.mean(), 100 * (qp == qm).mean(), 100 * (abs(qp - qm) <= 1).mean(), np.corrcoef(qp, qm)[0, 1]))
            qp = o[:, 3]
            print('  distribution measured %s | replay A %s' % (np.round(np.bincount(qm, minlength=7)[1:7] / n, 3), np.round(np.bincount(qp, minlength=7)[1:7] / n, 3)))
            by = (176 * qp + 128 * o[:, 0] + 43 * o[:, 2]).mean()
            print('  predicted bytes per ray (176 q + 128 node + 43 tri) %.0f = %.0f sectors; measured RT L1 sectors per ray (plateau check, interleaved): 1e5 82, 3e6 108' % (by, by / 32))
            # per element cost model: serial latency of one ray vs measured warp rounds is not comparable; give the predicted cycles per ray (single lane, warm L1 costs)
            cyc = (740 * qp + 343 * o[:, 0] + 113 * o[:, 1] + 52 * o[:, 2]).mean()
            print('  predicted serial cycles per ray (740 q + 343 node + 113 leaf + 52 tri): %.0f' % cyc)


if __name__ == '__main__':
    work = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2] == 'measure': dump_tables(work); measure(work)
    compare(work)
