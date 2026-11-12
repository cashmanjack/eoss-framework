"""TPC_1 with the checked traversal group (stage 0 first): RT work on even SMs and compute on odd SMs, against both on every SM. One process, two streams.

    tpc1_run.py WORK            (builds in /dev/shm/t9c, deleted after; needs /dev/shm/t9/setup3 built from setup3.cu with --compute and /dev/shm/t9/words.txt)
RT: persistent loop kernel (build_patch --slim --remap --loop 32 --outer R), W = 20 warps per block (640 threads, 72 registers: a second RT block never fits on an SM), n = G x 640 x R rays,
    scenes soup:100000 with random rays and tri1 with one coherent ray. Compute: plain kernel on stream 2 (alu | l1 | mem), 384 thread blocks, 24 registers, 1 GB buffer for mem.
Configs (compute blocks per SM capped by the kernel: 2 on all SMs, 4 on odd SMs): alone142 (RT on all SMs, G 142), alone71 (G 71: blocks land on the even SMs, one per SM), shared (G 142 + compute 2 blocks per SM, all SMs), split (G 71 + compute 4 blocks per odd SM).
Every run reports compute alone (300 ms before the RT launches, same placement) and compute during each RT launch (iterations / host window around the launch).
Checks: every record of the last launch vs OptiX (hit, t, primitive); stage 0 first for each placement (no TTU instruction, compute placement printed).
"""
import json, os, statistics as st, subprocess, sys, shutil
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
S3 = os.path.join(HERE, '..', '..', 'harness', 'setup3')
sys.path.insert(0, os.path.join(HERE, '..', 'Memory_4'))
from prep import make_rays  # noqa: E402
D = '/dev/shm/t9c'; BASE = '/dev/shm/t9/setup3'
VENV = os.path.expanduser('~/eoss/.venv/bin/python')
W = 20; LANES = W * 32; K = 32


def sh(c, **kw): return subprocess.run(c, capture_output=True, text=True, **kw)


def build(stage, G, R, out):
    cmd = [VENV, os.path.join(S3, 'build_patch.py'), '--stage', str(stage), '--bin', BASE, '--slim', '--remap', '--compact-out', '--regs', '72', '--max-out', '61',
           '--loop', str(K), '--outer', str(R), '--stride', str(G * LANES * 96), '--out', out]
    p = sh(cmd)
    if p.returncode: sys.exit(p.stdout + p.stderr)
    c = sh([VENV, os.path.join(S3, 'check_seq.py'), out])
    if 'ALL PASS' not in c.stdout.splitlines()[-1]: sys.exit(c.stdout[-2000:])


def go(binp, rays, out, scene, G, R, extra):
    cmd = [sys.executable, os.path.join(S3, 'run.py'), binp, 'file:' + rays, 'zero', out, '--scene', scene, '--warps', str(W), '--blockrays', '--slim', '--loop', '--blocks', str(G), '--outer', str(R), '--repeat', '6', '--quick'] + extra
    p = sh(cmd); ok = 'exit 0' in p.stdout and 'no error' in p.stdout
    if not ok: sys.exit('FAILED %s\n%s\n%s' % (out, p.stdout[-2500:], p.stderr[-1500:]))
    return p.stdout


def main(work):
    os.makedirs(work, exist_ok=True); os.makedirs(D, exist_ok=True); R = int(os.environ.get('TPC_R', 96))
    shutil.copy('/dev/shm/t9/words.txt', D); shutil.copy('/dev/shm/t9/setup3_dev.ptx', D)
    b = {}
    for G in (71, 142):
        for stg in (0, 1): b[(stg, G)] = '%s/s%d_g%d' % (D, stg, G); build(stg, G, R, b[(stg, G)])
    # smallest TTU launch first: stage 1, G 71, tiny R (separate build), no compute
    build(1, 71, 2, D + '/s1_tiny'); build(0, 71, 2, D + '/s0_tiny')
    rows = []
    for scene, kind in (('soup:100000', 'rnd'), ('tri1', 'coh')):
        tag = scene.replace(':', '')
        n_max = 142 * LANES * R; rays = '%s/rays_%s.bin' % (D, tag)
        (make_rays(n_max) if kind == 'rnd' else np.repeat(np.array([[0, 0, -1, 0, 0, 0, 1, 1e16]], np.float32), n_max, axis=0)).tofile(rays)
        # tiny: rays file must hold exactly G x LANES x R rays: separate small file
        small = '%s/rays_small_%s.bin' % (D, tag); np.fromfile(rays, np.float32, count=71 * LANES * 2 * 8).tofile(small)
        go(D + '/s0_tiny', small, '%s/s0_tiny_%s.json' % (work, tag), scene, 71, 2, ['--dump-tris', D + '/t.bin'])
        go(D + '/s1_tiny', small, '%s/s1_tiny_%s.json' % (work, tag), scene, 71, 2, [])
        print(tag, 'tiny stage 0 and stage 1 ok', flush=True)
        for name, G, comp in [('alone142', 142, None), ('alone71', 71, None)] + [(p + '_' + t, 142 if p == 'shared' else 71, '%s:%s' % (t, 'all' if p == 'shared' else 'odd')) for t in ('alu', 'l1', 'mem') for p in ('shared', 'split')]:
            n = G * LANES * R; rf = '%s/rays_g%d_%s.bin' % (D, G, tag)
            if not os.path.exists(rf): np.fromfile(rays, np.float32, count=n * 8).tofile(rf)
            cx = ['--compute', comp] if comp else []
            if comp:   # stage 0 first: compute placement, no TTU instruction
                o = go(b[(0, G)], rf, '%s/s0_%s_%s.json' % (work, name, tag), scene, G, R, cx)
                print('  stage 0', name, [l for l in o.splitlines() if l.startswith('compute')], flush=True)
            fin = D + '/fin.bin'; ref = D + '/ref.bin'
            o = go(b[(1, G)], rf, '%s/%s_%s.json' % (work, name, tag), scene, G, R, cx + ['--optix-repeat', '1', '--dump-final', fin, '--dump-ref', ref])
            j = json.load(open('%s/%s_%s.json' % (work, name, tag))); L = j['launches'][1:]
            ms = [l['ms'] for l in L]; rate = n / st.median(ms) / 1000
            a = np.fromfile(fin, np.uint32).reshape(-1, 14); r = np.fromfile(ref, np.uint32).reshape(-1, 8)
            hit = a[:, 0] != 0; rh = r[:, 0] != 0; both = hit & rh
            bad = int((hit != rh).sum()) + int((both & (np.abs(a[:, 4].view(np.float32)) != r[:, 1].view(np.float32))).sum()) + int((both & ((a[:, 3] & 0x1fffffff).astype(np.int64) != r[:, 2].astype(np.int64))).sum())
            unst = int((a[:, 13] == 0x77777777).sum())
            row = dict(scene=tag, config=name, G=G, rays=n, ms=ms, rays_per_us=rate, wrong=bad, unstored=unst, compute=comp)
            if comp:
                rr = [l['citers'] / l['cwin_s'] for l in L if l.get('cwin_s')]; row['compute_with_rt'] = st.median(rr); row['compute_alone'] = j['compute_alone_iters'] / j['compute_alone_s']
                row['blocks_even'] = j['compute_blocks_even']; row['blocks_odd'] = j['compute_blocks_odd']
            rows.append(row)
            print('  %-12s G %3d  RT %8.3f ms %7.1f rays/us (%.2f per SM-slot) wrong %d unstored %d%s' % (name, G, st.median(ms), rate, rate / G, bad, unst,
                  '' if not comp else ' | compute %.3g it/s with RT, %.3g alone (%.2fx) | blocks even %d odd %d' % (row['compute_with_rt'], row['compute_alone'], row['compute_with_rt'] / row['compute_alone'], row['blocks_even'], row['blocks_odd'])), flush=True)
            for f in (fin, ref):
                if os.path.exists(f): os.remove(f)
        os.remove(rays)
        for f in os.listdir(D):
            if f.startswith('rays_g'): os.remove(os.path.join(D, f))
    json.dump(rows, open(os.path.join(work, 'tpc1_ttu.json'), 'w'), indent=1)
    shutil.rmtree(D)


if __name__ == '__main__': main(sys.argv[1])
