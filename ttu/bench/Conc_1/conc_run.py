"""Conc_1 (Phase 10, concurrent kernels), RTX 4090 / any GPU: our own checked TTU traversal kernel beside a plain persistent compute kernel (setup3 --compute), both directions.
    conc_run.py WORK GROUP REPS      GROUP: smoke | main     (REPS outer repetitions; every config once per repetition)
Needs /dev/shm/t9/{setup3, setup3_dev.ptx, words.txt} (setup3.cu with --compute TYPE:PLACE, PLACE all | odd | high | sib). NSM, HALF from the GPU (gpucfg).
RT: lean build of the checked group (as F101), W = 20 warps per block, R = 96 outer; G blocks land on SMs 0, 2, ..., 2G - 2 (placement rule, probe_placement).
Compute placement: high = SMs >= NSM/2 (other TPCs than the RT blocks when G <= NSM/4), sib = odd SMs below NSM/2 (TPC siblings of the RT SMs when G = NSM/4), odd = every odd SM (siblings of G = NSM/2 RT SMs), 4 blocks of 384 threads on each placed SM.
Checks per run: every record of the last launch vs OptiX; compute resident blocks per SM half and parity printed (RT block SMs are NOT stored in the lean build: inferred from the placement rule and the full compute SMs). Raw rows WORK/conc_<group>_<TS>.json (mode x, never overwritten).
"""
import json, os, statistics as st, sys, shutil, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); TPC = os.path.join(HERE, '..', 'TPC_1')
sys.path.insert(0, TPC); sys.path.insert(0, os.path.join(HERE, '..', '..', 'config')); sys.path.insert(0, os.path.join(HERE, '..', 'Memory_4'))
import gpucfg  # noqa: E402
import tpc1_run as T  # noqa: E402
import tpc3_run as T3  # noqa: E402
from tpc1_run import go, D, W, LANES  # noqa: E402
CFG = gpucfg.load(); NSM = CFG.sm_count; HALF = NSM // 2; Q = NSM // 4
BF, HX = ['--lean'], ['--lean']
def build(stage, G, R, out): T3.build(stage, G, R, out, BF)
def configs(group):
    c = [('alone%d' % Q, Q, None), ('alone%d' % HALF, HALF, None)]
    for t in ('alu', 'l1', 'mem'):
        c += [('hi_' + t, Q, t + ':high'), ('sib_' + t, Q, t + ':sib'), ('split_' + t, HALF, t + ':odd')]
    return [c[0], c[2], c[3]] if group == 'smoke' else c
def main(work, group, reps):
    reps = int(reps); os.makedirs(work, exist_ok=True); os.makedirs(D, exist_ok=True); R = int(os.environ.get('CONC_R', 96))
    out_json = os.path.join(work, 'conc_%s_%s.json' % (group, time.strftime('%Y%m%dT%H%M%S')))
    if os.path.exists(out_json): sys.exit('exists')
    shutil.copy('/dev/shm/t9/words.txt', D); shutil.copy('/dev/shm/t9/setup3_dev.ptx', D)
    Gs = sorted({g for _, g, _ in configs(group)}); b = {}
    for G in Gs:
        for stg in (0, 1): b[(stg, G)] = '%s/s%d_g%d' % (D, stg, G); build(stg, G, R, b[(stg, G)])
    build(1, Q, 2, D + '/s1_tiny'); build(0, Q, 2, D + '/s0_tiny')
    rows = []
    for scene, kind in (('soup:100000', 'rnd'), ('tri1', 'coh')):
        tag = scene.replace(':', ''); n_max = max(Gs) * LANES * R; rays = '%s/rays_%s.bin' % (D, tag)
        (T.make_rays(n_max) if kind == 'rnd' else np.repeat(np.array([[0, 0, -1, 0, 0, 0, 1, 1e16]], np.float32), n_max, axis=0)).tofile(rays)
        small = '%s/rays_small_%s.bin' % (D, tag); np.fromfile(rays, np.float32, count=Q * LANES * 2 * 8).tofile(small)
        go(D + '/s0_tiny', small, '%s/s0_tiny_%s_%s.json' % (work, tag, time.strftime('%Y%m%dT%H%M%S')), scene, Q, 2, ['--dump-tris', D + '/t.bin', '--lean'])
        go(D + '/s1_tiny', small, '%s/s1_tiny_%s_%s.json' % (work, tag, time.strftime('%Y%m%dT%H%M%S')), scene, Q, 2, ['--lean'])
        print(tag, 'tiny stage 0 and stage 1 ok', flush=True)
        for G in Gs: np.fromfile(rays, np.float32, count=G * LANES * R * 8).tofile('%s/rays_g%d_%s.bin' % (D, G, tag))
        os.remove(rays)
        for rep in range(reps):
            for name, G, comp in configs(group):
                n = G * LANES * R; rf = '%s/rays_g%d_%s.bin' % (D, G, tag); cx = ['--compute', comp] if comp else []
                if comp and rep == 0:
                    o = go(b[(0, G)], rf, '%s/s0_%s_%s_%s.json' % (work, name, tag, time.strftime('%Y%m%dT%H%M%S')), scene, G, R, cx + HX)
                    print('  stage 0', name, [l for l in o.splitlines() if l.startswith('compute')], flush=True)
                fin = D + '/fin.bin'; ref = D + '/ref.bin'; rj = '%s/%s_%s_r%d_%s.json' % (work, name, tag, rep, time.strftime('%Y%m%dT%H%M%S'))
                go(b[(1, G)], rf, rj, scene, G, R, cx + HX + ['--optix-repeat', '1', '--dump-final', fin, '--dump-ref', ref])
                j = json.load(open(rj)); L = j['launches'][1:]; ms = [l['ms'] for l in L]; rate = n / st.median(ms) / 1000
                a = np.fromfile(fin, np.uint32).reshape(-1, 14); r = np.fromfile(ref, np.uint32).reshape(-1, 8)
                hit = a[:, 0] != 0; rh = r[:, 0] != 0; both = hit & rh
                bad = int((hit != rh).sum()) + int((both & (np.abs(a[:, 4].view(np.float32)) != r[:, 1].view(np.float32))).sum()) + int((both & ((a[:, 3] & 0x1fffffff).astype(np.int64) != r[:, 2].astype(np.int64))).sum())
                unst = int((a[:, 13] == 0x77777777).sum())
                row = dict(scene=tag, config=name, rep=rep, G=G, rays=n, ms=ms, rays_per_us=rate, wrong=bad, unstored=unst, compute=comp, raw=os.path.basename(rj))
                if comp:
                    rr = [l['citers'] / l['cwin_s'] for l in L if l.get('cwin_s')]; row['compute_with_rt'] = st.median(rr); row['compute_alone'] = j['compute_alone_iters'] / j['compute_alone_s']
                    for k in ('even', 'odd', 'low', 'high'): row['blocks_' + k] = j['compute_blocks_' + k]
                rows.append(row)
                print('  r%d %-10s %-12s G %3d RT %7.1f rays/us wrong %d unstored %d %s' % (rep, tag, name, G, rate, bad, unst, '' if not comp else ' | compute %.3fx alone | blocks even %d odd %d low %d high %d' % (row['compute_with_rt'] / row['compute_alone'], row['blocks_even'], row['blocks_odd'], row['blocks_low'], row['blocks_high'])), flush=True)
                for f in (fin, ref):
                    if os.path.exists(f): os.remove(f)
                with open(out_json + '.tmp', 'w') as f: json.dump(rows, f, indent=1)
                if bad or unst: sys.exit('STOP: wrong result or unstored record in %s %s rep %d' % (tag, name, rep))
        for f in os.listdir(D):
            if f.startswith('rays_g'): os.remove(os.path.join(D, f))
    os.rename(out_json + '.tmp', out_json); shutil.rmtree(D)
if __name__ == '__main__': main(*sys.argv[1:4])
