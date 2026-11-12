"""TPC_1 queue 6: the F92 to F94 configurations (base and sweep groups) with the LEAN RT build (env TPC_BUILD: lean (default) | lean_ld | lean_st | base), 5 repetitions as in F93/F94. Same checked traversal group (byte identical group, check_seq ALL PASS on every build).

    tpc5_run.py WORK GROUP REPS          GROUP: base | sweep | optb | smoke
needs /dev/shm/t9/setup3 (setup3.cu with --compute, --ccarve, --csmem), setup3_dev.ptx and words.txt there.
Per scene (soup:100000 random rays, tri1 coherent): stage 0 and stage 1 tiny first, then for each repetition (outer loop, so drift is spread over configs) every config:
alone142 / alone71 as RT references, and the compute configs (split = RT on even SMs G 71, compute on odd SMs; shared = both on every SM).
Stage 0 (no TTU instruction, placement check) is run for every compute config in repetition 0.
Every run: all records of the last launch vs OptiX (hit, t, primitive); any wrong or unstored record, nonzero exit or missing 'no error' stops the whole script.
"""
import json, os, statistics as st, sys, shutil
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tpc1_run as T  # noqa: E402
from tpc1_run import go as go0, D, W, LANES  # noqa: E402
import tpc3_run as T3  # noqa: E402
BN = os.environ.get('TPC_BUILD', 'lean')
BF, HX = {'base': ([], []), 'lean': (['--lean'], ['--lean']), 'lean_ld': (['--lean-ld'], []), 'lean_st': (['--lean-st'], ['--lean'])}[BN]
def build(stage, G, R, out): T3.build(stage, G, R, out, BF)
def go(binp, rays, out, scene, G, R, extra): return go0(binp, rays, out, scene, G, R, list(extra) + HX)


def cfgs(group):
    """list of (name, G, compute spec or None, extra host args)"""
    c = [('alone142', 142, None, []), ('alone71', 71, None, [])]
    def sp(name, typ, place, lim=None, kb=None, extra=()):
        spec = '%s:%s' % (typ, place) + (':%d' % lim if lim else '') + (':%d' % kb if kb else '')
        return (name, 142 if place == 'all' else 71, spec, list(extra))
    if group in ('base', 'smoke'):
        for t in ('alu', 'l1', 'mem'):
            c += [sp('shared_' + t, t, 'all'), sp('split_' + t, t, 'odd')]
        if group == 'smoke': c = c[:2] + [sp('split_l2_1', 'l2', 'odd', 1)]
    if group == 'sweep':
        for t in ('alu', 'l1w', 'l2', 'mem', 'smem'):
            for lim in (1, 2, 4):
                c.append(sp('split_%s_b%d' % (t, lim), t, 'odd', lim, 32 if t == 'l1w' else None))
    if group == 'optb':
        for kb in (8, 16, 32, 64, 128):   # L1 working set per block, 4 blocks per sibling SM
            c.append(sp('split_l1w_%dKB' % kb, 'l1w', 'odd', 4, kb))
        for carve in (0, 100):   # sibling L1 at its largest / smallest (kernel carveout + 8 KB unused dynamic shared memory per block when 100)
            c.append(sp('split_l1_carve%d' % carve, 'l1', 'odd', 4, None, ['--ccarve', str(carve)] + (['--csmem', '8192'] if carve == 100 else [])))
            c.append(sp('split_l1w_128KB_carve%d' % carve, 'l1w', 'odd', 4, 128, ['--ccarve', str(carve)] + (['--csmem', '8192'] if carve == 100 else [])))
        c.append(sp('split_alu_carve100', 'alu', 'odd', 4, None, ['--ccarve', '100', '--csmem', '8192']))   # carveout alone, no L1 use: is the change itself harmless
    return c


def main(work, group, reps):
    os.makedirs(work, exist_ok=True); os.makedirs(D, exist_ok=True); R = 96; reps = int(reps)
    shutil.copy('/dev/shm/t9/words.txt', D); shutil.copy('/dev/shm/t9/setup3_dev.ptx', D)
    b = {}
    for G in (71, 142):
        for stg in (0, 1): b[(stg, G)] = '%s/s%d_g%d' % (D, stg, G); build(stg, G, R, b[(stg, G)])
    build(1, 71, 2, D + '/s1_tiny'); build(0, 71, 2, D + '/s0_tiny')
    rows = []
    for scene, kind in (('soup:100000', 'rnd'), ('tri1', 'coh')):
        tag = scene.replace(':', '')
        n_max = 142 * LANES * R; rays = '%s/rays_%s.bin' % (D, tag)
        (T.make_rays(n_max) if kind == 'rnd' else np.repeat(np.array([[0, 0, -1, 0, 0, 0, 1, 1e16]], np.float32), n_max, axis=0)).tofile(rays)
        small = '%s/rays_small_%s.bin' % (D, tag); np.fromfile(rays, np.float32, count=71 * LANES * 2 * 8).tofile(small)
        go(D + '/s0_tiny', small, '%s/s0_tiny_%s.json' % (work, tag), scene, 71, 2, ['--dump-tris', D + '/t.bin'])
        go(D + '/s1_tiny', small, '%s/s1_tiny_%s.json' % (work, tag), scene, 71, 2, [])
        print(tag, 'tiny stage 0 and stage 1 ok', flush=True)
        for G in (71, 142):
            rf = '%s/rays_g%d_%s.bin' % (D, G, tag); np.fromfile(rays, np.float32, count=G * LANES * R * 8).tofile(rf)
        os.remove(rays)
        for rep in range(reps):
            for name, G, comp, ex in cfgs(group):
                n = G * LANES * R; rf = '%s/rays_g%d_%s.bin' % (D, G, tag); cx = (['--compute', comp] if comp else []) + ex
                if comp and rep == 0:
                    o = go(b[(0, G)], rf, '%s/s0_%s_%s.json' % (work, name, tag), scene, G, R, cx)
                    print('  stage 0', name, [l for l in o.splitlines() if l.startswith('compute')], flush=True)
                fin = D + '/fin.bin'; ref = D + '/ref.bin'
                go(b[(1, G)], rf, '%s/%s_%s_r%d.json' % (work, name, tag, rep), scene, G, R, cx + ['--optix-repeat', '1', '--dump-final', fin, '--dump-ref', ref])
                j = json.load(open('%s/%s_%s_r%d.json' % (work, name, tag, rep))); L = j['launches'][1:]
                ms = [l['ms'] for l in L]; rate = n / st.median(ms) / 1000
                a = np.fromfile(fin, np.uint32).reshape(-1, 14); r = np.fromfile(ref, np.uint32).reshape(-1, 8)
                hit = a[:, 0] != 0; rh = r[:, 0] != 0; both = hit & rh
                bad = int((hit != rh).sum()) + int((both & (np.abs(a[:, 4].view(np.float32)) != r[:, 1].view(np.float32))).sum()) + int((both & ((a[:, 3] & 0x1fffffff).astype(np.int64) != r[:, 2].astype(np.int64))).sum())
                unst = int((a[:, 13] == 0x77777777).sum())
                row = dict(scene=tag, config=name, rep=rep, G=G, rays=n, ms=ms, rays_per_us=rate, wrong=bad, unstored=unst, compute=comp, extra=ex)
                if comp:
                    rr = [l['citers'] / l['cwin_s'] for l in L if l.get('cwin_s')]; row['compute_with_rt'] = st.median(rr); row['compute_alone'] = j['compute_alone_iters'] / j['compute_alone_s']
                    row['blocks_even'] = j['compute_blocks_even']; row['blocks_odd'] = j['compute_blocks_odd']
                rows.append(row)
                print('  r%d %-26s G %3d RT %7.1f rays/us wrong %d unstored %d%s' % (rep, name, G, rate, bad, unst, '' if not comp else ' | compute %.2fx alone | blocks even %d odd %d' % (row['compute_with_rt'] / row['compute_alone'], row['blocks_even'], row['blocks_odd'])), flush=True)
                for f in (fin, ref):
                    if os.path.exists(f): os.remove(f)
                if bad or unst: sys.exit('WRONG RESULT in %s %s rep %d: stop' % (tag, name, rep))
                json.dump(rows, open(os.path.join(work, 'tpc5_%s_%s.json' % (BN, group)), 'w'), indent=1)
        for f in os.listdir(D):
            if f.startswith('rays_g'): os.remove(os.path.join(D, f))
    shutil.rmtree(D)


if __name__ == '__main__': main(*sys.argv[1:4])
