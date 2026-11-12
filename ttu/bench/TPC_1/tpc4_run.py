"""TPC_1 queue 6: lean split. lean = L1 allocating loads + 4 stores; lean_ld = only the L1 allocating loads (14 stores); lean_st = only 4 stores (default loads). base and lean as controls. See tpc3_run.py.

    tpc4_run.py WORK REPS [smoke]
needs /dev/shm/t9/setup3 (setup3.cu), setup3_dev.ptx and words.txt there (same as tpc2_run.py).
RT builds (all the checked traversal group, byte identical group, check_seq ALL PASS on every build), G 71 (RT on the even SMs):
  base : 63 STRONG.SYS loads and 14 STRONG.SYS stores in the kernel, no L1 allocating load (the default build)
  hoist: constants loaded once (47 loads), same stores
  lean : 20 of the ray loads are L1 allocating (STRONG.SM), only 4 stores (hit, primitive, t, rounds left)
Sibling compute (odd SMs, 4 blocks of 384 threads): alu, smem (shared memory only), l1w 1 KB / 32 KB windows per block (L1 hits, 4 KB / 128 KB per SM), l1 (original, spills).
Scenes: tri1 coherent (the puzzle), soup:100000 random. Every run: all records of the last launch vs OptiX; any wrong or unstored record or bad exit stops the script.
Order: per scene, stage 0 and stage 1 tiny (no compute) for every build, then repetitions (outer loop) of every build x config; stage 0 with the compute first for each config in repetition 0.
"""
import json, os, statistics as st, sys, shutil, subprocess
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tpc1_run as T  # noqa: E402
from tpc1_run import go, D, W, LANES, K, S3, BASE, VENV, sh  # noqa: E402

BUILDS = {'base': ([], []), 'lean': (['--lean'], ['--lean']), 'lean_ld': (['--lean-ld'], []), 'lean_st': (['--lean-st'], ['--lean'])}   # name: (build_patch flags, host flags)


def build(stage, G, R, out, flags):
    cmd = [VENV, os.path.join(S3, 'build_patch.py'), '--stage', str(stage), '--bin', BASE, '--slim', '--remap', '--compact-out', '--regs', '72', '--max-out', '61',
           '--loop', str(K), '--outer', str(R), '--stride', str(G * LANES * 96), '--out', out] + flags
    p = sh(cmd)
    if p.returncode: sys.exit(p.stdout + p.stderr)
    c = sh([VENV, os.path.join(S3, 'check_seq.py'), out])
    if 'ALL PASS' not in c.stdout.splitlines()[-1]: sys.exit(c.stdout[-2000:])


def cfgs(smoke):
    c = [('alone71', None)]
    c += [('split_alu', 'alu:odd:4'), ('split_smem', 'smem:odd:4'), ('split_l1w_1KB', 'l1w:odd:4:1'), ('split_l1w_32KB', 'l1w:odd:4:32'), ('split_l1', 'l1:odd:4')]
    return c[:2] if smoke else c


def main(work, reps, smoke=None):
    os.makedirs(work, exist_ok=True); os.makedirs(D, exist_ok=True); R = 96; G = 71; reps = int(reps)
    shutil.copy('/dev/shm/t9/words.txt', D); shutil.copy('/dev/shm/t9/setup3_dev.ptx', D)
    b = {}
    for name, (fl, hx) in BUILDS.items():
        for stg in (0, 1): b[(name, stg)] = '%s/%s_s%d' % (D, name, stg); build(stg, G, R, b[(name, stg)], fl)
        build(1, G, 2, '%s/%s_s1_tiny' % (D, name), fl); build(0, G, 2, '%s/%s_s0_tiny' % (D, name), fl)
    rows = []
    for scene, kind in (('tri1', 'coh'), ('soup:100000', 'rnd')):
        tag = scene.replace(':', ''); n = G * LANES * R
        rf = '%s/rays_%s.bin' % (D, tag)
        (T.make_rays(n) if kind == 'rnd' else np.repeat(np.array([[0, 0, -1, 0, 0, 0, 1, 1e16]], np.float32), n, axis=0)).tofile(rf)
        small = '%s/rays_small_%s.bin' % (D, tag); np.fromfile(rf, np.float32, count=G * LANES * 2 * 8).tofile(small)
        for name, (fl, hx) in BUILDS.items():   # smallest TTU launch first: stage 0 then stage 1, tiny R, no compute
            go('%s/%s_s0_tiny' % (D, name), small, '%s/s0_tiny_%s_%s.json' % (work, name, tag), scene, G, 2, hx + (['--dump-tris', D + '/t.bin'] if name == 'base' else []))
            go('%s/%s_s1_tiny' % (D, name), small, '%s/s1_tiny_%s_%s.json' % (work, name, tag), scene, G, 2, hx)
            print(tag, name, 'tiny stage 0 and stage 1 ok', flush=True)
        for rep in range(reps):
            for name, (fl, hx) in BUILDS.items():
                for cname, comp in cfgs(smoke):
                    cx = hx + (['--compute', comp] if comp else [])
                    if comp and rep == 0:
                        o = go(b[(name, 0)], rf, '%s/s0_%s_%s_%s.json' % (work, name, cname, tag), scene, G, R, cx)
                        print('  stage 0', name, cname, [l for l in o.splitlines() if l.startswith('compute')], flush=True)
                    fin = D + '/fin.bin'; ref = D + '/ref.bin'
                    go(b[(name, 1)], rf, '%s/%s_%s_%s_r%d.json' % (work, name, cname, tag, rep), scene, G, R, cx + ['--optix-repeat', '1', '--dump-final', fin, '--dump-ref', ref])
                    j = json.load(open('%s/%s_%s_%s_r%d.json' % (work, name, cname, tag, rep))); L = j['launches'][1:]
                    ms = [l['ms'] for l in L]; rate = n / st.median(ms) / 1000
                    a = np.fromfile(fin, np.uint32).reshape(-1, 14); r = np.fromfile(ref, np.uint32).reshape(-1, 8)
                    hit = a[:, 0] != 0; rh = r[:, 0] != 0; both = hit & rh
                    bad = int((hit != rh).sum()) + int((both & (np.abs(a[:, 4].view(np.float32)) != r[:, 1].view(np.float32))).sum()) + int((both & ((a[:, 3] & 0x1fffffff).astype(np.int64) != r[:, 2].astype(np.int64))).sum())
                    unst = int((a[:, 13] == 0x77777777).sum())
                    row = dict(scene=tag, build=name, config=cname, rep=rep, G=G, rays=n, ms=ms, rays_per_us=rate, wrong=bad, unstored=unst, compute=comp)
                    if comp:
                        rr = [l['citers'] / l['cwin_s'] for l in L if l.get('cwin_s')]; row['compute_with_rt'] = st.median(rr); row['compute_alone'] = j['compute_alone_iters'] / j['compute_alone_s']
                        row['blocks_even'] = j['compute_blocks_even']; row['blocks_odd'] = j['compute_blocks_odd']
                    rows.append(row)
                    print('  %s r%d %-5s %-15s RT %7.1f rays/us wrong %d unstored %d%s' % (tag, rep, name, cname, rate, bad, unst, '' if not comp else ' | compute %.2fx alone | blocks even %d odd %d' % (row['compute_with_rt'] / row['compute_alone'], row['blocks_even'], row['blocks_odd'])), flush=True)
                    for f in (fin, ref):
                        if os.path.exists(f): os.remove(f)
                    json.dump(rows, open(os.path.join(work, 'tpc3.json'), 'w'), indent=1)
                    if bad or unst: sys.exit('WRONG RESULT in %s %s %s rep %d: stop' % (tag, name, cname, rep))
        os.remove(rf)
    shutil.rmtree(D)


if __name__ == '__main__': main(*sys.argv[1:4])
