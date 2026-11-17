"""HG_1: one warp, our harness against OptiX, same rays, same in process GAS (predictions: model/predictions_hg1.md).

    run.py WORK STEP        STEP: s0 | s1first | all | cold

Binaries (built in /dev/shm/hg1, see README): hl_s0 / hl_s1 = build_patch --slim --remap --compact-out --regs 72 --max-out 61 --loop 32 --lclock
(clock64 before the first group and after the resume loop; group untouched, check_seq ALL PASS), setup3 host built with
--optix-clk support, setup3_dev.ptx compiled from hg1_dev.cu (raygen probes ta, tb, tc, td).

Safety (as every TTU run): stage 0 (no TTU) before stage 1, smallest launch first (1 block, 1 warp, 1 launch), harness run.py refuses an
unchecked binary or a busy GPU and runs under `timeout -k 2 30`; any bad exit, wrong result or missing block stops the script.
Raw files are never overwritten: every output has a timestamp in its name and is opened with mode x.
"""
import json, os, statistics as st, struct, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'config')); import gpucfg  # noqa: E402
CFG = gpucfg.load(); os.environ['TTU_GPU'] = str(CFG.index)
S3 = os.path.join(HERE, '..', '..', 'harness', 'setup3')
sys.path.insert(0, os.path.join(HERE, '..', 'Memory_4'))
D = os.environ.get('HG1_DIR', '/dev/shm/hg1b')
B0, B1 = D + '/hl_s0', D + '/hl_s1'
C0, C1 = D + '/clk_s0', D + '/clk_s1'   # plain --clock builds: the group only, one query
TS = time.strftime('%Y%m%dT%H%M%S')
K = 32
f32 = lambda u: struct.unpack('<f', struct.pack('<I', int(u)))[0]

# (name, --scene, ray argument)
SCENES = [('tri1', 'tri1', 'hit'), ('grid12', 'grid:12', 'tri:6'), ('grid144', 'grid:144', 'tri:72'), ('grid200', 'grid:200', 'tri:100'),
          ('grid2500', 'grid:2500', 'tri:1250'), ('grid4096', 'grid:4096', 'tri:2048'), ('grid20736', 'grid:20736', 'tri:10368'), ('soup', 'soup:100000', 'SOUP')]


def sh(c, **kw): return subprocess.run(c, capture_output=True, text=True, **kw)


def outname(work, name):
    p = os.path.join(work, '%s_%s' % (name, TS))
    for ext in ('.json', '.final.bin', '.ref.bin', '.tris.bin', '.log'):
        if os.path.exists(p + ext): sys.exit('refuse to overwrite ' + p + ext)
    return p


def go(binp, rays, base, scene, extra, loop=True):
    cmd = [sys.executable, os.path.join(S3, 'run.py'), binp, rays, 'zero', base + '.json', '--scene', scene, '--warps', '1', '--blocks', '1']
    cmd += (['--blockrays', '--slim', '--loop', '--lclock', '--dump-final', base + '.final.bin'] if loop else []) + ['--dump-ref', base + '.ref.bin', '--dump-tris', base + '.tris.bin'] + extra
    p = sh(cmd)
    open(base + '.log', 'x').write(p.stdout + p.stderr)
    ok = 'exit 0' in p.stdout and ' error' not in p.stdout.replace('no error', '') and 'ttu_one: no error' in p.stdout and '1 block(s) ran' in p.stdout
    print('   ', os.path.basename(base), 'OK' if ok else 'FAILED', flush=True)
    if not ok: sys.exit('STOP (fault, hang, wrong exit or missing block):\n' + p.stdout[-2500:] + p.stderr[-1500:])
    return json.load(open(base + '.json'))


def check_vs_optix(base):
    """Every lane of the harness final result vs the OptiX reference of the same ray (hit, primitive, |t|): (lanes, wrong, mean rounds, max rounds)."""
    a = np.fromfile(base + '.final.bin', dtype=np.uint32).reshape(-1, 14)
    r = np.fromfile(base + '.ref.bin', dtype=np.uint32).reshape(-1, 8)[:len(a)]
    bad = 0
    for i in range(len(a)):
        hit = a[i, 0] != 0
        if hit != bool(r[i, 0]): bad += 1; continue
        if hit and ((a[i, 3] & 0x1fffffff) != r[i, 2] or abs(abs(f32(a[i, 4])) - f32(r[i, 1])) > 1e-6 * max(1.0, f32(r[i, 1]))): bad += 1
    rounds = K - a[:, 13].astype(int)
    return len(a), bad, float(rounds.mean()), int(rounds.max()), bool((a[:, 13] > 0).all())


def spans(j):
    """Per launch (first dropped): harness warp span; OptiX TRACE = max tc - min tb, RAYGEN = max td - min ta over the 32 lanes."""
    h = [e - s for l in j['launches'][1:] for s, e in l['warp_clk']]
    ox = []
    stride = j.get('optix_clk_stride', 4)
    for run in j.get('optix_clk', []):
        c = np.array(run, dtype=np.uint64).reshape(-1, stride).astype(np.int64)
        if stride == 4:
            ox.append((int(c[:, 2].max() - c[:, 1].min()), int(c[:, 3].max() - c[:, 0].min()), int((c[:, 1] - c[:, 0]).max()), int((c[:, 3] - c[:, 2]).max())))
        else:   # ta, (tb_k, tc_k) x 3, td: (trace 0, raygen, pre, post, trace 1, trace 2); traces not run (nrep < 3) read 0
            n = j.get('optix_nrep', 1)
            tr = [int(c[:, 2 + 2 * k].max() - c[:, 1 + 2 * k].min()) if k < n else 0 for k in range(3)]
            ox.append((tr[0], int(c[:, 7].max() - c[:, 0].min()), int((c[:, 1] - c[:, 0]).max()), int((c[:, 7] - c[:, 2 + 2 * (n - 1)]).max()), tr[1], tr[2]))
    return h, ox


def scene_rays(work):
    sys.path.insert(0, os.path.join(HERE, '..', 'Memory_4'))
    from prep import make_rays
    f = os.path.join(work, 'rays_soup32.bin')
    if not os.path.exists(f): make_rays(32).tofile(f)
    return 'file:' + f


def t_s0(work):
    print(' stage 0 first (no TTU code), 1 block, 1 warp, 1 launch, one triangle')
    base = outname(work, 's0_tri1')
    go(B0, 'hit', base, 'tri1', ['--repeat', '1', '--optix-repeat', '2', '--optix-clk'])


def t_s1first(work):
    print(' stage 1 smallest: 1 block, 1 warp, 1 launch, one triangle')
    base = outname(work, 's1first_tri1')
    j = go(B1, 'hit', base, 'tri1', ['--repeat', '1', '--optix-repeat', '2', '--optix-clk'])
    print('    check vs OptiX:', check_vs_optix(base)); print('    spans:', spans(j))


def t_all(work, flush=0, tag='warm', only=None, nrep=1, rot=False):
    for name, scene, ray in SCENES:
        if only and name not in only: continue
        rays = scene_rays(work) if ray == 'SOUP' else ray
        extra = ['--optix-clk', '--optix-nrep', str(nrep)] + (['--optix-rot'] if rot else []) + (['--flush', str(flush)] if flush else [])
        b0 = outname(work, 's0_%s_%s' % (name, tag))
        go(B0, rays, b0, scene, extra + ['--repeat', '1', '--optix-repeat', '2'])
        base = outname(work, '%s_%s' % (name, tag))
        j = go(B1, rays, base, scene, extra + ['--repeat', '100', '--optix-repeat', '200'])
        n, bad, mr, xr, fin = check_vs_optix(base)
        if bad or not fin: sys.exit('STOP: %s: %d of %d lanes differ from OptiX (or unfinished %s)' % (name, bad, n, not fin))
        h, ox = spans(j)
        row = dict(scene=name, tag=tag, lanes=n, rounds_mean=mr, rounds_max=xr, harness=st.median(h), harness_n=len(h),
                   trace=st.median(o[0] for o in ox), raygen=st.median(o[1] for o in ox), pre=st.median(o[2] for o in ox), post=st.median(o[3] for o in ox))
        if nrep > 1: row.update(trace1=st.median(o[4] for o in ox), trace2=st.median(o[5] for o in ox), gap1=st.median(o[4] for o in ox) / row['harness'])
        row['gap'] = row['trace'] / row['harness']
        print('   ', {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
        open(os.path.join(work, 'row_%s_%s_%s.json' % (name, tag, TS)), 'x').write(json.dumps(row))


def t_clk(work):
    """Test B: the group only (existing --clock build), one query, same scenes; soup = the first query of the 32 random rays."""
    for name, scene, ray in SCENES:
        rays = scene_rays(work) if ray == 'SOUP' else ray
        b0 = outname(work, 'clk_s0_%s' % name)
        go(C0, rays, b0, scene, ['--repeat', '2'], loop=False)
        base = outname(work, 'clk_%s' % name)
        j = go(C1, rays, base, scene, ['--repeat', '100'], loop=False)
        h = [e - s for l in j['launches'][1:] for s, e in l['warp_clk']]
        row = dict(scene=name, tag='clk', group=st.median(h), n=len(h), minmax=(min(h), max(h)))
        print('   ', row, flush=True)
        open(os.path.join(work, 'row_clk_%s_%s.json' % (name, TS)), 'x').write(json.dumps(row))


if __name__ == '__main__':
    work, step = sys.argv[1], sys.argv[2]
    os.makedirs(work, exist_ok=True)
    if step == 's0': t_s0(work)
    elif step == 's1first': t_s1first(work)
    elif step == 'all': t_all(work)
    elif step == 'cold': t_all(work, 256, 'cold', only=['tri1'])
    elif step == 'nrep1': t_all(work, tag='nrep1')
    elif step == 'rep3': t_all(work, tag='rep3', nrep=3)
    elif step == 'rot3': t_all(work, tag='rot3', nrep=3, rot=True, only=['tri1', 'soup'])
    elif step == 'clk': t_clk(work)
    else: sys.exit('step?')
