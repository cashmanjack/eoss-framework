"""Lane group runner (predictions: model/predictions_lane_groups.md). Checked remap build (t1r_s0 / t1r_s1 in /dev/shm/t9u, both check_seq ALL PASS), prepared by prepare_groups.py. TTU_GPU set.
    run.py WORK stage0 | first | runs
      stage0: the largest scene (1,000,000), 32 warps, grp16 rays, NO TTU instruction (stage 0 binary)
      first : the smallest TTU launch (soup:1000, grp2 32 rays, 1 warp, 1 launch)
      runs  : per size, per ray set (miss = g 1, grp2 ... grp32, coh): 1 warp x 30 launches and 32 warps x 30 launches; stops at the first failure
Raw JSON with a timestamp name, never overwritten (the summary is opened with mode x)."""
import json, os, statistics as st, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); S3 = os.path.join(HERE, '..', '..', 'harness', 'setup3'); D = '/dev/shm/t9u'; B1, B0 = D + '/t1r_s1', D + '/t1r_s0'
TS = time.strftime('%Y%m%dT%H%M%S'); SIZES = (1000, 10000, 100000, 1000000); SETS = ('miss', 'grp2', 'grp4', 'grp8', 'grp16', 'grp32', 'coh')   # design A (counts not fixed); 'runsB' uses SETS_B
SETS_B = ('cls1', 'cls2', 'cls4', 'cls8', 'cls16', 'cls32')            # design B: 32 rays of one (nodes, slots, triangles) class (prepare_class.py)


def go(binp, out, scene, ray, extra):
    if os.path.exists(out): sys.exit('exists ' + out)
    p = subprocess.run([sys.executable, os.path.join(S3, 'run.py'), binp, ray, 'zero', out, '--scene', scene] + extra, capture_output=True, text=True)
    ok = 'exit 0' in p.stdout and 'ttu_one: no error' in p.stdout and '1 block(s) ran' in p.stdout
    print('   ', os.path.basename(out), 'OK' if ok else 'FAILED', flush=True)
    if not ok: sys.exit('STOP:\n' + p.stdout[-2500:] + p.stderr[-1500:])
    return json.load(open(out)) if os.path.exists(out) else None


def stats(j):
    L = j['launches'][1:] if len(j['launches']) > 1 else j['launches']; w = j['warps']
    span = [max(e for _, e in l['warp_clk']) - min(s for s, _ in l['warp_clk']) for l in L]
    return dict(w=w, c=st.median(span) / (32 * w), lat=st.median([e - s for l in L for s, e in l['warp_clk']]), span=st.median(span), launches=len(L))


def rayfile(name, n, w):
    if name == 'miss': return '%s/miss%s%d.bin' % (D, '32_' if w == 1 else '', n)
    if name.startswith('cls') and w == 12: return '%s/%s_w12_%d.bin' % (D, name, n)   # 384 rays: the harness needs a multiple of 32 W
    if name.startswith('cls'): return '%s/%s%s_%d.bin' % (D, name, '_32' if w == 1 else '', n)
    if name == 'coh': return '%s/coh%s%d.bin' % (D, '32_' if w == 1 else '', n)
    return '%s/%s%s_%d.bin' % (D, name, '_32' if w == 1 else '', n)


if __name__ == '__main__':
    work, step = sys.argv[1], sys.argv[2]; os.makedirs(work, exist_ok=True)
    if step == 'stage0':
        go(B0, '%s/g_stage0_%s.json' % (work, TS), 'soup:1000000', 'file:' + rayfile('grp16', 1000000, 32), ['--warps', '32', '--repeat', '1'])
    elif step == 'first':
        go(B1, '%s/g_first_%s.json' % (work, TS), 'soup:1000', 'file:' + rayfile('grp2', 1000, 1), ['--warps', '1', '--repeat', '1'])
    elif step in ('stage0S', 'firstS'):   # shifted start files (predictions: model/predictions_knee2.md); the start file has 1,024 lines (lane pattern tiled)
        n, w, bn = (1000000, 32, B0) if step == 'stage0S' else (1000, 1, B1)
        go(bn, '%s/s_%s_%s.json' % (work, step, TS), 'soup:%d' % n, 'file:' + rayfile('cls4' if n == 1000000 else 'cls2', n, w), ['--warps', str(w), '--repeat', '1', '--start-file', '%s/sh%d_%d.txt' % (D, 4 if n == 1000000 else 2, n)])
    elif step == 'runsN':   # Phase 10i test 2: far (cls g = 32 / D), near (near<D>) and the identical floor, D = 2 4 8 16 32, sizes 100,000 and 1,000,000, 32 warps, 3 rotated repetitions (predictions: model/predictions_phase10i.md)
        res = []; cells = [(n, nm) for n in (100000, 1000000) for nm in ['cls32'] + ['cls%d' % (32 // d) for d in (2, 4, 8, 16, 32)] + ['near%d' % d for d in (2, 4, 8, 16, 32)] if os.path.exists(rayfile(nm, n, 32))]
        for rep in range(3):
            for k in range(len(cells)):
                n, nm = cells[(k + rep * 4) % len(cells)]
                j = go(B1, '%s/n_%s_%d_r%d_%s.json' % (work, nm, n, rep, TS), 'soup:%d' % n, 'file:' + rayfile(nm, n, 32), ['--warps', '32', '--repeat', '30'])
                s = stats(j); s.update(size=n, rays=nm, rep=rep); res.append(s); print('   ', s, flush=True)
        json.dump(res, open('%s/n_summary_%s.json' % (work, TS), 'x'))
    elif step == 'runsK2':   # W = 12 on six sizes, new sizes at W = 16 and 32, shifted variant (120 runs)
        res = []; S6 = (1000, 10000, 30000, 100000, 300000, 1000000)
        def one(kind, n, W, nm, extra=()):
            tag = '%s_%s_%d_w%d' % (kind, nm, n, W)
            j = go(B1, '%s/k2_%s_%s.json' % (work, tag, TS), 'soup:%d' % n, 'file:' + rayfile(nm, n, W), ['--warps', str(W), '--repeat', '30'] + list(extra))
            s = stats(j); s.update(size=n, rays=nm, kind=kind); res.append(s); print('   ', s, flush=True)
        for n in S6:
            for nm in SETS_B: one('plain', n, 12, nm)
        for n in (30000, 300000):
            for W in (16, 32):
                for nm in SETS_B: one('plain', n, W, nm)
        for n in S6:
            for W in (16, 32):
                for g in (2, 4, 8, 16, 32): one('shift', n, W, 'cls%d' % g, ['--start-file', '%s/sh%d_%d.txt' % (D, g, n)])
        json.dump(res, open('%s/k2_summary_%s.json' % (work, TS), 'x'))
    elif step == 'runsK':   # knee map: design B rays, W = 4 8 16 32 warps, 4 sizes x 6 g x 4 W = 96 runs (predictions: model/predictions_knee.md)
        res = []
        for n in SIZES:
            for W in (4, 8, 16, 32):
                for nm in SETS_B:
                    j = go(B1, '%s/k_%s_%d_w%d_%s.json' % (work, nm, n, W, TS), 'soup:%d' % n, 'file:' + rayfile(nm, n, 32), ['--warps', str(W), '--repeat', '30'])
                    s = stats(j); s.update(size=n, rays=nm); res.append(s); print('   ', s, flush=True)
        json.dump(res, open('%s/k_summary_%s.json' % (work, TS), 'x'))
    elif step in ('runs', 'runsB'):
        res = []
        for n in SIZES:
            for nm in (SETS if step == 'runs' else SETS_B):
                for w in (1, 32):
                    j = go(B1, '%s/g_%s_%d_w%d_%s.json' % (work, nm, n, w, TS + ('B' if step == 'runsB' else '')), 'soup:%d' % n, 'file:' + rayfile(nm, n, w), ['--warps', str(w), '--repeat', '30'])
                    s = stats(j); s.update(size=n, rays=nm); res.append(s); print('   ', s, flush=True)
        json.dump(res, open('%s/g_summary%s_%s.json' % (work, 'B' if step == 'runsB' else '', TS), 'x'))
