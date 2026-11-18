"""Conc_3 (Phase 10d): page test for F238 and clock matched memory stream arms, on top of Conc_1 (same checked lean TTU group, stage 0 and tiny launches first, same per record checks).
    conc3_run.py WORK GROUP REPS      GROUP: smoke | pg | mem | odd
pg : the page test (model/predictions_phase10d.md section 1): streams with the same line hash and different page spread (pgfew, pgmany, pgfar, dramfew, seqpg) and the F229 reference stream (memcg), on sib (4 blocks per SM) and hi (2 per SM).
odd: control for pg (predictions 1b): pgodd (stride 241 lines, not a power of two) with pgmany again in the same group.
mem: the F229 memory stream arms rerun (mem and memcg 1, 2, 4 blocks; sleep 150 and 800 us; memseq 1, 4 / 1, 2).
An alone run precedes EVERY arm (alone baselines at matched clocks). The sampler (clksamp, ring buffer, launch windows only) gets the window of every launch from the raw json.
Raw: WORK/conc_<group>_<TS>.json (rows), WORK/clk_<group>_<TS>.csv (launch window rows, gzipped if over 100 MB); never overwritten."""
import gzip, json, os, shutil, statistics as st, subprocess, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..', 'Conc_1'))
import conc_run as C  # noqa: E402
Q, HALF, LANES, go, D = C.Q, C.HALF, C.LANES, C.go, C.D
HX = C.HX


def arms(group):
    a = []
    def add(t, pl, lim, d=0):
        nm = '%s_%s%d' % ('hi' if pl == 'high' else pl, t, lim) + ('_d%d' % (d // 1000) if d else '')
        a.append((nm, Q, '%s:%s:%d:%d' % (t, pl, lim, d) if d else '%s:%s:%d' % (t, pl, lim)))
    if group == 'odd':
        for pl, lim in (('sib', 4), ('high', 2)):
            add('pgodd', pl, lim); add('pgodd', pl, lim, 150000)
            add('pgmany', pl, lim)
    elif group in ('pg', 'smoke'):
        for pl, lim in (('sib', 4), ('high', 2)):
            for t in ('memcg', 'pgfew', 'pgmany', 'pgfar', 'dramfew', 'seqpg'): add(t, pl, lim)
            for t in ('pgfew', 'pgmany'): add(t, pl, lim, 150000)
        if group == 'smoke': return [a[1]]
    else:
        for t in ('mem', 'memcg'):
            for pl in ('sib', 'high'):
                for lim in (1, 2, 4): add(t, pl, lim)
            for pl, lim in (('sib', 4), ('high', 2)):
                for d in (150000, 800000): add(t, pl, lim, d)
        for pl, lim in (('sib', 1), ('sib', 4), ('high', 1), ('high', 2)): add('memseq', pl, lim)
    return a


def main(work, group, reps):
    reps = int(reps); os.makedirs(work, exist_ok=True); os.makedirs(D, exist_ok=True); R = int(os.environ.get('CONC_R', 96)); ts = time.strftime('%Y%m%dT%H%M%S')
    out_json = os.path.join(work, 'conc_%s_%s.json' % (group, ts)); csvf = os.path.join(work, 'clk_%s_%s.csv' % (group, ts))
    if os.path.exists(out_json) or os.path.exists(csvf): sys.exit('exists')
    gc = C.gpucfg.load()
    shutil.copy('/dev/shm/t9/words.txt', D); shutil.copy('/dev/shm/t9/setup3_dev.ptx', D)
    b = {}
    for stg in (0, 1): b[(stg, Q)] = '%s/s%d_g%d' % (D, stg, Q); C.build(stg, Q, R, b[(stg, Q)])
    C.build(1, Q, 2, D + '/s1_tiny'); C.build(0, Q, 2, D + '/s0_tiny')
    sp = subprocess.Popen([os.path.join(HERE, 'clksamp'), str(gc.index), csvf], stdin=subprocess.PIPE, text=True); time.sleep(1)
    if sp.poll() is not None: sys.exit('sampler failed %s' % sp.returncode)
    rows = []; seq = 0
    def window(rj, key):
        j = json.load(open(rj))
        for i, l in enumerate(j['launches']):
            if 't0' in l: sp.stdin.write('%.6f %.6f %s %d\n' % (l['t0'], l['t1'], key, i))
        sp.stdin.flush()
    try:
        for scene, kind in (('soup:100000', 'rnd'), ('tri1', 'coh')):
            tag = scene.replace(':', ''); n_max = Q * LANES * R; rays = '%s/rays_%s.bin' % (D, tag)
            (C.T.make_rays(n_max) if kind == 'rnd' else np.repeat(np.array([[0, 0, -1, 0, 0, 0, 1, 1e16]], np.float32), n_max, axis=0)).tofile(rays)
            small = '%s/rays_small_%s.bin' % (D, tag); np.fromfile(rays, np.float32, count=Q * LANES * 2 * 8).tofile(small)
            go(D + '/s0_tiny', small, '%s/s0_tiny_%s_%s.json' % (work, tag, time.strftime('%Y%m%dT%H%M%S')), scene, Q, 2, ['--dump-tris', D + '/t.bin', '--lean'])
            go(D + '/s1_tiny', small, '%s/s1_tiny_%s_%s.json' % (work, tag, time.strftime('%Y%m%dT%H%M%S')), scene, Q, 2, ['--lean'])
            print(tag, 'tiny stage 0 and stage 1 ok', flush=True)
            rf = '%s/rays_g%d_%s.bin' % (D, Q, tag); np.fromfile(rays, np.float32, count=Q * LANES * R * 8).tofile(rf); os.remove(rays)
            seen0 = set()
            for rep in range(reps):
                for name, G, comp in [x for arm in arms(group) for x in (('alone%d' % Q, Q, None), arm)]:
                    n = G * LANES * R; cx = ['--compute', comp] if comp else []
                    if comp and name not in seen0:
                        seen0.add(name)
                        o = go(b[(0, G)], rf, '%s/s0_%s_%s_%s.json' % (work, name, tag, time.strftime('%Y%m%dT%H%M%S')), scene, G, R, cx + HX)
                        print('  stage 0', name, [l for l in o.splitlines() if l.startswith('compute')], flush=True)
                    fin = D + '/fin.bin'; ref = D + '/ref.bin'; rj = '%s/%s_%s_r%d_%s.json' % (work, name, tag, rep, time.strftime('%Y%m%dT%H%M%S'))
                    go(b[(1, G)], rf, rj, scene, G, R, cx + HX + ['--optix-repeat', '1', '--dump-final', fin, '--dump-ref', ref])
                    j = json.load(open(rj)); L = j['launches'][1:]; ms = [l['ms'] for l in L]; rate = n / st.median(ms) / 1000
                    a = np.fromfile(fin, np.uint32).reshape(-1, 14); r = np.fromfile(ref, np.uint32).reshape(-1, 8)
                    hit = a[:, 0] != 0; rh = r[:, 0] != 0; both = hit & rh
                    bad = int((hit != rh).sum()) + int((both & (np.abs(a[:, 4].view(np.float32)) != r[:, 1].view(np.float32))).sum()) + int((both & ((a[:, 3] & 0x1fffffff).astype(np.int64) != r[:, 2].astype(np.int64))).sum())
                    unst = int((a[:, 13] == 0x77777777).sum()); key = '%s_%s_r%d_s%d' % (name, tag, rep, seq)
                    row = dict(scene=tag, config=name, rep=rep, seq=seq, key=key, G=G, rays=n, ms=ms, rays_per_us=rate, wrong=bad, unstored=unst, compute=comp, raw=os.path.basename(rj))
                    if comp:
                        rr = [l['citers'] / l['cwin_s'] for l in L if l.get('cwin_s')]; row['compute_with_rt'] = st.median(rr); row['compute_alone'] = j['compute_alone_iters'] / j['compute_alone_s']
                    window(rj, key); seq += 1; rows.append(row)
                    print('  r%d %-10s %-18s RT %7.1f rays/us wrong %d unstored %d %s' % (rep, tag, name, rate, bad, unst, '' if not comp else '| compute %.3fx alone' % (row['compute_with_rt'] / row['compute_alone'])), flush=True)
                    for f in (fin, ref):
                        if os.path.exists(f): os.remove(f)
                    with open(out_json + '.tmp', 'w') as f: json.dump(rows, f, indent=1)
                    if bad or unst: sys.exit('STOP: wrong result or unstored record in %s %s rep %d' % (tag, name, rep))
            os.remove(rf)
    finally:
        try: sp.stdin.close()
        except Exception: pass
        sp.wait()
        if os.path.exists(csvf) and os.path.getsize(csvf) > 100e6:   # raw file over 100 MB: gzip, verify, then remove the plain copy
            with open(csvf, 'rb') as fi, gzip.open(csvf + '.gz', 'xb') as fo: shutil.copyfileobj(fi, fo)
            if subprocess.run('zcat %s.gz | cmp - %s' % (csvf, csvf), shell=True).returncode == 0: os.remove(csvf)
    os.rename(out_json + '.tmp', out_json); shutil.rmtree(D)


if __name__ == '__main__': main(*sys.argv[1:4])
