"""Unit_3 runner (predictions: model/predictions_unit3.md). Checked remap build (Throughput_1 / Unit_1), soup:N scenes, rays from files (no hit rays, incoherent and coherent).

    run.py WORK first|runs|ncu      DATA = /dev/shm/t9u (t1r_s0/t1r_s1, miss<N>.bin, coh<N>.bin, miss32_<N>.bin, coh32_<N>.bin made by the prepare step); TTU_GPU set
      first : the smallest TTU launch (soup:1000, coherent 32 rays, 1 warp, 1 launch)
      runs  : per size, per ray set (incoherent, coherent): 1 warp x 30 launches and 32 warps x 30 launches; stops at the first failure
      ncu   : per size, incoherent and coherent, 32 warps, 2 launches, ncu l1tex__m_xbar2l1tex_read_sectors.sum on the second launch
Raw JSON per run with a timestamp name, never overwritten."""
import json, os, statistics as st, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); S3 = os.path.join(HERE, '..', '..', 'harness', 'setup3'); D = '/dev/shm/t9u'; B1 = D + '/t1r_s1'
TS = time.strftime('%Y%m%dT%H%M%S'); SIZES = (1000, 10000, 100000, 1000000)


def go(out, scene, ray, extra, blocks=1):
    if os.path.exists(out): sys.exit('exists ' + out)
    p = subprocess.run([sys.executable, os.path.join(S3, 'run.py'), B1, ray, 'zero', out, '--scene', scene] + extra, capture_output=True, text=True)
    ok = 'exit 0' in p.stdout and 'ttu_one: no error' in p.stdout and ('%d block(s) ran' % blocks) in p.stdout
    print('   ', os.path.basename(out), 'OK' if ok else 'FAILED', flush=True)
    if not ok: sys.exit('STOP:\n' + p.stdout[-2500:] + p.stderr[-1500:])
    return json.load(open(out)) if os.path.exists(out) else None


def stats(j):
    L = j['launches'][1:] if len(j['launches']) > 1 else j['launches']; w = j['warps']
    span = [max(e for _, e in l['warp_clk']) - min(s for s, _ in l['warp_clk']) for l in L]
    return dict(w=w, c=st.median(span) / (32 * w), lat=st.median([e - s for l in L for s, e in l['warp_clk']]), span=st.median(span), launches=len(L))


if __name__ == '__main__':
    work, step = sys.argv[1], sys.argv[2]; os.makedirs(work, exist_ok=True)
    if step == 'first':
        go('%s/u3_first_%s.json' % (work, TS), 'soup:1000', 'file:%s/coh32_1000.bin' % D, ['--warps', '1', '--repeat', '1'])
    elif step == 'runs':
        res = []
        for n in SIZES:
            for nm in ('miss', 'coh'):
                for w in (1, 32):
                    f = '%s/%s%s%d.bin' % (D, nm, '32_' if w == 1 else '', n)
                    j = go('%s/u3_%s_%d_w%d_%s.json' % (work, nm, n, w, TS), 'soup:%d' % n, 'file:' + f, ['--warps', str(w), '--repeat', '30'])
                    s = stats(j); s.update(size=n, rays=nm); res.append(s); print('   ', s, flush=True)
        json.dump(res, open('%s/u3_summary_%s.json' % (work, TS), 'x'))
    elif step == 'ncu':
        for n in SIZES:
            for nm in ('miss', 'coh'):
                out = '%s/u3_ncu_%s_%d_%s.json' % (work, nm, n, TS); csv = out.replace('.json', '.csv')
                go(out, 'soup:%d' % n, 'file:%s/%s%d.bin' % (D, nm, n), ['--warps', '32', '--repeat', '2', '--ncu', 'l1tex__m_xbar2l1tex_read_sectors.sum', '--ncu-out', csv, '--ncu-skip', '1'] if False else ['--warps', '32', '--repeat', '2', '--ncu', 'l1tex__m_xbar2l1tex_read_sectors.sum', '--ncu-out', csv])
