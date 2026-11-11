"""Throughput_9: RT capacity of one SM (lanes in flight) with masked lanes, mixed warps, and both SMs of one TPC.

    run.py STAGE0 STAGE1 WORK [part]     binaries built with --clock --sync --smfilter --remap --compact-out --regs 64 --max-out 61 (checked)
    /usr/bin/python3 run.py --report WORK

All warps start together after a barrier (--sync); one query per lane (first query only), 31 launches, the first dropped.
Idle lanes exit before TTUOPEN (--idle mask). Scenes: tri1 (one coherent hit ray: cheap query) and soup:100000 (random ray per lane).
Parts: A  one SM (--sms 0,0): L in 1, 8, 16, 32 active lanes x W warps 1..32.
       B  mixed warps: per warp lane list 32,1 (alternating) and 32,1,1,1, W = 2..32.
       C  both SMs of TPC 0 (--sms 0,1, 142 blocks, the other blocks exit): L 1, 8, 32 x W.
Stage 0 (no TTU) runs first (smallest config), then stage 1 smallest first (1 warp, 1 lane).
"""
import json, os, statistics as st, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.join(HERE, '..', '..', 'harness', 'setup3', 'run.py')
sys.path.insert(0, os.path.join(HERE, '..', 'Memory_4'))
from prep import make_rays  # noqa: E402

WS = [1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32]
LS = [1, 8, 16, 32]
SCENES = [('tri1', 'tri1'), ('soup', 'soup:100000')]


def go(binp, work, name, scene, W, extra):
    out = os.path.join(work, name + '.json')
    if os.path.exists(out): return
    rays = 'hit' if scene == 'tri1' else 'file:' + os.path.join(work, 'rays_w%d.bin' % W)
    p = subprocess.run([sys.executable, HARNESS, binp, rays, 'zero', out, '--scene', scene, '--warps', str(W), '--repeat', '31'] + extra, capture_output=True, text=True)
    ok = p.returncode == 0 and 'exit 0' in p.stdout and ' error' not in p.stdout.replace('no error', '')
    print(name, 'OK' if ok else 'FAILED', flush=True)
    if not ok: sys.exit(p.stdout[-2500:] + p.stderr[-1500:])


def run(b0, b1, work, part='ABC'):
    os.makedirs(work, exist_ok=True)
    for w in WS: make_rays(32 * w).tofile(os.path.join(work, 'rays_w%d.bin' % w))
    go(b0, work, 's0_tri1_w2_L8', 'tri1', 2, ['--sms', '0,0', '--active', '8'])      # stage 0 first: no TTU code
    go(b1, work, 'A_tri1_w1_L1', 'tri1', 1, ['--sms', '0,0', '--active', '1'])       # smallest TTU launch first
    if 'A' in part:
        for sc, scene in SCENES:
            for L in LS:
                for W in WS: go(b1, work, 'A_%s_w%d_L%d' % (sc, W, L), scene, W, ['--sms', '0,0', '--active', str(L)])
    if 'B' in part:
        for sc, scene in SCENES:
            for pat in ('32,1', '32,1,1,1'):
                for W in WS[1:]: go(b1, work, 'B_%s_w%d_p%s' % (sc, W, pat.replace(',', '-')), scene, W, ['--sms', '0,0', '--wactive', pat])
    if 'C' in part:
        for sc, scene in SCENES:
            for L in (1, 8, 32):
                for W in WS: go(b1, work, 'C_%s_w%d_L%d' % (sc, W, L), scene, W, ['--sms', '0,1', '--blocks', '142', '--active', str(L)])


def stats(path, lanes):
    """lanes(w) = active lanes of warp w. Returns per warp median latency, throughput per SM (lanes/cycle), number of blocks."""
    j = json.load(open(path)); lat = {}; thr = []
    for l in j['launches'][1:]:
        for b in l['per_block']:
            wc = b['warp_clk']
            for w, (s, e) in enumerate(wc): lat.setdefault((b['sm'], w), []).append(e - s)
            span = max(e for _, e in wc) - min(s for s, _ in wc)
            thr.append((b['sm'], sum(lanes(w) for w in range(len(wc))) / span))
    return lat, thr


def check(path, lanes):
    """Lane 0 of every warp (always active) against the OptiX reference (hit, prim, t; rays that return a pending entry are skipped), masked lanes of warp 0 unwritten, results equal in all launches."""
    import struct
    f = lambda u: struct.unpack('<f', struct.pack('<I', u))[0]
    j = json.load(open(path)); bad = cnt = 0
    for k, wo in enumerate(j['warp_out']):
        if any(wo[x] for x in (56, 57, 58, 59, 60, 61, 70, 71)): continue
        rr = j['ref'][(k % j['warps']) * 32 if j['scene'] != 'tri1' else 0]; hit = wo[3]; cnt += 1
        ok = (hit == 0 and rr['hit'] == 0) or (hit == 1 and rr['hit'] == 1 and (wo[44] & 0x1fffffff) == rr['prim'] and abs(-f(wo[45]) - f(int(rr['t'], 16))) <= 1e-6 * max(1, f(int(rr['t'], 16))))
        bad += not ok
    la = lanes(0); bad += sum(1 for l in range(la, 32) if j['out'][l][12] != 0x77777777)
    bad += sum(1 for l in j['launches'] if not l['same_as_first'])
    return cnt, bad


def peak(path, L):
    """Most warps in flight at one moment (start <= t < end of the TTU group per warp, one block), median over launches 1..30; lanes = warps x L."""
    j = json.load(open(path)); pk = []
    for l in j['launches'][1:]:
        b = l['per_block'][0]['warp_clk']; ev = sorted([(s, 1) for s, _ in b] + [(e, -1) for _, e in b], key=lambda x: (x[0], x[1])); c = m = 0
        for _, d in ev: c += d; m = max(m, c)
        pk.append(m)
    return st.median(pk)


def report(work):
    for sc, _ in SCENES:
        print('\nmost warps in flight at one moment (Part A, one SM): ' + sc)
        for L in LS: print('  L=%2d  ' % L + ' '.join('W%d:%d' % (W, peak(os.path.join(work, 'A_%s_w%d_L%d.json' % (sc, W, L)), L)) for W in WS if os.path.exists(os.path.join(work, 'A_%s_w%d_L%d.json' % (sc, W, L)))))
    nbad = 0; nchk = 0
    for fn in sorted(os.listdir(work)):
        if not fn.endswith('.json') or fn.startswith('s0_'): continue
        L = int(fn.split('_L')[1].split('.')[0]) if '_L' in fn else None
        pat = [int(x) for x in fn.split('_p')[1].split('.')[0].split('-')] if '_p' in fn else None
        c, b = check(os.path.join(work, fn), (lambda w: pat[w % len(pat)]) if pat else (lambda w: L)); nchk += c; nbad += b
        if b: print('BAD', fn, b)
    print('correctness: %d warp results checked against OptiX, %d problems' % (nchk, nbad))
    ms = lambda v: st.median(v)
    for sc, _ in SCENES:
        print('\n== Part A (%s): one SM, all warps start together. latency = median over warps (cycles); lanes/cyc = active lanes / block span; Little = lanes/cyc x unloaded latency (W 1)' % sc)
        for L in LS:
            rows = []; base = None
            for W in WS:
                p = os.path.join(work, 'A_%s_w%d_L%d.json' % (sc, W, L))
                if not os.path.exists(p): continue
                lat, thr = stats(p, lambda w: L)
                lt = ms([ms(v) for v in lat.values()]); rate = ms([t for _, t in thr])
                if base is None: base = lt
                rows.append((W, W * L, lt, lt / base, rate, rate * base))
            print(' L=%d' % L); print('   W   lanes   latency  x unloaded  lanes/cyc  Little(lanes)')
            for r in rows: print('  %2d  %5d  %8.0f  %9.2f  %9.4f  %10.1f' % r)
    for sc, _ in SCENES:
        print('\n== Part B (%s): mixed warps, one SM. Per warp class: latency (cycles) vs the same warp class run alone in Part A at the same W' % sc)
        for pat in ('32,1', '32,1,1,1'):
            lst = [int(x) for x in pat.split(',')]; print(' pattern', pat)
            print('   W  full-warp lat  1-lane lat   (A: all-full lat   A: all-1-lane lat)  lanes/cyc')
            for W in WS[1:]:
                p = os.path.join(work, 'B_%s_w%d_p%s.json' % (sc, W, pat.replace(',', '-')))
                if not os.path.exists(p): continue
                lat, thr = stats(p, lambda w: lst[w % len(lst)])
                full = [ms(v) for (s, w), v in lat.items() if lst[w % len(lst)] == 32]; one = [ms(v) for (s, w), v in lat.items() if lst[w % len(lst)] == 1]
                a32 = os.path.join(work, 'A_%s_w%d_L32.json' % (sc, W)); a1 = os.path.join(work, 'A_%s_w%d_L1.json' % (sc, W))
                f = lambda q: ms([ms(v) for v in stats(q, lambda w: 1)[0].values()]) if os.path.exists(q) else float('nan')
                print('  %2d  %12.0f  %10.0f   (%13.0f  %15.0f)  %9.4f' % (W, ms(full) if full else float('nan'), ms(one) if one else float('nan'), f(a32), f(a1), ms([t for _, t in thr])))
    for sc, _ in SCENES:
        print('\n== Part C (%s): SMs 0 and 1 (one TPC) both active vs SM 0 alone (Part A). per SM lanes/cyc, latency' % sc)
        for L in (1, 8, 32):
            print(' L=%d\n   W  lanes/SM  lat(SM0 alone)  lat(pair)  lanes/cyc SM0 alone  SM0 pair  SM1 pair  pair total' % L)
            for W in WS:
                p = os.path.join(work, 'C_%s_w%d_L%d.json' % (sc, W, L)); a = os.path.join(work, 'A_%s_w%d_L%d.json' % (sc, W, L))
                if not (os.path.exists(p) and os.path.exists(a)): continue
                lat, thr = stats(p, lambda w: L); la, ta = stats(a, lambda w: L)
                r = {s: ms([t for ss, t in thr if ss == s]) for s in (0, 1)}
                print('  %2d  %6d  %12.0f  %9.0f  %12.4f  %12.4f  %8.4f  %9.4f' % (W, W * L, ms([ms(v) for v in la.values()]), ms([ms(v) for v in lat.values()]), ms([t for _, t in ta]), r[0], r[1], r[0] + r[1]))


if __name__ == '__main__':
    report(sys.argv[2]) if sys.argv[1] == '--report' else run(*sys.argv[1:5])
