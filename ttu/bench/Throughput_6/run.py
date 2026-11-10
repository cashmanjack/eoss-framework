"""Throughput_6: throughput vs number of active SMs, 32 warps per SM (saturated), coherent and incoherent.

    run.py BIN_S0 BIN_S1 WORK            BIN: --clock --sync --gtime --remap --compact-out --regs 64 --max-out 61, checked
    /usr/bin/python3 run.py --report WORK

G blocks of 32 warps, one block per SM (G <= 142 lands on G distinct SMs; block b on SM 2b for b < 71, then odd SMs).
Scenes: tri1 and par:64:s with one coherent ray in every lane; soup:100000 (6.2 MB GAS) and soup:3000000 (186 MB)
with a different random ray in every lane of every block (--blockrays). First query only (no resume). 6 launches,
the first dropped: BVH warm in L2 (the 186 MB one only partly).
Rate per SM = 1,024 rays / block span (SM clock cycles); whole GPU rate = G x 1,024 rays / (last block end - first block
start) on the global timer (1,024 ns steps), reported only when the blocks overlap.
"""
import json, os, statistics as st, struct, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.join(HERE, '..', '..', 'harness', 'setup3', 'run.py')
sys.path.insert(0, os.path.join(HERE, '..', 'Memory_4'))
from prep import make_rays  # noqa: E402

GS = [1, 2, 4, 8, 16, 24, 32, 48, 64, 71, 72, 80, 100, 120, 142]
SCENES = [('tri1', 'tri1', 'hit', False), ('par64', 'par:64:s', 't2', False), ('soup1e5', 'soup:100000', None, True), ('soup3e6', 'soup:3000000', None, True)]
W = 32


def go(binp, rays, out, scene, g, w, extra=()):
    p = subprocess.run([sys.executable, HARNESS, binp, rays, 'zero', out, '--scene', scene, '--warps', str(w), '--blocks', str(g), '--repeat', '6'] + list(extra),
                       capture_output=True, text=True)
    ok = 'exit 0' in p.stdout and ' error' not in p.stdout.replace('no error', '') and ('%d block(s) ran' % g) in p.stdout
    print(os.path.basename(out), 'OK' if ok else 'FAILED', flush=True)
    if not ok:
        sys.exit(p.stdout[-2500:] + p.stderr[-1500:])


def run(b0, b1, work):
    os.makedirs(work, exist_ok=True)
    big = make_rays(142 * W * 32)
    for name, scene, ray, inc in SCENES:
        for g in GS:
            rays = ray
            extra = []
            if inc:
                f = os.path.join(work, 'rays_%d.bin' % g); big[:g * W * 32].tofile(f); rays = 'file:' + f; extra = ['--blockrays']
            if name == 'tri1' and g == 1:   # smallest first: one warp
                go(b1, rays, os.path.join(work, '%s_g1_w1.json' % name), scene, 1, 1, extra)
            if g == GS[0] or g == GS[-1]:
                go(b0, rays, os.path.join(work, 's0_%s_g%d.json' % (name, g)), scene, g, W, extra)
            go(b1, rays, os.path.join(work, '%s_g%d_w%d.json' % (name, g, W)), scene, g, W, extra)


def blocks_of(j):
    out = []
    for l in j['launches'][1:]:
        out.append(l['per_block'])
    return out


def report(work):
    f = lambda u: struct.unpack('<f', struct.pack('<I', u))[0]
    print('aggregate = sum over blocks of 1,024 rays / block span (SM cycles); valid when all blocks overlap (last column)')
    print('%-8s | %3s | %12s %7s | %10s %9s | %8s | %s' % ('scene', 'G', 'rays/cyc/SM', 'vs G=1', 'aggregate', 'x 1 SM', 'overlap', 'checks'))
    for name, scene, ray, inc in SCENES:
        base = None
        for g in GS:
            p = os.path.join(work, '%s_g%d_w%d.json' % (name, g, W))
            if not os.path.exists(p):
                continue
            j = json.load(open(p)); bl = blocks_of(j)
            agg = []; ov = []
            for lb in bl:
                spans = [max(e for _, e in b['warp_clk']) - min(s for s, _ in b['warp_clk']) for b in lb]
                agg.append(sum(1024.0 / s for s in spans))
                gs = [min(s for s, _ in b['gt']) for b in lb]; ge = [max(e for _, e in b['gt']) for b in lb]
                ov.append(min(ge) > max(gs))
            a = st.median(agg); r = a / g
            if base is None: base = r
            ref = j['ref']; bad = 0; cnt = 0
            for k, wo in enumerate(j['warp_out']):
                lane = k * 32; hit = wo[3]; pend = any(wo[x] for x in (56, 57, 58, 59, 60, 61, 70, 71))
                if inc and pend: continue
                rr = ref[lane if inc else 0]; cnt += 1
                ok = (hit == 0 and rr['hit'] == 0) or (hit == 1 and rr['hit'] == 1 and (wo[44] & 0x1fffffff) == rr['prim'] and abs(-f(wo[45]) - f(int(rr['t'], 16))) <= 1e-6 * max(1, f(int(rr['t'], 16))))
                bad += not ok
            print('%-8s | %3d | %12.5f %7.2f | %10.4f %9.1f | %8s | %d lanes checked, %d wrong' % (name, g, r, r / base, a, a / base, '%d/%d' % (sum(ov), len(ov)), cnt, bad))


if __name__ == '__main__':
    report(sys.argv[2]) if sys.argv[1] == '--report' else run(*sys.argv[1:4])
