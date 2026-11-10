"""Memory_4, large working set: 65,536 rays on soup:3000000 (186 MB GAS), so that what one pass
touches no longer fits in the 96 MB L2 (the 8,192 ray set touched about 31 MB).

    bigset.py BIN_S0 BIN_S1 WORK

Steps (all GPU runs through ttu/harness/setup3/run.py):
 1. prep.py (OptiX only) -> rays, box keys.
 2. root start run of the interleaved order, recording every ray's leaf block and the GAS.
 3. exact keys: candidates per hit = the node at or below its leaf block, other nodes sharing that
    leaf base, then the previous node; hardware checks (each ray started at its key node) move only
    the wrong rays to their next candidate, until no ray is wrong.
 4. grouped order from the exact keys; start files.
 5. timing: interleaved, grouped, b_grouped_warp, b_grouped_ray at W 4 and 16, warm and cold.
 6. ncu at W 16: interleaved, grouped, b_grouped_warp; warm profiles only pass 1 (--ncu-skip).
"""
import bisect, json, os, struct, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.append(os.path.join(HERE, '..', 'BVH_5'))
import gas  # noqa: E402
from analyze import analyze, chunk_files, launches_per_chunk  # noqa: E402
from run import GA, GB  # noqa: E402

HARNESS = os.path.join(HERE, '..', '..', 'harness', 'setup3', 'run.py')
SCENE = 'soup:3000000'; N = 65536


def go(binp, rays, out, w, extra):
    r = subprocess.run([sys.executable, HARNESS, binp, 'file:' + rays, 'zero', out, '--scene', SCENE, '--warps', str(w)] + extra,
                       capture_output=True, text=True)
    ok = r.returncode == 0 and 'exit 0' in r.stdout and ' error' not in r.stdout.replace('no error', '')
    print(os.path.basename(out), 'OK' if ok else 'FAILED', flush=True)
    if not ok:
        sys.exit(r.stdout[-3000:] + r.stderr[-3000:])


def lanes(path, key):
    j = json.load(open(path)); out = []
    for f in chunk_files(path, j, 0):
        out += json.load(open(f))[key]
    return out


def write_starts(d, m, keys):
    fmt = lambda k: ','.join(map(str, k)) if k else 'root'
    for o in ('grouped', 'interleaved'):
        ks = [keys[i] for i in m['orders'][o]]; warp = []
        for w in range(0, len(ks), 32):
            g = ks[w:w + 32]; p = ()
            while len(p) < min(map(len, g)) and all(k[len(p)] == g[0][len(p)] for k in g):
                p = g[0][:len(p) + 1]
            warp += [p] * len(g)
        for mode, s in (('warp', warp), ('ray', ks)):
            open(os.path.join(d, 'starts_%s_%s.txt' % (o, mode)), 'w').write(''.join(fmt(k) + '\n' for k in s))


def main(b0, b1, work):
    d = os.path.join(work, 's3000000big'); os.makedirs(d, exist_ok=True)
    if not os.path.exists(os.path.join(d, 'prep.json')):
        subprocess.run([sys.executable, os.path.join(HERE, 'prep.py'), b0, d, SCENE, str(N)], check=True)
    m = json.load(open(os.path.join(d, 'prep.json'))); inter = m['orders']['interleaved']
    # 2. leaf run
    lr = os.path.join(d, 'leaf.json')
    if not os.path.exists(lr):
        go(b1, os.path.join(d, 'rays_interleaved.bin'), lr, 16, ['--resume', '16', '--dump-gas', os.path.join(d, 'leaf.gas')])
    G = open(os.path.join(d, 'leaf.gas'), 'rb').read(); W = gas.walk(G)
    bases = sorted((off + struct.unpack_from('<i', G, off + 0x10)[0], off) for off in W if any(k == 'leaf' for _, k, _, _ in gas.slots(G, off)))
    bk = [b for b, _ in bases]
    offs = lanes(lr, 'lane_leaf_off')
    cand = {}
    for pos, i in enumerate(inter):
        o = offs[pos]
        if o < 0:
            continue
        k = bisect.bisect_right(bk, o) - 1
        c = [k] + [q for q in range(bisect.bisect_left(bk, bk[k]), k)] if bk[k] == o else [k]   # other nodes sharing this leaf base
        if bk[k] == o:
            p = k - 1
            while p > 0 and bk[p] == bk[k]:
                p -= 1
            c.append(p)
        cand[i] = [W[bases[q][1]]['path'] for q in c]
    keys = [tuple(k) for k in m['keys']]; pick = {i: 0 for i in cand}
    for i in cand:
        keys[i] = cand[i][0]
    # 3. hardware checks
    for it in range(4):
        write_starts(d, m, keys)
        cr = os.path.join(d, 'check%d.json' % it)
        go(b1, os.path.join(d, 'rays_interleaved.bin'), cr, 16, ['--resume', '16', '--start-file', os.path.join(d, 'starts_interleaved_ray.txt')])
        got = lanes(cr, 'lane_best'); ref = json.load(open(cr))['ref']
        wrong = [inter[pos] for pos in range(N) if ref[pos]['hit'] and not (got[pos][0] != '0xffffffff' and got[pos][1] == ref[pos]['prim'])]
        wrong_miss = sum(1 for pos in range(N) if not ref[pos]['hit'] and got[pos][0] != '0xffffffff')
        print('check %d: %d hits wrong, %d misses wrong' % (it, len(wrong), wrong_miss), flush=True)
        if not wrong:
            break
        for i in wrong:
            if i in pick and pick[i] + 1 < len(cand[i]):
                pick[i] += 1; keys[i] = cand[i][pick[i]]
    else:
        sys.exit('keys not resolved')
    # 4. grouped order from exact keys
    rays = np.fromfile(os.path.join(d, 'rays_base.bin'), np.float32).reshape(-1, 8)
    g = sorted(range(N), key=lambda i: (keys[i], i)); m['orders']['grouped'] = g; m['keys'] = [list(k) for k in keys]
    rays[g].tofile(os.path.join(d, 'rays_grouped.bin'))
    pw = lambda perm, dep: float(np.mean([len({keys[i][:dep] for i in perm[w:w + 32]}) for w in range(0, N, 32)]))
    m['distinct_per_warp_exact'] = {o: {dd: pw(m['orders'][o], dd) for dd in (1, 2, 3, 4)} for o in ('grouped', 'interleaved')}
    json.dump(m, open(os.path.join(d, 'prep.json'), 'w')); write_starts(d, m, keys)
    print('distinct exact subtrees per warp:', m['distinct_per_warp_exact'], flush=True)
    # 5. timing
    os.makedirs(os.path.join(d, 't'), exist_ok=True)
    V = [('interleaved', 'interleaved', None), ('grouped', 'grouped', None), ('b_grouped_warp', 'grouped', 'warp'), ('b_grouped_ray', 'grouped', 'ray')]
    for w in (4, 16):
        for st, ex in (('warm', ['--passes', '2']), ('cold', ['--flush', '256'])):
            for name, o, mode in V:
                sf = ['--start-file', os.path.join(d, 'starts_%s_%s.txt' % (o, mode))] if mode else []
                go(b1, os.path.join(d, 'rays_%s.bin' % o), os.path.join(d, 't', '%s_w%d_%s.json' % (name, w, st)), w, ['--resume', '16'] + ex + sf)
    # 6. ncu at W 16 (warm: skip pass 0, whose launch count the timing run of the same sequence gives)
    os.makedirs(os.path.join(d, 'n'), exist_ok=True)
    for st, ex in (('warm', ['--passes', '2']), ('cold', ['--flush', '256'])):
        for g_, mm in (('A', GA), ('B', GB)):
            runs = [('base', b0, 'interleaved', None, [])] + [(nm, b1, o, mode, ['--resume', '16']) for nm, o, mode in V[:3]]
            for name, binp, o, mode, rs in runs:
                sf = ['--start-file', os.path.join(d, 'starts_%s_%s.txt' % (o, mode))] if mode else []
                skip = []
                if st == 'warm':
                    tpath = os.path.join(d, 't', '%s_w16_warm.json' % name) if name != 'base' else None
                    k = sum(n for p, _, n in launches_per_chunk(tpath) if p == 0) if tpath else N // 512
                    skip = ['--ncu-skip', str(k)]
                base = os.path.join(d, 'n', ('base_w16_%s_%s' % (st, g_)) if name == 'base' else '%s_w16_%s_%s' % (name, st, g_))
                go(binp, os.path.join(d, 'rays_%s.bin' % o), base + '.json', 16, rs + ex + sf + ['--ncu', mm, '--ncu-out', base + '.csv'] + skip)


if __name__ == '__main__':
    main(*sys.argv[1:4])
