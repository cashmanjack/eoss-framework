"""Struct_1 analysis (CPU only). usage: /usr/bin/python3 model/struct_analyze.py flags|inst  -> markdown tables on stdout
Raw files: notes/struct1/<tag>_<TS>.json (+ .ncu.csv, .res); GAS dumps (flags item) in /dev/shm/st1/dumps."""
import csv, glob, json, os, re, struct, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../ttu/bench/BVH_5'); import gas
W = os.path.expanduser('~/eoss/notes/struct1')

def ncu(base):
    rows = [r for r in csv.reader(l for l in open(base + '.ncu.csv') if l.startswith('"'))]
    h, last = rows[0], rows[-1]; return lambda m: float(last[h.index(m)].replace(',', ''))

def load(tag):
    fs = sorted(f for f in glob.glob('%s/%s_2026*.json' % (W, tag)) if '.bigonly' not in f)
    assert len(fs) == 1, (tag, fs); base = fs[0][:-5]; j = json.load(open(fs[0])); j['base'] = base; j['ns'] = 1e6 * float(np.median(j['ms'])) / j['big']; j['ms_all'] = j['ms']
    reps = [j['ns']] + [1e6 * float(np.median(json.load(open(f))['ms'])) / j['big'] for f in sorted(glob.glob('%s/%s_r[0-9]_2026*.json' % (W, tag))) if '.bigonly' not in f]
    j['ns_runs'] = reps; j['ns'] = float(np.median(reps)); j['ns_spread'] = (max(reps) - min(reps)) / np.median(reps)
    if os.path.exists(base + '.ncu.csv'):
        g = ncu(base); n = j['big']
        j['req'] = (g('l1tex__t_sectors.sum') - g('l1tex__t_sectors_pipe_lsu.sum')) / n; j['lsu'] = g('l1tex__t_sectors_pipe_lsu.sum') / n; j['fills'] = g('l1tex__m_xbar2l1tex_read_sectors.sum') / n; j['winst'] = g('smsp__inst_executed.sum') / n
    if 'probe_clk' in j:
        c = np.array(j['probe_clk'], dtype=np.int64).reshape(j['probe_launches'], 32, 8)
        j['tr0'] = float(np.median(c[:, :, 2].max(axis=1) - c[:, :, 1].min(axis=1))); j['tr2'] = float(np.median(c[:, :, 6].max(axis=1) - c[:, :, 5].min(axis=1)))
    return j

def dumpstats(base, cust):
    d = glob.glob('/dev/shm/st1/dumps/%s.gas' % os.path.basename(base))
    if not d: return {}
    b = open(d[0], 'rb').read(); root = struct.unpack_from('<Q', b, 0x20)[0]; st = [root]; nn = 0; nl = 0; refs = 0; depth = {root: 1}; maxd = 1
    while st:
        o = st.pop(); nn += 1
        for s, k, ch, box in gas.slots(b, o):
            if k == 'node': st.append(ch); depth[ch] = depth[o] + 1; maxd = max(maxd, depth[ch])
            else:
                nl += 1
                if cust: c = ch & 7; refs += 8 if c == 0 else c
    return dict(nodes=nn, leaf=nl, prims=refs, depth=maxd, used=struct.unpack_from('<Q', b, 8)[0], refs_hdr=struct.unpack_from('<I', b, 0x60)[0], nodes_hdr=struct.unpack_from('<I', b, 0x58)[0])

def flags():
    print('| config | buffer B | final B | header used B | nodes (walk) | leaf slots | refs (hdr +0x60) | refs / prim | depth | ns / ray | rel time | run spread | unit req / ray | rel req | SM sectors / ray | warp inst / ray | one warp cold cyc | one warp warm cyc |'); print('|' + '---|' * 18)
    for scene, tags, N in (('soup 100k', [t for t in sorted(set(re.sub(r'_2026.*', '', os.path.basename(f)) for f in glob.glob(W + '/f_s100k_*.json')))], 100000),
                           ('soup 1M', [t for t in sorted(set(re.sub(r'_2026.*', '', os.path.basename(f)) for f in glob.glob(W + '/f_s1m_*.json')))], 1000000),
                           ('custom 100k pts', [t for t in sorted(set(re.sub(r'_2026.*', '', os.path.basename(f)) for f in glob.glob(W + '/f_cu_*.json')))], 100000)):
        tags = [t for t in tags if '.bigonly' not in t and not re.search(r'_r\d$', t)]; ref = load(tags[[x.split('_', 2)[2] for x in tags].index('none_u0_c0' if scene == 'soup 100k' else 'none_c0')]) if tags else None
        for t in tags:
            j = load(t); ds = dumpstats(j['base'], scene.startswith('custom')); 
            calls = ''
            if os.path.exists(j['base'] + '.res'):
                o = np.fromfile(j['base'] + '.res', np.uint32).reshape(-1, 8); calls = ' calls/q %.2f' % o[:, 3].mean()
            if not ds and not scene.startswith('custom'): ds = dict(nodes=j['hdr'][11] & 0xffffffff, refs_hdr=j['hdr'][12] & 0xffffffff, used=j['hdr'][1], leaf='', depth='')
            refs = ds.get('prims', ds.get('refs_hdr', 0)) if ds else 0
            print('| %s %s%s | %d | %d | %s | %s | %s | %s | %s | %s | %.4f | %.3f | %.3f | %.1f | %.3f | %.1f | %.0f | %s | %s |' % (scene, t.split('_', 2)[2], calls, j['buffer_bytes'], j['final_bytes'], ds.get('used', ''), ds.get('nodes', ''), ds.get('leaf', ''), ds.get('refs_hdr', ''), '%.3f' % (refs / N) if refs else '', ds.get('depth', ''), j['ns'], j['ns'] / ref['ns'], j['ns_spread'], j['req'], j['req'] / ref['req'], j['lsu'], j['winst'],
                  ('%.0f' % j['tr0']) if 'tr0' in j else '', ('%.0f' % j['tr2']) if 'tr2' in j else ''))

def inst():
    print('| config | ns / ray (rep1, rep2) | rel to id1 | unit req / ray | SM sectors / ray | warp inst / ray | rel warp inst | one warp cold cyc | warm cyc | hit frac | instance ids |'); print('|' + '---|' * 11)
    for scene, pre in (('soup', 'i_s100k_'), ('custom pts', 'i_cu_')):
        names = ['none', 'id1', 'r90_1', 'r37_1', 'id2', 'r90_2', 'id1any'] if scene == 'soup' else ['none', 'id1', 'id2', 'r90_2']
        J = {n: load(pre + n + ('_r1' if scene == 'soup' else '')) for n in names}; J2 = {n: load(pre + n + '_r2') for n in names} if scene == 'soup' else {}
        for n in names:
            j = J[n]; o = np.fromfile(j['base'] + '.res', np.uint32).reshape(-1, 8); ids = np.unique(o[o[:, 0] & 1 == 1][:, 4]) if scene == 'soup' else np.unique(o[:, 4])
            hit = (o[:, 0] & 1).mean() if scene == 'soup' else float('nan'); calls = '' if scene == 'soup' else ' calls/q %.2f' % o[:, 3].mean()
            r2 = (' , %.4f' % J2[n]['ns']) if J2 else ''
            print('| %s %s%s | %.4f%s | %.3f | %.1f | %.3f | %.1f | %.3f | %.0f | %.0f | %.4f | %s |' % (scene, n, calls, j['ns'], r2, j['ns'] / J['id1']['ns'], j['req'], j['lsu'], j['winst'], j['winst'] / J['id1']['winst'], j['tr0'], j['tr2'], hit, ','.join(str(int(x)) for x in ids[:4])))
    if True:   # per ray equality of id1, id2, none results
        a, b, c = [np.fromfile(J[n]['base'] + '.res', np.uint32).reshape(-1, 8) for n in ('none', 'id1', 'id2')] if False else (None, None, None)

if __name__ == '__main__':
    {'flags': flags, 'inst': inst}[sys.argv[1]]()
