"""Arkade distance function variants (queue 14): launch time (probes off, CUDA events, median of 5) and result agreement with the double precision build. CPU only.
Run: /usr/bin/python3 -I model/q14/ark_variants.py"""
import re, os, glob, statistics as st
import numpy as np
LOG = os.path.expanduser('~/eoss/notes/q14/logs'); OUT = '/dev/shm/cb2'
cfgs = (('a1e5', '0.005', 1e5), ('a1e5', '0.0112', 1e5), ('a1e6', '0.0016', 1e5))
def ms(v, f, r):
    x = [float(re.search(r'ms ([\d.]+)', open(p).read()).group(1)) for p in sorted(glob.glob('%s/arkvar_%s_%s_%s_[0-9].log' % (LOG, v, f, r)))]
    return st.median(x), min(x), max(x)
def cb(v, f, r):
    t = open('%s/arkvar_%s_%s_%s_probes.log' % (LOG, v, f, r)).read(); m = re.search(r'CBK .* f0_cycles (\d+) f0_calls (\d+)', t); k = float(re.search(r'KERNEL queries \d+ ms ([\d.]+)', t).group(1))
    return int(m.group(1)) / int(m.group(2)), k
L = ['| config | variant | launch ms median (min to max of 5) | speedup over double | probe latency per call (cycles) | neighbours found (ids) differing from double | max relative difference of the distances (same id) |', '|---|---|---|---|---|---|---|']
for f, r, nq in cfgs:
    base = np.fromfile('%s/out_double_%s_%s.bin' % (OUT, f, r), dtype=[('i', '<i4'), ('d', '<f4')]); md = ms('double', f, r)[0]
    for v in ('double', 'float', 'square'):
        o = np.fromfile('%s/out_%s_%s_%s.bin' % (OUT, v, f, r), dtype=[('i', '<i4'), ('d', '<f4')]); a, b, c = ms(v, f, r)
        # compare as sets per query (the 16 slots are in insertion order of the traversal)
        kq = len(o) // 16; oi = o['i'].reshape(kq, 16); bi = base['i'].reshape(kq, 16); od = o['d'].reshape(kq, 16); bd = base['d'].reshape(kq, 16)
        diffq = sum(1 for q in range(kq) if set(oi[q]) != set(bi[q])); nfound = int((bi >= 0).sum()); nd = int(sum(len(set(oi[q]) ^ set(bi[q])) for q in range(kq)))
        rel = 0.0
        for q in range(kq):
            if set(oi[q]) == set(bi[q]):
                mo = {int(i): float(d) for i, d in zip(oi[q], od[q]) if i >= 0}; mb = {int(i): float(d) for i, d in zip(bi[q], bd[q]) if i >= 0}
                for i in mb: rel = max(rel, abs(mo[i] - mb[i]) / max(mb[i], 1e-30))
        L.append('| %s r %s | %s | %.2f (%.2f to %.2f) | %.2fx | %.0f | %d of %d queries differ (%d of %d slots) | %.2g |' % (f, r, v, a, b, c, md / a, cb(v, f, r)[0], diffq, kq, nd, nfound, rel))
print('\n'.join(L)); open(os.path.expanduser('~/eoss/model/results_arkade_float_4070ti.md'), 'w').write('# Arkade distance function: double (std::pow(float, int) promotes to double) against float powf and a plain square (queue 14, 4070 Ti, L2 build, 2D synthetic data)\n\n' + '\n'.join(L) + '\n')
