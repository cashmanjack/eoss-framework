"""Phase 10i test 1 (c): RT ratio on the siblings against pages per warp load (stream l2fp<MB>p<P>), footprints 64 and 512 MB. CPU only.
/usr/bin/python3 model/phase10i_pages.py DIR       (the latest DIR/c4_pages_*.json and its c4clk csv)
r = RT rays/us with the stream over the alone run just before it (same repetition), median over repetitions; knee = the P where r crosses the geometric midpoint of r(P=1) and r(P=32)."""
import csv, glob, json, math, os, statistics as st, sys
d = sys.argv[1]; jf = sorted(glob.glob(os.path.join(d, 'c4_pages_*.json')))[-1]; rows = json.load(open(jf)); by = {r['seq']: r for r in rows}
clk = {}
for r in csv.DictReader(open(jf.replace('c4_pages_', 'c4clk_pages_').replace('.json', '.csv'))): clk.setdefault(r['key'], []).append(float(r['sm']))
res = {}
for r in rows:
    if not r['compute']: continue
    al = by[r['seq'] - 1]; F, P = [int(x) for x in r['config'][8:-1].split('p')]
    res.setdefault((F, P), []).append((r['rays_per_us'] / al['rays_per_us'], st.median(r['lines_per_s']) / 1e9, st.median(clk.get(r['key'], [0])), st.median(clk.get(al['key'], [0])), r['wrong']))
print('footprint MB, pages/load: r (min-max over reps), G lines/s, SM MHz with / alone')
for (F, P), v in sorted(res.items()):
    rs = [x[0] for x in v]; print('%4d %2d  r %.3f (%.3f-%.3f)  %.2f  %.0f / %.0f  wrong %d  n %d' % (F, P, st.median(rs), min(rs), max(rs), st.median(x[1] for x in v), st.median(x[2] for x in v), st.median(x[3] for x in v), sum(x[4] for x in v), len(v)))
for F in (64, 512):
    r = {P: st.median(x[0] for x in res[(F, P)]) for (f, P) in res if f == F}
    if 1 in r and 32 in r:
        mid = math.sqrt(r[1] * r[32]); knee = next((P for P in sorted(r) if r[P] <= mid), None) if r[1] > r[32] else None
        print('F %d: r(1) %.3f r(32) %.3f ratio %.2f, midpoint %.3f, first P at or below it: %s' % (F, r[1], r[32], r[1] / r[32], mid, knee))
