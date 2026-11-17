"""Struct_2 analysis: model/struct2_analyze.py notes/struct2"""
import glob, json, os, re, statistics as st, sys
d = sys.argv[1]; med = lambda x: st.median(x)
def sp(v): return 100 * (max(v) - min(v)) / med(v)
print('BUILD (ms, median of 15 builds, medians over 3 processes, spread over processes)')
B = {}
for f in sorted(glob.glob(d + '/build_N*.json')):
    m = re.match(r'build_N(\d+)_(\w+?)_p(\d)_', os.path.basename(f)); j = json.load(open(f)); B.setdefault((int(m[1]), m[2]), []).append(j)
for (N, fl), L in sorted(B.items()):
    b = [med(j['build_ms']) for j in L]; ex = ''
    if fl == 'update': ex = ' refit %.3f ms' % med([med(j['refit_ms']) for j in L])
    if fl == 'compact': ex = ' compact %.3f ms' % med([med(j['compact_ms']) for j in L])
    print('N %7d %-8s build %.3f ms (spread %.1f%%) bytes %.2f MB%s' % (N, fl, med(b), sp(b), L[0]['bytes'] / 1e6, ex))
print('INST (flat M=0 reference, same 1,000,000 triangles)')
I = {}
for f in sorted(glob.glob(d + '/inst_M*.json')):
    m = re.match(r'inst_M(\d+)_p(\d)_', os.path.basename(f)); j = json.load(open(f)); I.setdefault(int(m[1]), []).append(j)
flat = med([med(j['ms']) for j in I[0]])
for M, L in sorted(I.items()):
    t = [med(j['ms']) for j in L]; print('M %6d T %7s trace %.3f ms (spread %.1f%%) vs flat %.2f | GAS build %.2f ms IAS build %.3f ms | bytes %.1f MB | hit frac %.3f' % (M, L[0]['T'], med(t), sp(t), med(t) / flat, med([j['gas_build_ms'] for j in L]), med([j['ias_build_ms'] for j in L]), L[0]['bytes'] / 1e6, L[0]['hit_frac']))
print('PART (2,097,152 traced rays in all rows)')
P = {}
for f in sorted(glob.glob(d + '/part_A*.json')):
    m = re.match(r'part_A(\d+)_s(\d)_p(\d)_', os.path.basename(f)); j = json.load(open(f)); P.setdefault((int(m[1]), int(m[2])), []).append(j)
base = med([med(j['ms']) for j in P[(32, 0)]])
for (A, s), L in sorted(P.items(), reverse=True):
    t = [med(j['ms']) for j in L]; print('active %2d spread %d threads %d time %.3f ms (spread %.1f%%) per traced ray vs 32: %.2f | checksum %s' % (A, s, L[0]['threads'], med(t), sp(t), med(t) / base, ','.join(str(j['checksum']) for j in L)))
print('NCU')
for f in sorted(glob.glob(d + '/ncu_*.csv')):
    t = open(f).read().splitlines()
    for i, l in enumerate(t):
        if l.startswith('"ID"'):
            h = [c.strip('"') for c in l.split('","')]; v = [c.strip('"') for c in t[i + 2].split('","')]; g = dict(zip(h, v)); n = lambda k: float(g[k].replace(',', ''))
            print(os.path.basename(f).split('_2026')[0], 'warp inst %.4g thread inst %.4g sectors %.4g time_ns %.4g' % (n('smsp__inst_executed.sum'), n('smsp__thread_inst_executed.sum'), n('l1tex__t_sectors.sum'), n('gpu__time_duration.sum'))); break
