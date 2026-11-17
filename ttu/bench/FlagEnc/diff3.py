import glob, os, json
D = '/dev/shm/flagenc'
ld = lambda p: open(p, 'rb').read()
W = {os.path.basename(p).split('_2026')[0]: ld(p) for p in sorted(glob.glob(D + '/*.bin'))}
b = W['base']
big = ['f_CULL_BACK','f_CULL_FRONT','f_CULL_OPAQUE','f_CULL_NON_OPAQUE','f_TERMINATE','m_1','m_15','m_128','t_2','tmin_0p1']
d = json.load(open(D + '/diff2.json'))
offs = sorted(x[0] for x in d['f_CULL_BACK'])
print('big group: first/last differing slot', hex(offs[0]), hex(offs[-1]), len(offs))
# are the big variants identical to each other in the non-pointer code region 0x0..0x10000 where base nonzero?
def code_slots(v): return {i: v[i:i+16] for i in range(0, 0x10000, 16) if b[i:i+16] != bytes(16)}
ref = big[0]
for n in big[1:]:
    diff = [i for i in range(0, 0x10000, 16) if b[i:i+16] != bytes(16) and W[ref][i:i+16] != W[n][i:i+16]]
    print(ref, 'vs', n, len(diff), [hex(i - 0x10000) for i in diff[:6]])
# shift test: for t_2, find shift s (slots) such that variant[i] == base[i-s] over the differing region
v = W['t_2']
lo, hi = offs[0] + 0x10000, offs[-1] + 0x10000
for s in range(-40, 41):
    m = sum(1 for i in range(lo, hi, 16) if 0 <= i - 16*s < 0x10000 and v[i:i+16] == b[i-16*s:i-16*s+16] and b[i:i+16] != bytes(16))
    if m > 20: print('shift', s, 'slots matching', m, 'of', (hi-lo)//16)
