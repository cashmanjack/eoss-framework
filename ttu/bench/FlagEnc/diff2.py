"""Code only diff: slots that differ from BOTH base runs and are identical between the two base runs, ignoring slots with
process pointers (0x55xx / 0x56xx / 0x7fxx in bits 40..47 of a 64 bit word). Prints xor of the 128 bit slot."""
import glob, os, struct
D = '/dev/shm/flagenc'
ld = lambda p: open(p, 'rb').read()
b = [ld(p) for p in sorted(glob.glob(D + '/base_*.bin'))]
def ptr(w): return ((w >> 40) & 0xFF) in (0x55, 0x56, 0x7f, 0x7e) and (w >> 48) == 0
rows = {}
for p in sorted(glob.glob(D + '/*.bin')):
    n = os.path.basename(p).split('_2026')[0]
    if n == 'base': continue
    v = ld(p); out = []
    for i in range(0, 0x10000, 16):
        s0, s1, sv = b[0][i:i+16], b[1][i:i+16], v[i:i+16]
        if s0 != s1 or sv == s0 or s0 == bytes(16): continue
        lo, hi = struct.unpack('<QQ', sv); lo0, hi0 = struct.unpack('<QQ', s0)
        if ptr(lo) or ptr(hi) or ptr(lo0) or ptr(hi0): continue
        out.append((i - 0x10000, lo0, hi0, lo, hi))
    rows[n] = out
    print('%-20s %3d slots' % (n, len(out)))
import json
json.dump({k: v for k, v in rows.items()}, open('/dev/shm/flagenc/diff2.json', 'w'))
for n, out in rows.items():
    print('\n' + n)
    for off, lo0, hi0, lo, hi in out[:10]:
        print('  fnaddr%+#7x  %016x %016x -> %016x %016x' % (off, hi0, lo0, hi, lo))
