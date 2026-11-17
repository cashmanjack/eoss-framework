"""Diff each variant's 64 KB window against the base (noise floor = second base run). Prints instruction slots that differ
(16 byte slots, offset relative to fnaddr-0x10000) with the low/high 64 bit words, and classifies the offset."""
import glob, sys, struct, os
D = '/dev/shm/flagenc'
def load(p): return open(p, 'rb').read()
bases = sorted(glob.glob(D + '/base_*.bin'))
b0, b1 = load(bases[0]), load(bases[1])
def slots(a, b): return [i for i in range(0, len(a), 16) if a[i:i+16] != b[i:i+16]]
noise = set(slots(b0, b1))
print('noise slots (base vs base):', len(noise), 'first', [hex(x) for x in sorted(noise)[:8]])
for p in sorted(glob.glob(D + '/*.bin')):
    n = os.path.basename(p).split('_2026')[0]
    if n == 'base': continue
    d = [i for i in slots(b0, load(p)) if i not in noise]
    print('\n%s: %d slots differ beyond noise' % (n, len(d)))
    v = load(p)
    for i in d[:12]:
        lo0, hi0 = struct.unpack('<QQ', b0[i:i+16]); lo1, hi1 = struct.unpack('<QQ', v[i:i+16])
        print('  fnaddr%+#x  base %016x %016x -> %016x %016x   xor lo %016x hi %016x' % (i - 0x10000, hi0, lo0, hi1, lo1, lo0 ^ lo1, hi0 ^ hi1))
