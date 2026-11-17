"""Find the raygen constants in each variant: MOV R37, imm (low 32 bits 0x00257802) and IMAD.MOV.U32 R17, RZ, RZ, imm (low 32 0xff117424),
R13 (tmin) 0xff0d7424. Prints offset, immediate and its decode, plus whether the traversal group region is byte identical to base."""
import glob, os, struct
D = '/dev/shm/flagenc'
W = {os.path.basename(p).split('_2026')[0]: open(p, 'rb').read() for p in sorted(glob.glob(D + '/*.bin'))}
b = W['base']
def find(v, lo32):
    return [(i - 0x10000, struct.unpack('<Q', v[i:i+8])[0] >> 32) for i in range(0, len(v), 16) if struct.unpack('<I', v[i:i+4])[0] == lo32]
print('%-20s %-34s %-30s %s' % ('variant', 'MOV R37 (off, imm)', 'R17 tmax (off, imm)', 'R13 tmin'))
for n, v in W.items():
    print('%-20s %-34s %-30s %s' % (n, [(hex(o), hex(x)) for o, x in find(v, 0x00257802)][:3], [(hex(o), hex(x)) for o, x in find(v, 0xff117424)][:2], [(hex(o), hex(x)) for o, x in find(v, 0xff0d7424)][:2]))
# traversal region: fnaddr-0x10000 .. -0xdca0 (code identical on the three drivers); and group 0xfb50 .. +0x0
for n, v in W.items():
    diffs = [i - 0x10000 for i in range(0, 0x10000 - 0x4000, 16) if v[i:i+16] != b[i:i+16]]
    # count only slots where base is non-zero
    diffs = [d for d in diffs if b[d + 0x10000: d + 0x10010] != bytes(16)]
    print('%-20s differing nonzero slots in fnaddr-0x10000..-0x4000: %d' % (n, len(diffs)), [hex(d) for d in diffs[:5]])
