"""Shape check of the IAS traversal group read from the private IAS pipeline dump (F91). No GPU, nothing written.
    ias_group_check.py IAS_DUMP.bin
Checks the A rules (checklist parts A) on the group at dump offset 0xbb60: one TTUOPEN (.DUAL), TTUMACROFUSE 0xF = number of TTU instructions after it,
8 TTUST (4 predicated: three @!P0 and one @P0), one TTUGO, 6 TTULD, bit 74 only on the last TTULD and unpredicated, bit 72 only on TTUOPEN, even register pairs.
Prints the instructions as offset, predicate, store offsets. This is the sequence a future --group ias build would copy."""
import sys
W = lambda b: int.from_bytes(b, 'little'); low12 = lambda b: W(b) & 0xfff
d = open(sys.argv[1], 'rb').read(); at = 0xbb60
ins = [d[at + 16 * k:at + 16 * k + 16] for k in range(17)]
OPEN, FUSE, ST, GO, LD = 0x3d0, 0x9d4, 0x3d1, 0x3d3, 0x3d2
res = []
ok = lambda n, c, i='': res.append((n, bool(c), i))
ok('TTUOPEN first, bit 72 (dual) set', low12(ins[0]) == OPEN and (W(ins[0]) >> 72) & 1)
n = (W(ins[1]) >> 40) & 0x1f
ok('TTUMACROFUSE second, count 15 = TTU instructions that follow', low12(ins[1]) == FUSE and n == 15 and len(ins) - 2 == n, n)
g = ins[2:]
kinds = [low12(b) for b in g]
ok('shape: 8 TTUST, TTUGO, 6 TTULD', kinds == [ST] * 8 + [GO] + [LD] * 6, [hex(k) for k in kinds])
pr = lambda b: ((W(b) >> 12) & 7, (W(b) >> 15) & 1)
ok('four predicated stores: three @!P0 (0x100, 0x180, 0x190) and one @P0 (0x200)', [pr(b) for b in g[:8]] == [(7, 0)] * 4 + [(0, 1)] * 3 + [(0, 0)], [pr(b) for b in g[:8]])
offs = [(W(b) >> 40) & 0xffff for b in g[:8]]; ok('store offsets 0, 0x40, 0x50, 0x80, 0x100, 0x180, 0x190, 0x200', offs == [0, 0x40, 0x50, 0x80, 0x100, 0x180, 0x190, 0x200], [hex(o) for o in offs])
lds = [(W(b) >> 40) & 0xffff for b in g[9:]]; ok('load offsets 0x300 0x310 0x320 0x380 0x3C0 0x3D0', lds == [0x300, 0x310, 0x320, 0x380, 0x3c0, 0x3d0], [hex(o) for o in lds])
close = [i for i, b in enumerate(g[9:]) if (W(b) >> 74) & 1]; ok('bit 74 only on the last TTULD, unpredicated', close == [5] and pr(g[-1]) == (7, 0), close)
ok('bit 72 only on TTUOPEN', not any((W(b) >> 72) & 1 for b in g))
reg = lambda b, lo: (W(b) >> lo) & 0xff
odd = [r for b in g if low12(b) in (ST, LD) for r in ((reg(b, 32), reg(b, 64)) if low12(b) == ST else (reg(b, 16), reg(b, 64))) if r != 255 and r % 2]
ok('every TTUST / TTULD register pair base is even or RZ', not odd, odd)
for k, b in enumerate(ins): print('%2d %s pred %s' % (k, hex(low12(b)), pr(b)))
for nme, c, i in res: print('%-85s %s %s' % (nme, 'PASS' if c else 'FAIL', i))
print('ALL PASS' if all(r[1] for r in res) else 'FAILED'); sys.exit(0 if all(r[1] for r in res) else 1)
