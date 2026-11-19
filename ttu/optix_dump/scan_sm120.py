"""Scan memdump windows (files given as args, name ..._m<hexoffset>.bin = window start fnaddr-<offset>) for TTU instructions
and print them with raw fields and the EoSS decoder (SM from --sm, default 120). Output contains NVIDIA code: keep it in notes/.
    python3 -I scan_sm120.py [--sm 120] [--groups] FILE...
Fields (128 bit instruction, lo = bits 63:0, hi = 127:64): opcode lo[11:0]; guard predicate lo[15:12] (7 = PT, bit 15 = negate);
TTUST offset lo[55:40], Rb lo[39:32], Rc hi[7:0]; TTULD Rd lo[23:16], Rd2 hi[7:0], offset lo[55:40];
TTUMACROFUSE N lo[47:40]; TTUOPEN dual flag and close bit are hi bits (close: bit 74 = hi[10], bit 72 = hi[8]); scoreboard fields in hi[?] left to the decoder."""
import os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from eoss.isa.model import ISASpec
from eoss.tui.decode import Decoder
args = sys.argv[1:]
sm = 120
if '--sm' in args: i = args.index('--sm'); sm = int(args[i + 1]); del args[i:i + 2]
groups = '--groups' in args; args = [a for a in args if a != '--groups']
dec = Decoder(ISASpec.load(sm=sm))
NAMES = {0x3d0: 'TTUOPEN', 0x9d4: 'TTUMACROFUSE', 0x3d1: 'TTUST', 0x3d2: 'TTULD', 0x3d3: 'TTUGO', 0x3d5: 'TTUCCTL'}
for f in args:
    m = re.search(r'_m([0-9a-f]+)\.bin$', f); base = -int(m.group(1), 16)
    d = open(f, 'rb').read(); rows = []
    for i in range(0, len(d) - 15, 16):
        lo = int.from_bytes(d[i:i + 8], 'little'); hi = int.from_bytes(d[i + 8:i + 16], 'little')
        op = lo & 0xfff
        if op in NAMES: rows.append((base + i, i, lo, hi, op))
    print('%s: %d TTU instructions' % (os.path.basename(f), len(rows)))
    for off, i, lo, hi, op in rows:
        dd = dec.decode(i, d[i:i + 16])
        pred = (lo >> 12) & 0xf
        extra = ''
        if op == 0x9d4: extra = 'N=%d' % ((lo >> 40) & 0xff)
        if op == 0x3d1: extra = 'off=%#x Rb=R%d Rc=R%d' % ((lo >> 40) & 0xffff, (lo >> 32) & 0xff, hi & 0xff)
        if op == 0x3d2: extra = 'off=%#x Rd=R%d Rd2=R%d close74=%d close72=%d' % ((lo >> 40) & 0xffff, (lo >> 16) & 0xff, hi & 0xff, (hi >> 10) & 1, (hi >> 8) & 1)
        if op == 0x3d0: extra = 'hi=%#x' % hi
        print('fn%+#07x  %-13s pred=%#x %s  lo=%016x hi=%016x  | %s %s [%s]' % (off, NAMES[op], pred, extra, lo, hi, dd.mnemonic, dd.operands, dd.defn.name if dd.defn else '-'))
