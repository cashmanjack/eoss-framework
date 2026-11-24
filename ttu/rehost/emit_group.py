"""Emit the SM89 GAS traversal TTU group (OPEN.DUAL, FUSE 17, 10 ST, GO, 6 LD.CLOSE last) from the EoSS encoder.

Rehosting rule (Jack, 2026-11-24): every instruction of a rehosted kernel comes from the EoSS encoder with fields
chosen here. This module reads no dump, no OptiX code and no binary: its only inputs are the encoder and the
tables below (offsets, registers, predicates, scoreboards, stall counts = interface facts from notes/protocol.md).
A separate offline script (notes/rehost_dump_compare.py, private, never imported by a build) compares the output
with the checked group's fields and reports differences.

    python3 -m ttu.rehost.emit_group      # prints the 19 instructions and their decode (no GPU)
"""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from eoss.enc.generator import Encoder

RZ = 'RZ'
# (offset, Rb, Rc, Pg, Pg_not): TTUST Rb is the high 64 bit source, Rc the low one. Pg 'P0' / PT = predicate register.
STORES = [
    (0xC0, 'R52', 'R56', 'P0', 0), (0xD0, 'R30', 'R46', 'P0', 0), (0x200, 'R54', 'R34', 'P0', 1),
    (0x100, 'R54', 'R34', 'P0', 0), (0x180, 'R32', 'R70', 'P0', 0), (0x190, 'R2', 'R44', 'P0', 0),
    (0x00, 'R62', 'R68', 'PT', 0), (0x40, 'R12', 'R82', 'PT', 0), (0x50, 'R16', 'R14', 'PT', 0), (0x80, 'R58', 'R60', 'PT', 0),
]
# (offset, Rd, Rd2): the last one closes the session
LOADS = [(0x300, 'R44', 'R32'), (0x310, 'R30', 'R68'), (0x320, 'R2', RZ), (0x380, 'R34', 'R62'), (0x3C0, 'R56', 'R58'), (0x3D0, 'R60', 'R70')]
FUSE = len(STORES) + 1 + len(LOADS)      # 17: counts every slot, predicated or not (protocol R1)
# opex = stall + 16 * yield (the encoder's table value; stall 6 on OPEN, 4 on FUSE, 1 on ST, GO, LD, 2 on the closing LD)
OPEX = dict(open=22, fuse=20, st=17, go=17, ld=17, close=18)
READ_SB, WRITE_SB = 0, 1                 # TTUST read scoreboard 0, TTULD write scoreboard 1 (protocol, Interface_2)

_enc = Encoder(sm='89')


def _b(cls, fields, opex):
    ins = _enc.build(cls, fields, opex=opex)
    assert ins.valid, (cls, fields, ins.failed_conditions)
    return bytes(ins.bytes)


def group():
    """The 19 instructions (OPEN, FUSE, 10 TTUST, TTUGO, 6 TTULD) as 16 byte values."""
    g = [_b('ttuopen_', {'dual': 1}, OPEX['open']), _b('ttumacrofuse_', {'Sb': FUSE}, OPEX['fuse'])]
    for off, rb, rc, pg, neg in STORES:
        f = {'ImmU16': off, 'Rb': rb, 'Rc': rc, 'src_rel_sb': READ_SB}
        if pg != 'PT': f.update(Pg=pg, Pg_not=neg)
        g.append(_b('ttust_', f, OPEX['st']))
    g.append(_b('ttugo_', {}, OPEX['go']))
    for k, (off, rd, rd2) in enumerate(LOADS):
        last = k == len(LOADS) - 1
        g.append(_b('ttuld__close' if last else 'ttuld__no_close',
                    {'ImmU16': off, 'Rd': rd, 'Rd2': rd2, 'dst_wr_sb': WRITE_SB}, OPEX['close'] if last else OPEX['ld']))
    assert len(g) == 19 and all(len(x) == 16 for x in g)
    return g


def isetp_p0_true(ra=90, imm=0xC0000):
    """ISETP.NE.AND P0, PT, R<ra>, imm, PT (immediate form): with R<ra> = 0 and imm nonzero P0 is true, which selects the
    P0 true path of the group (9 stores execute). icmp code 5 = NE."""
    return _b('isetp_simple__RsIR_RIR_noEX', {'Pu': 'P0', 'Ra': 'R%d' % ra, 'Ra_offset': imm, 'dstfmt': 5}, 17)


if __name__ == '__main__':
    from eoss.isa.model import ISASpec
    from eoss.tui.decode import Decoder
    dec = Decoder(ISASpec.load(sm=89))
    for k, b in enumerate(group()):
        r = dec.decode(0, b)
        print('%2d %s  failed=%s' % (k, b.hex(), r.n_failed))
