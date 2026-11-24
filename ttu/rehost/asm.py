"""A small assembler for the rehosted kernels: every instruction is built by the EoSS encoder (class and fields named here) and then given
its control word (stall, yield, scoreboards) with the project's own control writer. No dump, no OptiX code, no binary is read.

    from ttu.rehost import asm as A;  A.ldg(2, 98, 96, 0)     # LDG.E R98, [R96.64 + 0], write scoreboard 2

Predication: pass pg=(reg, not) to any builder (default PT). Registers are numbers, 255 = RZ. Every builder returns 16 bytes.
Each builder is checked by decoding in `selftest()` (mnemonic text and 0 failed checks)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from eoss.enc.generator import Encoder
from eoss.enc import control as ctrl

_enc = Encoder(sm='89')
RZ = 255
PT = 7


def _ctl(b, stall, wr=7, rd=7, wt=0):
    return ctrl.write_control(b, stall=stall, yield_=0 if stall >= 12 else 1, wr_bar=wr, rd_bar=rd, wt_bar=wt)


def _b(cls, f, pg=None, stall=1, wr=7, rd=7, wt=0):
    f = dict(f)
    if pg is not None and pg != (PT, 0):
        f['Pg'] = 'P%d' % pg[0] if pg[0] != PT else 'PT'
        f['Pg_not'] = pg[1]
    ins = _enc.build(cls, f, opex=17)
    assert ins.valid, (cls, f, ins.failed_conditions)
    return _ctl(bytes(ins.bytes), stall, wr, rd, wt)


R = lambda n: 'RZ' if n == RZ else 'R%d' % n
P = lambda n: 'PT' if n == PT else 'P%d' % n


def shr_u32(rd, ra, imm, pg=None, stall=6):
    """SHF.R.U32.HI Rd, RZ, imm, Ra  (Rd = Ra >> imm, logical). sz codes: 0 S64, 1 U64, 2 S32, 3 U32."""
    return _b('shf__RuIR_RIR', {'Rd': R(rd), 'Ra': 'RZ', 'Rc': R(ra), 'Ra_offset': imm, 'memdesc': 1, 'relu': 0, 'sz': 3, 'ftz': 1}, pg, stall)


def shl64_hi(rd, lo, imm, hi, pg=None, stall=6):
    """SHF.L.U64.HI Rd, Rlo, imm, Rhi  (high word of (Rhi:Rlo) << imm)."""
    return _b('shf__RuIR_RIR', {'Rd': R(rd), 'Ra': R(lo), 'Rc': R(hi), 'Ra_offset': imm, 'memdesc': 0, 'relu': 0, 'sz': 1, 'ftz': 1}, pg, stall)


def imad_imm(rd, ra, imm, rc=RZ, pg=None, stall=6):
    """IMAD.U32 Rd, Ra, imm, Rc  (low 32 bits of Ra * imm + Rc)."""
    return _b('imad__RsIR_RIR', {'Rd': R(rd), 'Ra': R(ra), 'Ra_offset': imm, 'Rc': R(rc), 'fmt': 'U32' if False else 0}, pg, stall)


def iadd3_imm(rd, ra, imm, pg=None, stall=6):
    """IADD3 Rd, Ra, imm, RZ."""
    return _b('iadd3_imm__RsIR_RIR', {'Rd': R(rd), 'Ra': R(ra), 'Ra_offset': imm & 0xffffffff, 'Rc': 'RZ'}, pg, stall)


def iadd3(rd, ra, rb, pg=None, stall=6):
    """IADD3 Rd, Ra, Rb, RZ."""
    return _b('iadd3_noimm__RRR_RRR', {'Rd': R(rd), 'Ra': R(ra), 'Rb': R(rb), 'Rc': 'RZ'}, pg, stall)


def mov(rd, rs, pg=None, stall=1):
    return _b('mov__RR', {'Rd': R(rd), 'Rb': R(rs), 'PixMaskU04': 15}, pg, stall)


def fadd_neg(rd, ra, rb, pg=None, stall=6):
    """FADD Rd, Ra, -Rb."""
    return _b('fadd__RRR_RR', {'Rd': R(rd), 'Ra': R(ra), 'Rb': R(rb), 'Sb_negate': 1}, pg, stall)


def fmul(rd, ra, rb, pg=None, stall=6):
    return _b('fmul__RIR_RI', {}, pg, stall) if False else _b('fmul__RRR_RR', {'Rd': R(rd), 'Ra': R(ra), 'Rb': R(rb)}, pg, stall)


def ffma(rd, ra, rb, rc, pg=None, stall=6):
    return _b('ffma__RRR_RRR', {'Rd': R(rd), 'Ra': R(ra), 'Rb': R(rb), 'Rc': R(rc)}, pg, stall)


# comparison codes (icmp / fcomp)
ICMP = {'LT': 1, 'EQ': 2, 'LE': 3, 'GT': 4, 'NE': 5, 'GE': 6}


def isetp_imm(pu, cmp, ra, imm, pp=PT, pg=None, stall=13):
    """ISETP.cmp.AND Pu, PT, Ra, imm, Pp."""
    return _b('isetp_simple__RsIR_RIR_noEX', {'Pu': P(pu), 'Ra': R(ra), 'Ra_offset': imm, 'dstfmt': ICMP[cmp], 'Pp': P(pp)}, pg, stall)


def isetp_reg(pu, cmp, ra, rb, pp=PT, pg=None, stall=13):
    """ISETP.cmp.AND Pu, PT, Ra, Rb, Pp."""
    return _b('isetp_simple__RRR_RRR_noEX', {'Pu': P(pu), 'Ra': R(ra), 'Rb': R(rb), 'dstfmt': ICMP[cmp], 'Pp': P(pp)}, pg, stall)


def fsetp_le(pu, ra, rb, pp=PT, pg=None, stall=13):
    """FSETP.LE.AND Pu, PT, Ra, Rb, Pp  (float compare, ordered)."""
    return _b('fsetp_simple__RRR_RRR', {'Pu': P(pu), 'Ra': R(ra), 'Rb': R(rb), 'cmp': 3, 'Pp': P(pp)}, pg, stall)


def ldg(rd, ra, imm=0, pg=None, sb=2, stall=1):
    """LDG.E.STRONG.SM Rd, [Ra.64 + imm] (L1 allocating, nvcc 12.8's ld.global.ca encoding, opcode 0x981, memory descriptor form). This is the form the project's loop builds already use on hardware
    (build_patch.ldg_ca); the encoder's `ldg_uniform__Ra64` variant (no descriptor) did NOT make the consumers wait (stale registers, found in the R3 stage 0 run), so it is not used.
    Writes scoreboard sb; the consumer waits with wt = 1 << sb. Optional predicate pg = (reg, not).
    A wait issued within about 2 cycles of the load does not see it pending (R3 stage 0 runs): give the last load of a batch stall >= 4 before the waiting instruction."""
    assert 0 <= imm < (1 << 23)
    b = ctrl.write_control((0x7981 | (rd << 16) | (ra << 24) | (imm << 40) | (0xc1eb900 << 64)).to_bytes(16, 'little'), stall=stall, yield_=1, wr_bar=sb, rd_bar=7, wt_bar=0)
    if pg is not None and pg != (PT, 0):
        b = (int.from_bytes(b, 'little') & ~(0xf << 12) | (pg[0] << 12) | (pg[1] << 15)).to_bytes(16, 'little')
    return b


def stg(ra, rb, imm=0, pg=None, rd=7):
    """STG.E [Ra.64 + imm], Rb (64 bit register address, weak; opcode 0x986, uniform register slot = URZ)."""
    return _b('stg_uniform__Ra64', {'Ra': R(ra), 'Rb': R(rb), 'Ra_offset': imm, 'Ra_URc': 63}, pg, 1, rd=rd)


def nop(stall=2, wt=0):
    return ctrl.write_control(ctrl.NOP_BYTES, wt_bar=wt, stall=stall)


def selftest():
    from eoss.isa.model import ISASpec
    from eoss.tui.decode import Decoder
    dec = Decoder(ISASpec.load(sm=89))
    items = [('shr_u32', shr_u32(98, 33, 25)), ('shl64_hi', shl64_hi(97, 44, 3, 45)), ('imad_imm', imad_imm(96, 44, 8)), ('imad_add', imad_imm(100, 98, 16, 6)),
             ('iadd3_imm', iadd3_imm(9, 9, 1, pg=(5, 0))), ('iadd3', iadd3(9, 9, 10)), ('mov', mov(2, 70)), ('fadd_neg', fadd_neg(105, 102, 82)), ('fmul', fmul(108, 105, 105)),
             ('ffma', ffma(108, 106, 106, 108)), ('isetp_imm', isetp_imm(4, 'EQ', 98, 0x50)), ('isetp_reg', isetp_reg(4, 'NE', 3, RZ, pp=4)), ('fsetp_le', fsetp_le(5, 108, 8, pp=4)),
             ('ldg', ldg(98, 96, 0, pg=(4, 0))), ('stg', stg(116, 98, 0, pg=(6, 0)))]
    for name, b in items:
        r = dec.decode(0, b)
        print('%-14s failed=%d  %s' % (name, r.n_failed, r.text))


if __name__ == '__main__':
    selftest()
