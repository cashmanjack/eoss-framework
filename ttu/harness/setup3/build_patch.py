"""Rewrite ttu_one's body in a copy of the setup3 binary.

    build_patch.py --stage 0|1 --bin /dev/shm/setup3/setup3 --dump DUMP.bin

Body: nvcc prologue | in pointer -> R92:R93 | out pointer -> R4:R5 | one load per
input register (load k writes register k) | NOP waiting on every scoreboard |
ISETP P0 (OptiX's own, operand R90) | NOP stall 15 | GROUP | NOP waiting on every scoreboard |
stores of R2, R3, R6..R95 (store k reads register k) | EXIT | BRA self.
GROUP is 19 NOPs in stage 0 and OptiX's fused traversal group in stage 1.

The OptiX instructions (the group and one ISETP) are read from the private dump at
build time and never written into the repository. The output binary holds them, so
it lives in /dev/shm and is deleted after the run.
"""
import argparse, hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
from eoss.binary.patcher import BinaryPatcher
from eoss.binary import elf
from eoss.enc import control as ctrl

DUMP_BASE = 0x14000        # the dump starts at fnaddr - 0x14000
GROUP_AT = -0xfb50          # TTUOPEN.DUAL of the traversal group, fnaddr relative
GROUP_LEN = 19              # OPEN, FUSE, 10 ST, GO, 6 LD
ISETP_AT = -0xfd10          # ISETP.NE.AND P0, PT, R46, 0xC0000, PT
INPUTS = [2, 3, 12, 13, 14, 15, 16, 17, 30, 31, 32, 33, 34, 35, 44, 45, 46, 47,
          52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 68, 69, 70, 71, 82, 83, 90]
OUTPUTS = [2, 3] + list(range(6, 96))
# --ias (BVH_4 / F91): the IAS traversal group read from the private IAS pipeline dump (memdump launch --ias): 17 instructions
# (OPEN.DUAL, FUSE 0xF, 8 ST of which 4 predicated, GO, 6 LD), and the pipeline's own ISETP (P0 = R24 != 1, fresh query: true).
# Registers the stores read (logical slot k = register k; the host fills them, see setup3.cu --ias): store 0 R2:R3 / R48:R49, 0x40 R10:R11 / R12:R13,
# 0x50 R14:R15 / R16:R17, 0x80 R52:R53 / R48:R49, 0x200 R6:R7 (pointer) / R24:R25 (zero), and the predicated off stores' sources.
IAS_GROUP_AT = 0xbb60       # dump file offset of TTUOPEN.DUAL
IAS_ISETP_AT = 0xbab0       # ISETP.NE.AND P0, PT, R24, 1, PT
IAS_GROUP_LEN = 17
IAS_INPUTS = [2, 3, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17, 24, 25, 26, 27, 28, 29, 40, 41, 48, 49, 50, 51, 52, 53, 90]
# --sm 120 (pc14, RTX 5090; prepared 2026-11-19, never run): the SM120 OptiX GAS traversal group from the SM120 dump (notes/setup4_sm120/dump_sm120_fn-0x20000_len0x20000.bin,
# starts at fnaddr - 0x20000): TTUOPEN.DUAL at fnaddr-0x10120, 19 instructions, the ISETP that decides P0 (R6 & 0xC0000) at fnaddr-0x10320. Plain builds only (like --ias).
# Registers differ from the SM89 group: INPUTS are derived from the group (store source pairs) and the host fills them with its --sm120 register image.
SM120_DUMP_BASE, SM120_GROUP_AT, SM120_ISETP_AT, SM120_ISETP_REG = 0x20000, -0x10120, -0x10320, 6
IN_PTR = 92
ALU_REG = 91                  # --alu: accumulator (plain layout R91; with --remap it is R45, free in that layout)

NOP_PLACEHOLDER = bytes(16)
W = lambda b: int.from_bytes(b, 'little')
B = lambda v: v.to_bytes(16, 'little')
low12 = lambda b: W(b) & 0xfff


def field(b, lo, val, width=8):
    return B(W(b) & ~(((1 << width) - 1) << lo) | (val << lo))


def cs2r_clock(rd):
    """CS2R Rd, SR_CLOCKLO (64 bit), as nvcc 12.8 encodes it for SM89: opcode 0x805,
    SR id 0x50 in bits 79:72, no scoreboard, stall 2."""
    lo = 0x7805 | (rd << 16)          # 0x7000: Pg = PT
    hi = (0x50 << 8) | (0x1 << 16)    # SR_CLOCKLO, size field as nvcc emits
    instr = (lo | (hi << 64)).to_bytes(16, 'little')
    return ctrl.write_control(instr, stall=2, yield_=1, wr_bar=7, rd_bar=7, wt_bar=0)


def bar_sync0():
    """BAR.SYNC 0x0, blocking form (opcode 0xb1d, barmode SYNC, bit 80 = 0). nvcc's
    __syncthreads uses BAR.SYNC.DEFER_BLOCKING (bit 80 = 1); with that form a warp
    keeps issuing past the barrier, so a clock read right after it sees the arrival
    time, not the release time (measured: 2,900 cycles of skew at 20 warps)."""
    instr = (0x7b1d).to_bytes(16, 'little')
    return ctrl.write_control(instr, stall=1, yield_=1, wr_bar=7, rd_bar=7, wt_bar=0)


def _instr(lo, hi_low, stall, yld, wr, rd, wt):
    return ctrl.write_control((lo | (hi_low << 64)).to_bytes(16, 'little'), stall=stall, yield_=yld, wr_bar=wr, rd_bar=rd, wt_bar=wt)


def nanosleep_p1(ns):
    """@P1 NANOSLEEP ns, as nvcc 12.8 encodes nanosleep.u32: opcode 0x95d, sleep time in ns in bits 63:32,
    PT in the predicate field at bits 89:87, stall 2, yield, no scoreboard."""
    return _instr(0x095d | (1 << 12) | (ns << 32), 7 << 23, 2, 1, 7, 7, 0)   # guard predicate (bits 14:12) = P1


def iadd3_imm(rd, ra, imm, stall):
    """IADD3 Rd, Ra, imm, RZ (nvcc 12.8: opcode 0x810, immediate in bits 63:32), no scoreboards."""
    return _instr(0x810 | (7 << 12) | (rd << 16) | (ra << 24) | (imm << 32), 0x7ffe0ff, stall, 0, 7, 7, 0)   # guard predicate PT; no yield hint (stall >= 12 forbids it)


def imov_cbank(rd, off, stall):
    """IMAD.MOV.U32 Rd, RZ, RZ, c[0x0][off] (nvcc 12.8: opcode 0x624, constant offset / 4 in bits 47:40; verified
    byte for byte against nvcc's own output for c[0x0][0x168] and c[0x0][0x16c])."""
    return _instr(0x7624 | (rd << 16) | (0xff << 24) | (((off >> 2) << 8) << 32), 0x78e00ff, stall, 1, 7, 7, 0)


# --slim (Throughput_6 / 7): the 8 ray words of a lane live in slots 8..15 of its record (one 32 byte sector); the
# other input words are lane invariant and come from the shared record at the start of `in` (a broadcast load
# through a pointer made from c[0x0][0x160]); only SLIM_OUT words are stored, to slots 0..12 of the lane's out record.
# The TTU group is not touched.
RAYSLOT = {12: 8, 13: 9, 14: 10, 15: 11, 16: 12, 17: 13, 82: 14, 83: 15,
           52: 8, 53: 9, 56: 14, 57: 15, 30: 12, 31: 13, 46: 10, 47: 11}      # copies read the same slot as their source
SLIM_OUT = [3, 32, 33, 44, 45, 56, 57, 58, 59, 60, 61, 70, 71]


def s2r_sr(rd, sr):
    """S2R Rd, SR_<sr> (same encoding as s2r_virtualsmid with another SR id; writes scoreboard 2)."""
    return _instr(0x7919 | (rd << 16), sr << 8, 2, 1, 2, 7, 0)


def s2r_virtualsmid(rd):
    """S2R Rd, SR_VIRTUALSMID (what PTX %smid compiles to), as nvcc 12.8 encodes it:
    opcode 0x919, SR id 0x43, writes scoreboard 2."""
    return _instr(0x7919 | (rd << 16), 0x43 << 8, 2, 1, 2, 7, 0)


def isetp_ne(pu, ra, rb, stall, wt):
    """ISETP.NE.AND Pu, PT, Ra, Rb, PT (nvcc 12.8 field layout; Pu in bits 81..83)."""
    return _instr(0x720c | (ra << 24) | (rb << 32), 0x3f05270 | (pu << 17), stall, 0, 7, 7, wt)


def exit_if(pg, pp):
    """@Pg EXIT Pp: exit when Pg and Pp are both true (nvcc 12.8 encoding)."""
    return _instr(0x094d | (pg << 12), pp << 23, 5, 1, 7, 7, 0)


def cs2r_globaltimer(rd):
    """CS2R Rd, SR_GLOBALTIMERLO (64 bit, ns, the same on every SM): SR id 0x52."""
    instr = (0x7805 | (rd << 16) | (((0x52 << 8) | (0x1 << 16)) << 64)).to_bytes(16, 'little')
    return ctrl.write_control(instr, stall=2, yield_=1, wr_bar=7, rd_bar=7, wt_bar=0)


def ldg_ca(rd, ra, imm):
    """LDG.E.STRONG.SM Rd, [Ra.64 + imm] (L1 allocating, as nvcc 12.8 encodes ld.global.ca):
    opcode 0x981, writes scoreboard 2."""
    assert 0 <= imm < (1 << 23)
    return _instr(0x7981 | (rd << 16) | (ra << 24) | (imm << 40), 0xc1eb900, 1, 1, 2, 7, 0)


def lop3(rd, ra, rb, rc, lut, stall, pu=None):
    """LOP3.LUT Rd, Ra, Rb, Rc, lut (nvcc 12.8: opcode 0x212; Rc bits 71:64, lut bits 79:72). With pu: LOP3.LUT Pu, RZ, ...
    (predicate = result != 0, Pu in bits 83:81; nvcc's form for a compare of an OR with zero)."""
    hi = 0x078e0000 | (lut << 8) | rc
    if pu is not None:
        rd = 255; hi = 0x07800000 | (lut << 8) | rc | ((pu << 1) << 16)
    return _instr(0x7212 | (rd << 16) | (ra << 24) | (rb << 32), hi, stall, 0, 7, 7, 0)


def lop3_abs_imm(pg, rd, ra, imm, stall):
    """@Pg LOP3.LUT Rd, Ra, imm, RZ, 0xc0 (Ra & imm; nvcc 12.8 opcode 0x812, immediate in bits 63:32)."""
    return _instr(0x0812 | (pg << 12) | (rd << 16) | (ra << 24) | (imm << 32), 0x078ec0ff, stall, 0, 7, 7, 0)


def mov_reg(rd, rs, stall, pg=7):
    """@Pg MOV Rd, Rs (nvcc 12.8: opcode 0x202, source in bits 39:32; hi bits as nvcc emits)."""
    return _instr(0x0202 | (pg << 12) | (rd << 16) | (rs << 32), 0x0f00, stall, 0, 7, 7, 0)


def bssy(b, off_bytes):
    """BSSY Bb, target (nvcc 12.8: opcode 0x945, convergence barrier register in bits 19:16, signed byte offset from the next
    instruction in bits 63:32; the target is the instruction after the matching BSYNC)."""
    return _instr(0x7945 | (b << 16) | ((off_bytes & 0xffffffff) << 32), 0x3800000, 2, 1, 7, 7, 0)


def bsync(b):
    """BSYNC Bb (nvcc 12.8: opcode 0x941, barrier register in bits 19:16)."""
    return _instr(0x7941 | (b << 16), 0x3800000, 5, 1, 7, 7, 0)


def bra_back_neg(pg, off_bytes, stall=5):
    """@!Pg BRA, target (q20 spin wait): bra_back with the guard predicate negated (bit 15)."""
    return _instr(0x0947 | (pg << 12) | (1 << 15) | ((off_bytes & 0xffffffff) << 32), 0x03ffff | (7 << 23), stall, 1, 7, 7, 0)


def bra_fwd(pg, pp, off_bytes, stall=5):
    """@Pg BRA Pp, target for a FORWARD (positive) byte offset: as bra_back, but the sign extension bits above bit 63 are 0 (bra_back sets them for a negative offset)."""
    assert 0 <= off_bytes < (1 << 31)
    return _instr(0x0947 | (pg << 12) | (off_bytes << 32), (pp << 23), stall, 1, 7, 7, 0)


def bra_back(pg, pp, off_bytes, stall=5):
    """@Pg BRA Pp, target (nvcc 12.8: opcode 0x947, signed byte offset from the next instruction in bits 63:32, second
    predicate Pp in bits 89:87)."""
    return _instr(0x0947 | (pg << 12) | ((off_bytes & 0xffffffff) << 32), 0x03ffff | (pp << 23), stall, 1, 7, 7, 0)


def ffma(rd, ra, rb, rc):
    """FFMA Rd, Ra, Rb, Rc (nvcc 12.8 register form: opcode 0x223, Rd bits 23:16, Ra 31:24, Rb 39:32, Rc 71:64; byte for byte the form of nvcc's own FFMA, checked against a compiled kernel), stall 1."""
    return _instr(0x7223 | (rd << 16) | (ra << 24) | (rb << 32), rc, 1, 1, 7, 7, 0)


THRASH_PTR = 26                  # --thrash: logical slots 26, 27 = this lane's thrash buffer address
def ownfence_words(binpath):
    """q22: the 6 words (lo, hi per instruction) of MEMBAR.ALL.GPU, ERRBAR, CCTL.IVALL as nvcc compiled them in the host's sib_fence kernel, read with cuobjdump at build time (no instruction words stored in the repo)."""
    import re, subprocess
    out = subprocess.run(['cuobjdump', '-sass', '-fun', 'sib_fence', binpath], capture_output=True, text=True).stdout.splitlines()
    words = []
    for k, ln in enumerate(out):
        for mn in ('MEMBAR.ALL.GPU', 'ERRBAR', 'CCTL.IVALL'):
            if re.search(r'\b' + re.escape(mn) + r'\b', ln) and len(words) < 6 and len(words) // 2 == ('MEMBAR.ALL.GPU', 'ERRBAR', 'CCTL.IVALL').index(mn):
                lo = int(re.findall(r'/\* (0x[0-9a-f]+) \*/', ln)[-1], 16); hi = int(re.findall(r'/\* (0x[0-9a-f]+) \*/', out[k + 1])[-1], 16); words += [lo, hi]
    assert len(words) == 6, 'ownfence: MEMBAR.ALL.GPU, ERRBAR, CCTL.IVALL not found in sib_fence of %s' % binpath
    return words
SM_A, SM_B, SM_ID = 18, 19, 20   # --smfilter: run only on SM A or SM B (VIRTUALSMID into R20)
GT_START, GT_END = 22, 24       # --gtime: global timer before the start clock and after the end clock
CLK_START, CLK_END = 84, 86   # R84:R85 before TTUOPEN, R86:R87 after the group's results


def _sections(data, b):
    import struct
    shoff, = struct.unpack_from('<Q', data, b + 0x28)
    shes, shn, shstr = struct.unpack_from('<HHH', data, b + 0x3a)
    stro = b + struct.unpack_from('<Q', data, b + shoff + shstr * shes + 0x18)[0]
    out = {}
    for k in range(shn):
        h = b + shoff + k * shes
        nm = struct.unpack_from('<I', data, h)[0]
        name = bytes(data[stro + nm:data.index(b'\0', stro + nm)]).decode('latin1')
        out[name] = h
    return out


def regcount_fields(data, secname):
    """(sh_info register count, [EIATTR_REGCOUNT values]) for the kernel in every cubin."""
    import struct
    res = []; i = 0
    while (i := data.find(b'\x7fELF\x02\x01\x01', i)) >= 0:
        b = i; i += 4
        try:
            secs = _sections(data, b)
        except (struct.error, ValueError):
            continue
        if secname not in secs:
            continue
        h = secs[secname]
        info, = struct.unpack_from('<I', data, h + 0x2c)
        sym = info & 0xffffff
        attrs = []
        if '.nv.info' in secs:
            hi = secs['.nv.info']
            off, size = struct.unpack_from('<QQ', data, hi + 0x18)
            j = b + off
            while j + 4 <= b + off + size:
                fmt, attr, ln = data[j], data[j + 1], struct.unpack_from('<H', data, j + 2)[0]
                if fmt == 4:
                    if attr == 0x2f and ln == 8 and struct.unpack_from('<I', data, j + 4)[0] == sym:
                        attrs.append((j + 8, struct.unpack_from('<I', data, j + 8)[0]))
                    j += 4 + ln
                else:
                    j += 4
        res.append((h, info >> 24, attrs))
    return res


def set_regcount(data, secname, regs):
    """Set a kernel's register count in every embedded cubin: the top byte of the
    .text.<kernel> sh_info and the EIATTR_REGCOUNT (0x2f) entry for its symbol in
    .nv.info (the driver uses the latter). Returns the old values."""
    import struct
    old = []
    for h, sh_regs, attrs in regcount_fields(data, secname):
        assert len(attrs) == 1, 'expected one EIATTR_REGCOUNT entry, found %d' % len(attrs)
        info, = struct.unpack_from('<I', data, h + 0x2c)
        struct.pack_into('<I', data, h + 0x2c, (info & 0xffffff) | (regs << 24))
        struct.pack_into('<I', data, attrs[0][0], regs)
        old.append((sh_regs, attrs[0][1]))
    return old


def ttuccttl_bytes():
    """TTUCCTL.IVALL as the EoSS SM89 encoder builds it (class ttucctl_, ivall bit 80, control as the other TTU instructions: stall 1, yield, no scoreboards)."""
    import struct
    from eoss.enc.generator import Encoder
    ins = Encoder(sm='89').build('ttucctl_', {'ftz': 1}, opex=17)
    assert ins.valid, ins.failed_conditions
    assert struct.unpack('<QQ', ins.bytes)[0] & 0xfff == 0x3d5
    return ins.bytes


def build(stage, binpath, dump, out, clock=False, sync=False, regs=None, max_out=95, preclk=False, barnops=1,
          twice=False, compact_out=False, smfilter=False, gtime=False, remap=False, thrash=0, sr133=False, dualgo=False, sleepb=0, alu=0, slim=False, loop=0, outer=0, stride=0, lean=False, hoist=False, coal=False, lean_ld=False, lean_st=False, cctl=False, ias=False, prewarm=0, l1t=0, l1t_nofirst=False, sectors=1, bypass=False, decoy=False, bypass_first=False, sibling=False, ownfence=False, sibccctl=False, ownccctl=False, delay=0, stream=0, lclock=False, aluwarps=False, aluiters=600, aludelay=0, ttuwarp=0, aluunroll=12, same_record=False, sm=89):
    global INPUTS, GROUP_LEN
    d = open(dump, 'rb').read()
    if sm == 120:
        assert not (ias or slim or loop or outer or dualgo or sr133 or thrash or hoist or coal or sibling or aluwarps), '--sm 120: plain, --remap, --smfilter, --gtime, --sleepb, --clock, --sync, --twice, --compact-out builds only (the S2R, CS2R and NOP encodings carry over: SR ids 0x43 and 0x52 checked against nvcc sm_120 output)'
        at = lambda o: d[SM120_DUMP_BASE + o:SM120_DUMP_BASE + o + 16]
        group = [at(SM120_GROUP_AT + 16 * k) for k in range(GROUP_LEN)]
        isetp = at(SM120_ISETP_AT)
        assert low12(group[0]) == 0x3d0 and (W(group[0]) >> 64) & 0x100 and low12(group[1]) == 0x9d4 and (W(group[1]) >> 40) & 0x1f == GROUP_LEN - 2, 'SM120 group not at expected place'
        assert low12(isetp) == 0x80c and (W(isetp) >> 24) & 0xff == SM120_ISETP_REG and (W(isetp) >> 32) & 0xffffffff == 0xc0000
        isetp = field(isetp, 24, 90)                  # compare R90 (= 0) instead of R6: P0 true
        INPUTS = sorted({90} | {r + k for b in group if low12(b) == 0x3d1 for r in ((W(b) >> 32) & 0xff, (W(b) >> 64) & 0xff) if r != 255 for k in (0, 1)})
    elif ias:
        assert not (slim or loop or outer or dualgo or sr133 or smfilter or thrash or hoist or coal or sleepb), '--ias: plain builds only (stage 0 and the checked shape; no slim, loop, outer, smfilter ...)'
        INPUTS, GROUP_LEN = IAS_INPUTS, IAS_GROUP_LEN
        group = [d[IAS_GROUP_AT + 16 * k:IAS_GROUP_AT + 16 * k + 16] for k in range(GROUP_LEN)]
        isetp = d[IAS_ISETP_AT:IAS_ISETP_AT + 16]
        assert low12(group[0]) == 0x3d0 and low12(group[1]) == 0x9d4 and (W(group[1]) >> 40) & 0x1f == GROUP_LEN - 2, 'IAS group not at expected place'
        assert low12(isetp) == 0x80c and (W(isetp) >> 24) & 0xff == 24 and (W(isetp) >> 32) & 0xffffffff == 1
        isetp = field(isetp, 24, 90)                  # compare R90 (= 0) instead of R24: P0 true, the 0x200 path of a fresh query
    else:
        at = lambda o: d[DUMP_BASE + o:DUMP_BASE + o + 16]
        group = [at(GROUP_AT + 16 * k) for k in range(GROUP_LEN)]
        isetp = at(ISETP_AT)
        assert low12(group[0]) == 0x3d0 and low12(group[1]) == 0x9d4, 'group not at expected place'
        assert low12(isetp) == 0x80c and (W(isetp) >> 24) & 0xff == 46
        isetp = field(isetp, 24, 90)                  # compare R90 instead of R46

    # --remap: move the group's 18 register pairs to R8..R43 (pair aligned, in order of first
    # use), P0's operand to R44, clocks to R46..R49, input pointer to R6:R7. Loads and stores
    # keep their logical slots (slot k = logical register k), so the host does not change.
    P = {}
    if remap:
        assert not preclk, 'remap: preclk not supported'
        assert compact_out, 'remap needs --compact-out (only remapped registers are stored)'
        pairs = []
        for b in group:
            op = low12(b)
            if op == 0x3d1:   used = [(W(b) >> 32) & 0xff, (W(b) >> 64) & 0xff]   # TTUST Rb, Rc
            elif op == 0x3d2: used = [(W(b) >> 16) & 0xff, (W(b) >> 64) & 0xff]   # TTULD Rd, Rc (Rd2)
            else:             used = []
            for r in used:
                if r != 255 and r not in pairs:
                    assert r % 2 == 0, 'pair base must be even: R%d' % r
                    pairs.append(r)
        nxt = 8
        for r in pairs:
            P[r], P[r + 1] = nxt, nxt + 1; nxt += 2
        assert nxt <= 44, nxt
        P[90] = 44
        P[ALU_REG] = 45
        P[84], P[85], P[86], P[87] = 46, 47, 48, 49
        P[SM_A], P[SM_B], P[SM_ID] = 50, 51, 52                       # --smfilter
        if thrash or prewarm or l1t: P[THRASH_PTR], P[THRASH_PTR + 1] = 58, 59                # --thrash buffer pointer
        P[GT_START], P[GT_START + 1], P[GT_END], P[GT_END + 1] = 54, 55, 56, 57   # --gtime
        def remap_ttu(b):
            op = low12(b)
            m = lambda r: 255 if r == 255 else P[r]          # RZ stays RZ
            if op == 0x3d1:
                b = field(field(b, 32, m((W(b) >> 32) & 0xff)), 64, m((W(b) >> 64) & 0xff))
            elif op == 0x3d2:
                b = field(field(b, 16, m((W(b) >> 16) & 0xff)), 64, m((W(b) >> 64) & 0xff))
            return b
        group = [remap_ttu(b) for b in group]
        isetp = field(isetp, 24, P[90])
    phys = lambda r: P.get(r, r)
    in_ptr = (2 if sm == 120 else 6) if remap else IN_PTR   # SM120 template: both IMAD.WIDE.U32 read the index in R7, so the remapped input pointer cannot be R6:R7 (found on the first remap run, 2026-11-19: nothing was stored); R2:R3 is free in the remap layout

    # --hoist / --coal (ablations of the persistent kernel, default off: every other build is unchanged).
    # Extra registers: at --regs > 72 from R72 up (pair and quad aligned), else from the free low registers (the declared
    # count must stay >= highest register + 3, so R69 is the highest usable one at 72).
    HZERO = [2, 3, 32, 33, 44, 45]                       # constant words that are zero (fill 'zero'): restored with a MOV from RZ
    HCOPY = [34, 35, 58, 59, 60, 61, 62, 63, 68, 69, 70, 71]   # constant words the group's TTULDs overwrite: loaded once, restored from copies
    HKEEP = [54, 55, 90]                                 # constant words nothing overwrites: loaded once into their own registers
    CO = {}
    if hoist or coal:
        assert slim and loop and outer and remap, '--hoist/--coal: need the persistent loop kernel (--slim --remap --loop --outer)'
        assert not lean and not smfilter, '--hoist/--coal: not with --lean or --smfilter'
    if hoist and coal and not (regs and regs > 72):
        raise AssertionError('--hoist --coal needs --regs 96 (20 extra registers)')
    if hoist or coal:
        hi = regs and regs > 72
        if hi:
            CO.update(Q=72, O2=76, OG=78, R16=79)
            CO['copy'] = list(range(80, 80 + len(HCOPY)))
        else:
            CO.update(Q=48, O2=46, OG=2, R16=3)
            CO['copy'] = [69, 59, 57, 56, 55, 54, 52, 51, 50, 49, 48, 47][:len(HCOPY)]
            assert not (hoist and coal)
        if coal: CO['copy'] = CO['copy'] if hoist else []
    COPY = dict(zip(HCOPY, CO.get('copy', []))) if hoist else {}
    assert len(set(COPY.values())) == len(COPY)

    p = BinaryPatcher(binpath)
    sec = [s for s in p.sections if s[2].endswith('.ttu_one')]
    assert len(sec) == 1
    ins = list(elf.iter_instructions(p.data, sec[0]))
    orig = [b for _, b in ins]
    imm = lambda b: (W(b) >> 40) & 0xffffff
    ldg = {imm(b) // 4: b for b in orig if low12(b) == 0x981}
    stg = {imm(b) // 4: b for b in orig if low12(b) == 0x986}
    wide = [b for b in orig if low12(b) in (0x625, 0x825)]     # IMAD.WIDE.U32: in then out pointer (SM89 0x625, SM120 immediate form 0x825)
    assert sorted(ldg) == list(range(96)) and sorted(stg) == list(range(96)), 'template layout changed'
    assert len(wide) == 2 and (W(wide[0]) >> 16) & 0xff == 2 and (W(wide[1]) >> 16) & 0xff == 4
    first_wide = orig.index(wide[0])
    prologue = orig[:first_wide]
    assert all(low12(b) != 0x981 for b in prologue)
    pt = lambda b: ((W(b) >> 12) & 7) == 7 and ((W(b) >> 15) & 1) == 0
    ex = [i for i, b in enumerate(orig) if low12(b) == 0x94d and pt(b)]        # unpredicated EXIT only
    assert ex

    body = list(prologue)
    CP = 60 if remap else 94
    OW, OC, OS, OWI = 0, 53, 58, 1     # --outer: R0 = 4 (word to byte factor), R53 = outer rounds left, R58 = stride in words, R1 = word index of the lane's current record
    outer_at = None
    if outer:
        assert loop and slim and remap and 1 <= outer <= 1 << 20 and ((0 < stride < 1 << 32) or (same_record and stride == 0)) and not sync, '--outer: needs --loop (slim, remap), no --sync, --stride words per round (or --same-record with stride 0: every round reuses the lane\'s own record, Phase 10i)'
        body.append(ctrl.write_control(ctrl.NOP_BYTES, stall=15))   # R4 = gid x 96 comes from an IMAD in the nvcc prologue (fast forwarding to the next IMAD only): wait for it
        body.append(mov_reg(OWI, 4, 6))
        if coal: body.append(mov_reg(CO['OG'], 0, 6))                  # global lane id (R0 is overwritten next)
        body.append(iadd3_imm(OW, 255, 4, 6))
        body.append(iadd3_imm(OC, 255, outer, 6))
        body.append(iadd3_imm(OS, 255, stride, 6))
        if coal: body.append(iadd3_imm(CO['R16'], 255, 16, 6))       # byte factor of the 16 byte records
        if hoist:   # constants loaded once, before the outer loop: the 12 the group overwrites into spare copy registers, the 3 nothing overwrites into their own
            body.append(imov_cbank(CP, 0x160, 11))
            body.append(imov_cbank(CP + 1, 0x164, 11))
            for r in HCOPY: body.append(field(field(ldg[r], 16, COPY[r]), 24, CP))
            for r in HKEEP: body.append(field(field(ldg[r], 16, phys(r)), 24, CP))
            body.append(ctrl.write_control(ctrl.NOP_BYTES, wt_bar=0x3f, stall=2))
        outer_at = len(body)
        body.append(ctrl.write_control(field(field(field(wide[0], 16, in_ptr), 24, OWI), 64, OW), stall=6))
        body.append(ctrl.write_control(field(field(field(wide[1], 16, 4), 24, OWI), 64, OW), stall=8))
        if coal: body.append(ctrl.write_control(field(field(field(wide[1], 16, CO['O2']), 24, CO['OG']), 64, CO['R16']), stall=8))   # out + 16 x lane record index
    else:
        body.append(field(wide[0], 16, in_ptr))
        body.append(wide[1])
    if slim and not hoist:                       # --slim: pointer pair to the shared record (R94:R95 plain, R60:R61 remapped; set before the outer loop with --hoist)
        assert not (clock or gtime or preclk or sr133 or twice or thrash or dualgo or sleepb), '--slim: no timing and no second session (SM filter allowed)'
        body.append(imov_cbank(CP, 0x160, 11))
        body.append(imov_cbank(CP + 1, 0x164, 11))
    # --loop K (Throughput_7 fair comparison): the group runs again, from a loop branch, for every lane whose query returned
    # pending entries, up to K rounds, exactly as OptiX's own traversal loop rebuilds the next query (tmax = best t, 0x180 and
    # 0x190 from the previous TTULD 0x3C0 / 0x3D0). Extra registers (remap layout only): best hit R62..R66, counter R67, temp R68.
    LB3, LB32, LB33, LB44, LB45, LCNT, LTMP = 62, 63, 64, 65, 66, 67, 68
    RELOAD = [30, 31, 34, 35, 56, 57, 58, 59, 60, 61, 62, 63, 68, 69]    # input words the TTULDs overwrite (and R56..R61 reused as pending)
    if loop:
        assert slim and remap and not (clock or gtime or preclk or sr133 or twice or thrash or dualgo or sleepb or alu), '--loop: slim + remap only, no timing (SM filter allowed)'
        assert 1 <= loop <= 64
    def reload_ld(r):
        if r in RAYSLOT:
            if lean or lean_ld: return ldg_ca(phys(r), in_ptr, 4 * RAYSLOT[r])
            return field(field(field(ldg[r], 16, phys(r)), 24, in_ptr), 40, 4 * RAYSLOT[r], 24)
        return field(field(ldg[r], 16, phys(r)), 24, CP)
    bssy_at = None
    if outer:   # the lanes of a warp leave the resume loop at different rounds: reconverge after it (as nvcc does around a divergent loop) so the next ray starts with a full warp
        bssy_at = len(body); body.append(NOP_PLACEHOLDER)
    def warm_batch(n, bypass=bypass):   # n pages: per lane n x `sectors` ordinary L1 allocating loads (page k = 4096 bytes apart, sector s = 32 bytes apart: all four sectors of a 128 byte line with --sectors 4), results to junk registers, then wait
        j = 0
        for k in range(n):
            for s in range(sectors):
                body.append(field(field(field(ldg[32], 16, 8 + (j % 32)), 24, phys(THRASH_PTR)), 40, 4096 * k + 32 * s, 24) if bypass else ldg_ca(8 + (j % 32), phys(THRASH_PTR), 4096 * k + 32 * s)); j += 1
                if j % 32 == 0:
                    body.append(ctrl.write_control(ctrl.NOP_BYTES, wt_bar=0x3f, stall=2))
        if j % 32:
            body.append(ctrl.write_control(ctrl.NOP_BYTES, wt_bar=0x3f, stall=2))
    if prewarm:   # --prewarm N: before anything else (inputs are loaded after, over the junk registers)
        assert remap and not ((twice and not decoy) or slim or loop or outer), '--prewarm: single session plain remap builds (or --twice --decoy)'
        for r in (THRASH_PTR, THRASH_PTR + 1):
            body.append(field(field(ldg[r], 16, phys(r)), 24, in_ptr))
        body.append(ctrl.write_control(ctrl.NOP_BYTES, wt_bar=0x3f, stall=2))
        if bypass_first:   # q18: first the same lines with non L1 allocating loads (they fill L2), then with L1 allocating loads (they now fill L1 from L2)
            assert not bypass, '--bypass-first is for an L1 allocating --prewarm'
            warm_batch(prewarm, True)
        warm_batch(prewarm)
    for r in INPUTS:
        if hoist and r in HKEEP:
            continue                                                     # loaded once before the outer loop
        if hoist and r in HCOPY:
            body.append(mov_reg(phys(r), COPY[r], 1)); continue          # restored from its copy
        if hoist and r in HZERO:
            body.append(mov_reg(phys(r), 255, 1)); continue              # a zero word
        if slim and r in RAYSLOT:
            body.append(ldg_ca(phys(r), in_ptr, 4 * RAYSLOT[r]) if (lean or lean_ld) else field(field(field(ldg[r], 16, phys(r)), 24, in_ptr), 40, 4 * RAYSLOT[r], 24))
        elif slim:
            body.append(field(field(ldg[r], 16, phys(r)), 24, CP))
        else:
            body.append(field(field(ldg[r], 16, phys(r)), 24, in_ptr))
    if decoy:   # q18: the FIRST session runs on a different tiny GAS: R34:R35 (0x100 instance node) and R70:R71 (0x180 root) come from record words 36..39; the second (timed) session reloads every input below
        assert twice and remap and not (cctl or l1t or thrash or slim or loop or outer or ias or smfilter), '--decoy: --twice --remap (plain, optional --prewarm)'
        for src, dst in ((36, 34), (37, 35), (38, 70), (39, 71)):
            body.append(field(field(ldg[src], 16, phys(dst)), 24, in_ptr))
    if smfilter:
        for r in (SM_A, SM_B):
            body.append(field(field(ldg[r], 16, phys(r)), 24, in_ptr))
    if loop:
        body.append(mov_reg(LB3, 255, 2))                      # no best hit yet
        body.append(iadd3_imm(LCNT, 255, loop, 2))             # rounds left
    if aluwarps:   # Sched_2: every warp but warp 0 of the block (threadIdx.x >= 32) leaves for the ALU path after the main EXIT; uniform per warp, before any TTU instruction and before the barrier (the ALU path has its own BAR.SYNC 0)
        assert sync and clock and remap and compact_out and not (smfilter or twice or sibling or loop or outer or alu or slim or thrash or dualgo or ias or sleepb or gtime or preclk), '--aluwarps: --sync --clock --remap --compact-out, plain'
        body.append(s2r_sr(phys(ALU_REG), 0x21))                                   # S2R R45, SR_TID.X (scoreboard 2)
        body.append(ctrl.write_control(lop3_abs_imm(7, phys(ALU_REG), phys(ALU_REG), 0xffffffe0, 6), wt_bar=0x4))   # R45 = tid & ~31
        if ttuwarp:   # the TTU warp is warp `ttuwarp`, every other warp is an ALU warp: R45 ^= ttuwarp * 32 (R46 is the start clock register, written later)
            body.append(iadd3_imm(phys(CLK_START), 255, 32 * ttuwarp, 6))
            body.append(lop3(phys(ALU_REG), phys(ALU_REG), phys(CLK_START), 255, 0x3c, 6))
        body.append(lop3(0, phys(ALU_REG), 255, 255, 0xfc, 13, pu=2))              # P2 = this warp is an ALU warp
        abra = len(body); body.append(bra_fwd(2, 7, 0))                           # @P2 BRA alu_path (offset patched below)
    body.append(ctrl.write_control(ctrl.NOP_BYTES, wt_bar=0x3f, stall=2))   # all loads done
    if smfilter:   # leave unless this block's SM is A or B; before any TTU instruction or barrier
        body.append(s2r_virtualsmid(phys(SM_ID)))
        body.append(isetp_ne(1, phys(SM_ID), phys(SM_A), 2, 0x4))
        body.append(isetp_ne(0, phys(SM_ID), phys(SM_B), 13, 0x4))
        body.append(exit_if(0, 1))
        if sibccctl:   # q22 (h): the block on SM B (P1 = SM is not A) jumps to its own path after the main code: spin on flag1, one TTUCCTL.IVALL, store flag2. No session on that block.
            assert sibling and twice and remap and clock and not (ownfence or cctl or l1t or thrash or prewarm or decoy or slim or loop or outer or ias), '--sibccctl: --smfilter --twice --remap --clock --sibling, plain'
            sibccctl_bra = len(body)
            body.append(bra_fwd(1, 7, 0))   # offset patched below
    body.append(isetp)
    body.append(ctrl.write_control(ctrl.NOP_BYTES, stall=15))               # P0 ready
    if sleepb:   # Memory_5 timing: the block on SM B (P1 = SM is not A, from the SM filter) sleeps before its trace
        assert smfilter and 0 < sleepb < 1000000, '--sleepb needs --smfilter, 0 < ns < 1 ms'
        body.append(nanosleep_p1(sleepb))
    if preclk:
        body.append(cs2r_clock(88))               # diagnostic: arrival at the barrier, R88:R89
    if sync:
        body.append(bar_sync0())                  # all warps of the block start together
        for _ in range(barnops):
            body.append(ctrl.write_control(ctrl.NOP_BYTES, stall=1))
    groups = []
    if twice:
        # first trace (untimed): one full session, then wait for all of its results
        groups.append(len(body))
        body += group if stage == 1 else [ctrl.NOP_BYTES] * GROUP_LEN
        body.append(ctrl.write_control(ctrl.NOP_BYTES, wt_bar=0x3f, stall=2))   # CLOSE done
        if l1t:   # --l1t N: ordinary loads of N lines per lane (4 KB apart), optional TTUCCTL, then the same loads timed (clock window = the reload only; session 2 is not timed)
            assert twice and remap and clock and not thrash, '--l1t: --twice --remap --clock'
            for r in (THRASH_PTR, THRASH_PTR + 1):
                body.append(field(field(ldg[r], 16, phys(r)), 24, in_ptr))
            body.append(ctrl.write_control(ctrl.NOP_BYTES, wt_bar=0x3f, stall=2))
            if not l1t_nofirst:
                warm_batch(l1t)
            if cctl:
                body.append(ttuccttl_bytes() if stage == 1 else ctrl.NOP_BYTES)
            else:
                body.append(ctrl.NOP_BYTES)
            body += [ctrl.write_control(ctrl.NOP_BYTES, stall=15)] * 4
            body.append(cs2r_clock(phys(CLK_START)))
            warm_batch(l1t)
            body.append(cs2r_clock(phys(CLK_END)))
            body.append(ctrl.write_control(ctrl.NOP_BYTES, wt_bar=0x3f, stall=2))
        elif cctl:   # Interface_7: TTUCCTL.IVALL between two complete sessions (never inside one); stage 0 puts a NOP in its place. Then 4 NOPs with stall 15.
            assert twice, '--cctl needs --twice'
            body.append(ttuccttl_bytes() if stage == 1 else ctrl.NOP_BYTES)
            body += [ctrl.write_control(ctrl.NOP_BYTES, stall=15)] * 4
        if sibling:   # q20: after the first session is closed: flag1 = 1 (the sibling block starts streaming), then spin on flag2 (it is done). Flags at the pointer in slots 26, 27 (+0, +128).
            assert twice and remap and clock and not (thrash or cctl or l1t or prewarm or decoy or slim or loop or outer or ias), '--sibling: --twice --remap --clock, plain'
            for r in (THRASH_PTR, THRASH_PTR + 1):
                body.append(field(field(ldg[r], 16, phys(r)), 24, in_ptr))
            body.append(ctrl.write_control(ctrl.NOP_BYTES, wt_bar=0x3f, stall=2))
            body.append(field(field(field(stg[3], 24, phys(THRASH_PTR)), 32, phys(THRASH_PTR)), 40, 0, 24))   # flag1 = low word of the flag address (nonzero)
            body.append(ctrl.write_control(ctrl.NOP_BYTES, wt_bar=0x3f, stall=2))
            if ownccctl:   # q22 (i): the TTU warp's own TTUCCTL.IVALL between the sessions (after its flag1 store, before the spin on flag2), outside any session; stage 0 a NOP; then 4 NOPs with stall 15 (as --cctl)
                assert not ownfence, '--ownccctl and --ownfence are separate arms'
                body.append(ttuccttl_bytes() if stage == 1 else ctrl.NOP_BYTES)
                body += [ctrl.write_control(ctrl.NOP_BYTES, stall=15)] * 4
            if ownfence:   # q22 (f), (g): the TTU warp's own fence.acq_rel.gpu between the sessions, outside any session: the three instructions nvcc emits for it (MEMBAR.ALL.GPU, ERRBAR, CCTL.IVALL), copied with their control words from the compiled sib_fence kernel (cuobjdump of the host binary)
                ow = ownfence_words(binpath); body += [ow[k].to_bytes(8, 'little') + ow[k + 1].to_bytes(8, 'little') for k in (0, 2, 4)]
                body.append(ctrl.write_control(ctrl.NOP_BYTES, stall=15))
            spin_top = len(body)
            body.append(field(field(field(ldg[32], 16, 8), 24, phys(THRASH_PTR)), 40, 128, 24))             # STRONG.SYS load of flag2 into junk R8
            body.append(ctrl.write_control(ctrl.NOP_BYTES, wt_bar=0x3f, stall=2))
            body.append(isetp_ne(6, 8, 255, 13, 0))                                                           # P6 = flag2 != 0
            body.append(bra_back_neg(6, (spin_top - (len(body) + 1)) * 16))                                   # @!P6 BRA spin_top
            body.append(ctrl.write_control(ctrl.NOP_BYTES, stall=15))
        if delay:   # q32 (ii): a counted ALU loop between the sessions, no memory access, no TTU instruction: the same time as --thrash without the loads (R45 is free with --remap)
            assert twice and remap and not (thrash or sibling or cctl or l1t or prewarm or decoy or slim or loop or outer or ias or gtime), '--delay: --twice --remap, plain'
            dl = phys(ALU_REG); body.append(iadd3_imm(dl, 255, delay, 2)); dtop = len(body)
            body.append(iadd3_imm(dl, dl, 0xffffffff, 6)); body.append(lop3(0, dl, 255, 255, 0xfc, 13, pu=5)); body.append(bra_back(5, 7, (dtop - (len(body) + 1)) * 16))
            body.append(ctrl.write_control(ctrl.NOP_BYTES, stall=15))
        if stream:   # q32 (ii): KB of ordinary L1 allocating loads (every sector of every line) from the TTU warp itself between the sessions: a loop of 16 pages x 4 sectors = 64 KB per round, the pointer advances by 64 KB (low word only: the host checks the buffer does not straddle a 4 GB boundary)
            assert twice and remap and sectors == 4 and stream % 64 == 0 and not (thrash or delay or sibling or cctl or l1t or prewarm or decoy or slim or loop or outer or ias or gtime), '--stream KB: --twice --remap --sectors 4, KB a multiple of 64'
            for r in (THRASH_PTR, THRASH_PTR + 1):
                body.append(field(field(ldg[r], 16, phys(r)), 24, in_ptr))
            body.append(ctrl.write_control(ctrl.NOP_BYTES, wt_bar=0x3f, stall=2))
            dl = phys(ALU_REG); body.append(iadd3_imm(dl, 255, stream // 64, 2)); stop_ = len(body)
            warm_batch(16)
            body.append(iadd3_imm(phys(THRASH_PTR), phys(THRASH_PTR), 65536, 6))
            body.append(iadd3_imm(dl, dl, 0xffffffff, 6)); body.append(lop3(0, dl, 255, 255, 0xfc, 13, pu=5)); body.append(bra_back(5, 7, (stop_ - (len(body) + 1)) * 16))
            body.append(ctrl.write_control(ctrl.NOP_BYTES, stall=15))
        if thrash:   # same warp, same SM: N L1 allocating loads per lane, N x 4 KB of lines per warp
            assert remap, '--thrash needs --remap'
            for r in (THRASH_PTR, THRASH_PTR + 1):
                body.append(field(field(ldg[r], 16, phys(r)), 24, in_ptr))
            body.append(ctrl.write_control(ctrl.NOP_BYTES, wt_bar=0x3f, stall=2))
            warm_batch(thrash)   # junk into group regs (reloaded below); with --sectors 4 every sector of the lines
        # its TTULDs overwrite input registers: load the inputs again, same ray
        for r in INPUTS:
            body.append(field(field(ldg[r], 16, phys(r)), 24, in_ptr))
        body.append(ctrl.write_control(ctrl.NOP_BYTES, wt_bar=0x3f, stall=2))   # all loads done
        body.append(isetp)
        body.append(ctrl.write_control(ctrl.NOP_BYTES, stall=15))               # P0 ready
    if sr133:
        assert not compact_out and not remap and max_out >= 89, 'sr133: R88, R89 are stored only by the plain layout'
        body.append(s2r_sr(88, 0x85))             # SR_TTU_TICKET_INFO before TTUOPEN (waited on by the NOP after the group)
    if gtime:
        body.append(cs2r_globaltimer(phys(GT_START)))
    if clock and not l1t:
        body.append(cs2r_clock(phys(CLK_START)))
    if lclock:
        assert slim and loop and remap and not (outer or clock or gtime or coal or lean or hoist or twice), '--lclock: --slim --remap --loop, no --outer, --clock, --gtime, --coal, --lean, --hoist'
        body.append(cs2r_clock(phys(CLK_START)))
    if dualgo:   # Interface_5: OPEN, FUSE 28, 10 ST, GO, the same 10 ST, GO, 6 LD (CLOSE last): one ray submitted twice
        assert stage != 1 or (low12(group[12]) == 0x3d3 and all(low12(b) == 0x3d1 for b in group[2:12])), 'group shape'
        fuse = W(group[1]) & ~(0x1f << 40) | (28 << 40)
        group = [group[0], fuse.to_bytes(16, 'little')] + group[2:13] + group[2:13] + group[13:19]
    GLEN = len(group)
    g0 = len(body)
    groups.append(g0)
    body += group if stage == 1 else [ctrl.NOP_BYTES] * GLEN
    for _ in range(alu):   # Throughput_4: plain dependent ALU work after the group's last instruction, before the wait
        body.append(iadd3_imm(phys(ALU_REG), phys(ALU_REG), 1, 15))
    body.append(ctrl.write_control(ctrl.NOP_BYTES, wt_bar=0x3f, stall=2))
    if loop:
        ph = phys
        body.append(isetp_ne(1, ph(3), 255, 6, 0))                                  # P1 = this query hit
        body.append(lop3(LTMP, ph(56), ph(57), ph(58), 0xfe, 6))                    # T = OR of the 8 pending words
        body.append(lop3(LTMP, LTMP, ph(59), ph(60), 0xfe, 6))
        body.append(lop3(LTMP, LTMP, ph(61), ph(70), 0xfe, 6))
        body.append(lop3(0, LTMP, ph(71), 255, 0xfc, 6, pu=2))                      # P2 = pending
        for lb, r in ((LB3, 3), (LB32, 32), (LB33, 33), (LB44, 44), (LB45, 45)):
            body.append(mov_reg(lb, ph(r), 1, pg=1))                                # @P1 best = this hit (later hits are never farther: tmax = best)
        body.append(iadd3_imm(LCNT, LCNT, 0xffffffff, 6))
        body.append(lop3(0, LCNT, 255, 255, 0xfc, 6, pu=3))                         # P3 = rounds left
        body.append(isetp_ne(4, LB3, 255, 6, 0))                                    # P4 = a best hit exists
        for d, sx in ((2, 70), (3, 71), (70, 56), (71, 57), (32, 58), (33, 59), (44, 60), (45, 61)):   # next query's 0x190 / 0x180 words
            body.append(mov_reg(ph(d), ph(sx), 1))
        for r in RELOAD:
            body.append(mov_reg(phys(r), COPY[r], 1) if hoist and r in COPY else reload_ld(r))
        body.append(ctrl.write_control(ctrl.NOP_BYTES, wt_bar=0x3f, stall=2))       # reloads done
        body.append(lop3_abs_imm(4, ph(17), LB45, 0x7fffffff, 4))                   # @P4 tmax = |best t| in 0x50 and 0xD0
        body.append(lop3_abs_imm(4, ph(31), LB45, 0x7fffffff, 6))
        loop_bra = len(body)
        body.append(bra_back(2, 3, (g0 - (loop_bra + 1)) * 16))                      # @P2 BRA P3, group
        if outer:
            body.append(bsync(0))
            body[bssy_at] = bssy(0, (len(body) - (bssy_at + 1)) * 16)
        for lb, r in ((LB3, 3), (LB32, 32), (LB33, 33), (LB44, 44), (LB45, 45)):
            body.append(mov_reg(ph(r), lb, 1, pg=4))                                # @P4 final result = best hit
        body.append(ctrl.write_control(ctrl.NOP_BYTES, stall=6))
    if clock and not l1t:
        body.append(cs2r_clock(phys(CLK_END)))
    if lclock:
        body.append(cs2r_clock(phys(CLK_END)))
    if gtime:
        body.append(cs2r_globaltimer(phys(GT_END)))
    if sr133:
        body.append(s2r_sr(89, 0x85))             # SR_TTU_TICKET_INFO after the group closed
        body.append(ctrl.write_control(ctrl.NOP_BYTES, wt_bar=0x3f, stall=2))
    if slim and (lean or lean_st):    # --lean / --lean-st: only hit flag (R3), primitive (R44), t (R45) and the rounds left are stored (slots 0, 1, 2, 13)
        outputs = [3, 44, 45]
    elif slim:
        outputs = list(SLIM_OUT)
    elif compact_out:   # only what the checks need: inputs (incl. TTULD destinations) and the clock
        outputs = sorted(set(INPUTS) | ({CLK_START, CLK_START + 1, CLK_END, CLK_END + 1} if clock else set())
                         | ({SM_A, SM_B, SM_ID} if smfilter else set())
                         | ({GT_START, GT_START + 1, GT_END, GT_END + 1} if gtime else set()))
    else:
        outputs = [r for r in OUTPUTS if r <= max_out]
    if coal:   # one 16 byte record per ray: hit word R3, primitive R44, t R45, rounds left; consecutive lanes write consecutive 16 byte records
        Q = CO['Q']
        for k, src in enumerate((phys(3), phys(44), phys(45), LCNT)):
            body.append(mov_reg(Q + k, src, 1))
        body.append(ctrl.write_control(ctrl.NOP_BYTES, stall=6))
        st128 = (W(field(field(stg[3], 24, CO['O2']), 32, Q)) & ~(0xffffff << 40)) | (1 << 74)    # STG.E.STRONG.SYS -> .128 (bit 74: the size field of nvcc's STG.E.128)
        body.append(B(st128))
        outputs = []
    else:
      for r in outputs:
        if slim:
            body.append(field(field(stg[r], 32, phys(r)), 40, 4 * (outputs.index(r) if (lean or lean_st) else SLIM_OUT.index(r)), 24))
        else:
            body.append(field(stg[r], 32, phys(r)))
    if loop and not coal:   # rounds left (K - rounds run) to slot 13
        body.append(field(field(stg[3], 32, LCNT), 40, 4 * 13, 24))
    if lclock:   # clock start (R46:R47) and end (R48:R49) to slots 16..19
        for k, r in enumerate((CLK_START, CLK_START + 1, CLK_END, CLK_END + 1)):
            body.append(field(field(stg[3], 32, phys(r)), 40, 4 * (16 + k), 24))
    if outer and os.environ.get('OUTER_DBG'):   # diagnostic: slot 13 = R1 (record word index), slot 12 = R0
        body.append(field(field(stg[3], 32, 1), 40, 4 * 13, 24)); body.append(field(field(stg[3], 32, 0), 40, 4 * 12, 24))
    if outer:  # next record of this lane: wait until every store has read its registers (read barrier 5), then loop
        n_st = 1 if coal else len(outputs) + 1
        n_st += 2 if os.environ.get('OUTER_DBG') else 0
        for k in range(n_st):
            body[-1 - k] = ctrl.write_control(body[-1 - k], rd_bar=5)
        body.append(ctrl.write_control(ctrl.NOP_BYTES, wt_bar=0x3f, stall=2))
        body.append(iadd3_imm(OWI, OWI, stride, 6))
        if coal: body.append(iadd3_imm(CO['OG'], CO['OG'], stride // 96, 6))
        body.append(iadd3_imm(OC, OC, 0xffffffff, 6))
        body.append(lop3(0, OC, 255, 255, 0xfc, 13, pu=5))                        # P5 = rounds left
        body.append(bra_back(5, 7, (outer_at - (len(body) + 1)) * 16))             # @P5 BRA L0
    assert low12(orig[ex[0] + 1]) == 0x947, 'expected BRA self after EXIT'
    body += [orig[ex[0]], orig[ex[0] + 1]]            # EXIT, BRA self
    bpath = None
    if sibccctl:
        bpath = len(body)
        for r in (THRASH_PTR, THRASH_PTR + 1):
            body.append(field(field(ldg[r], 16, phys(r)), 24, in_ptr))
        body.append(ctrl.write_control(ctrl.NOP_BYTES, wt_bar=0x3f, stall=2))
        spin_top = len(body)
        body.append(field(field(field(ldg[32], 16, 8), 24, phys(THRASH_PTR)), 40, 0, 24))                  # STRONG.SYS load of flag1 into junk R8
        body.append(ctrl.write_control(ctrl.NOP_BYTES, wt_bar=0x3f, stall=2))
        body.append(isetp_ne(6, 8, 255, 13, 0))                                                           # P6 = flag1 != 0
        body.append(bra_back_neg(6, (spin_top - (len(body) + 1)) * 16))                                   # @!P6 BRA spin_top
        body.append(ctrl.write_control(ctrl.NOP_BYTES, stall=15))
        body.append(ttuccttl_bytes() if stage == 1 else ctrl.NOP_BYTES)                                   # the only TTU instruction of this block; stage 0 has a NOP
        body += [ctrl.write_control(ctrl.NOP_BYTES, stall=15)] * 4
        body.append(ctrl.write_control(ctrl.NOP_BYTES, wt_bar=0x3f, stall=2))
        body.append(field(field(field(stg[3], 24, phys(THRASH_PTR)), 32, phys(THRASH_PTR)), 40, 128, 24)) # flag2 = nonzero
        body.append(ctrl.write_control(ctrl.NOP_BYTES, wt_bar=0x3f, stall=2))
        body += [orig[ex[0]], orig[ex[0] + 1]]
        body[sibccctl_bra] = bra_fwd(1, 7, (bpath - (sibccctl_bra + 1)) * 16)
    apath = None
    if aluwarps:   # the ALU path (after the main EXIT and BRA self): own BAR.SYNC 0, clock, 12 independent FFMA x aluiters in a counted loop, clock, store the clocks, EXIT
        apath = len(body)
        body.append(ctrl.write_control(ctrl.NOP_BYTES, wt_bar=0x3f, stall=2))
        body.append(bar_sync0())
        for _ in range(barnops): body.append(ctrl.write_control(ctrl.NOP_BYTES, stall=1))
        if aludelay:   # let the TTU warp issue its group first: a counted loop of dependent integer instructions (about 25 cycles per iteration, few issue slots)
            body.append(iadd3_imm(19, 255, aludelay, 6)); dtop_ = len(body)
            body.append(iadd3_imm(19, 19, 0xffffffff, 6)); body.append(lop3(0, 19, 255, 255, 0xfc, 13, pu=4)); body.append(bra_back(4, 7, (dtop_ - (len(body) + 1)) * 16))
            body.append(ctrl.write_control(ctrl.NOP_BYTES, stall=15))
        body.append(cs2r_clock(phys(CLK_START)))
        for r in range(8, 16): body.append(iadd3_imm(r, 255, 0, 2))               # accumulators R8..R15 = 0
        body.append(iadd3_imm(16, 255, 0x3f800000, 2)); body.append(iadd3_imm(17, 255, 0x3f800000, 2))   # Rb = Rc = 1.0f
        body.append(iadd3_imm(18, 255, aluiters, 6))                              # counter
        atop = len(body)
        for k in range(aluunroll): body.append(ffma(8 + k % 8, 8 + k % 8, 17 if k % 2 == 0 else 16, 255))   # Rd = Ra * Rb + RZ; Ra and Rb in different register banks (even Ra with odd Rb and the reverse): no bank conflict
        body.append(iadd3_imm(18, 18, 0xffffffff, 6))
        body.append(lop3(0, 18, 255, 255, 0xfc, 13, pu=5))                        # P5 = counter != 0
        body.append(bra_back(5, 7, (atop - (len(body) + 1)) * 16))
        body.append(ctrl.write_control(ctrl.NOP_BYTES, stall=15))
        body.append(cs2r_clock(phys(CLK_END)))
        body.append(ctrl.write_control(ctrl.NOP_BYTES, wt_bar=0x3f, stall=2))
        for r in (CLK_START, CLK_START + 1, CLK_END, CLK_END + 1): body.append(field(stg[r], 32, phys(r)))
        body.append(ctrl.write_control(ctrl.NOP_BYTES, wt_bar=0x3f, stall=2))
        body += [orig[ex[0]], orig[ex[0] + 1]]
        body[abra] = bra_fwd(2, 7, (apath - (abra + 1)) * 16)
    assert len(body) <= len(orig), (len(body), len(orig))
    body += [ctrl.NOP_BYTES] * (len(orig) - len(body))
    for (off, _), b in zip(ins, body):
        p.data[off:off + 16] = b
    if regs:
        top = max([phys(r) for r in set(INPUTS) | set(outputs)] + [in_ptr + 1, 5] + ([CP + 1] if slim else []) + ([LCNT + 1] if loop else []) + ([CO['Q'] + 3, CO['O2'] + 1, CO['OG'], CO['R16']] if coal else []) + list(COPY.values()))
        assert regs % 8 == 0 and regs >= top + 3 and regs <= 255, 'declared count must be >= highest register + 3 (R%d)' % top
        old = set_regcount(p.data, sec[0][2], regs)
        assert old, 'register count field not found'
        print('register count %s -> %d' % (sorted(set(old)), regs))
    p.save(out)
    sha = hashlib.sha256(open(out, 'rb').read()).hexdigest()
    top_reg = max([phys(r) for r in set(INPUTS) | set(outputs)] + [in_ptr + 1, 5] + ([LCNT + 1] if loop else []))
    meta = {'stage': stage, 'remap': remap, 'regmap': P, 'top_reg': top_reg, 'clk': [phys(CLK_START), phys(CLK_END)], 'gt': [phys(GT_START), phys(GT_END)], 'twice': twice, 'sibling': sibling, 'ownfence': ownfence, 'ownccctl': ownccctl, 'delay': delay, 'stream': stream, 'sibccctl': sibccctl, 'sibccctl_bra': sibccctl_bra if sibccctl else None, 'bpath': bpath, 'thrash': thrash, 'sr133': sr133, 'dualgo': dualgo, 'sleepb': sleepb, 'alu': alu, 'slim': slim, 'loop': loop, 'group_len': GLEN, 'smfilter': smfilter, 'gtime': gtime, 'groups': groups, 'clock': clock, 'sync': sync, 'barnops': barnops if sync else 0, 'preclk': preclk, 'regs': regs, 'sha256': sha, 'section': sec[0][2], 'group_index': g0, 'lean': lean, 'hoist': hoist, 'coal': coal, 'outer': outer, 'outer_at': outer_at, 'stride': stride, 'same_record': same_record, 'cctl': cctl, 'ias': ias, 'lclock': lclock, 'l1t': l1t, 'prewarm': prewarm, 'sectors': sectors, 'bypass': bypass, 'decoy': decoy, 'bypass_first': bypass_first,
            'sm': sm, 'inputs': INPUTS, 'outputs': outputs, 'in_ptr': in_ptr}
    meta.update(aluwarps=aluwarps, aluiters=aluiters, aluunroll=aluunroll, aludelay=aludelay, ttuwarp=ttuwarp, abra=abra if aluwarps else None, apath=apath)
    json.dump(meta, open(out + '.layout.json', 'w'), indent=1)
    used = g0 + GROUP_LEN + 1 + int(clock) + len(outputs) + 2
    print('built %s stage %d: %d of %d slots used, group at index %d, sha256 %s' % (out, stage, used, len(orig), g0, sha))
    return out


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument('--stage', type=int, choices=[0, 1], required=True)
    a.add_argument('--bin', default='/dev/shm/setup3/setup3')
    a.add_argument('--dump', default=os.path.expanduser('~/eoss/notes/setup4/sm89_pipeline_fnaddr-0x14000.bin'))
    a.add_argument('--out')
    a.add_argument('--sm', type=int, default=89, choices=[89, 120], help='120: the SM120 GAS group from the SM120 dump (pc14, plain builds only); the dump default then is notes/setup4_sm120/dump_sm120_fn-0x20000_len0x20000.bin')
    a.add_argument('--clock', action='store_true', help='clock reads around the group')
    a.add_argument('--sync', action='store_true', help='BAR.SYNC before the group (with --clock: before the start read)')
    a.add_argument('--regs', type=int, help='declared register count of ttu_one (>= 96)')
    a.add_argument('--max-out', type=int, default=95, help='store registers only up to this one')
    a.add_argument('--ias', action='store_true', help='BVH_4: the IAS traversal group (F91, 17 instructions) from an IAS pipeline dump (--dump notes/q3/ias/ias_a.bin) instead of the GAS group; plain builds only; the host needs --ias')
    a.add_argument('--ownfence', action='store_true', help='q22 (with --sibling): the TTU warp itself executes MEMBAR.ALL.GPU, ERRBAR, CCTL.IVALL between the sessions (before its spin on flag2)')
    a.add_argument('--ownccctl', action='store_true', help='q22 arm (i) (with --sibling): the TTU warp itself executes one TTUCCTL.IVALL between the sessions (before its spin on flag2); stage 0 a NOP')
    a.add_argument('--sibccctl', action='store_true', help='q22 arm (h), PREPARED NOT RUN (with --sibling --smfilter): the block on SM B has no session; it spins on flag1 and executes one TTUCCTL.IVALL, then stores flag2')
    a.add_argument('--delay', type=int, default=0, help='q32 (ii) (with --twice --remap): N iterations of a counted ALU loop between the sessions (no loads), the control for --thrash')
    a.add_argument('--stream', type=int, default=0, help='q32 (ii) (with --twice --remap --sectors 4): KB of ordinary loads (all sectors) from the TTU warp between the sessions, as a loop (multiple of 64)')
    a.add_argument('--sibling', action='store_true', help='q20 (with --twice --remap --clock): between the sessions store flag1 and spin on flag2 (host --sibling SM:KB:PASSES runs a plain block on another SM)')
    a.add_argument('--cctl', action='store_true', help='Interface_7 (with --twice): TTUCCTL.IVALL between the two sessions (stage 1 only; stage 0 has a NOP there)')
    a.add_argument('--twice', action='store_true', help='two full sessions with the same ray; time the second')
    a.add_argument('--compact-out', action='store_true', help='store only input and clock registers')
    a.add_argument('--aluwarps', action='store_true', help='Sched_2 (with --sync --clock --remap --compact-out): warps 1.. of the block (tid >= 32) run an ALU loop of independent FFMA instead of the TTU code; warp 0 is the TTU warp')
    a.add_argument('--aludelay', type=int, default=0, help='--aluwarps: delay loop iterations (about 25 cycles each) before the ALU warps start their clock and loop')
    a.add_argument('--ttuwarp', type=int, default=0, help='--aluwarps: the TTU warp is this warp of the block (default 0); all others are ALU warps')
    a.add_argument('--aluunroll', type=int, default=12, help='--aluwarps: FFMA per loop iteration (multiple of 8); a long unrolled body keeps the ALU warp ready (few loop instructions)')
    a.add_argument('--aluiters', type=int, default=600, help='--aluwarps: loop iterations (12 FFMA + 3 loop instructions each)')
    a.add_argument('--smfilter', action='store_true', help='blocks not on SM A or B (words 18, 19) exit before any TTU code')
    a.add_argument('--prewarm', type=int, default=0, help='N ordinary L1 allocating loads per lane (4 KB apart from slots 26, 27) before the first session; host --thrash --thrash-gas for BVH lines')
    a.add_argument('--l1t', type=int, default=0, help='with --twice --remap --clock: N loads per lane, [TTUCCTL with --cctl], the same N loads timed; session 2 untimed')
    a.add_argument('--l1t-nofirst', action='store_true', help='--l1t without the first loads (the timed loads find L1 cold)')
    a.add_argument('--sectors', type=int, default=1, choices=[1, 2, 4], help='--prewarm / --l1t / --thrash: loads per line (32 byte sectors, 4 = the whole 128 byte line; a 4 byte load fetches one sector only)')
    a.add_argument('--decoy', action='store_true', help='q18 (with --twice --remap): the first session uses a second one triangle GAS (host --decoy: record words 36..39 into R34, R35, R70, R71), the timed second session the target GAS')
    a.add_argument('--bypass-first', action='store_true', help='q18 (with --prewarm, not --bypass): the prewarm loads twice, first non L1 allocating (L2 fill), then L1 allocating')
    a.add_argument('--bypass', action='store_true', help='--prewarm / --l1t / --thrash: use the template (non L1 allocating, STRONG.SYS) load instead of the L1 allocating one: warms L2 and the TLB, not L1')
    a.add_argument('--thrash', type=int, default=0, help='with --twice --remap: N L1 allocating loads per lane between the two sessions (N x 4 KB per warp)')
    a.add_argument('--slim', action='store_true', help='Throughput_6/7: ray words from one sector, other inputs broadcast from a shared record, 13 result words stored, no clock or image stores')
    a.add_argument('--loop', type=int, default=0, help='Throughput_7 fair: resume loop in the kernel, up to K rounds (needs --slim --remap)')
    a.add_argument('--outer', type=int, default=0, help='persistent kernel: each lane runs OUTER records (record k of lane g = k x STRIDE words + g x 96), needs --loop')
    a.add_argument('--same-record', action='store_true', help='with --outer and --stride 0: every round of a lane reads and writes the SAME record (small output, Phase 10i short query pair); the host then needs no --outer')
    a.add_argument('--stride', type=int, default=0, help='with --outer: words between a lane\'s records = lanes in the grid x 96')
    a.add_argument('--lean', action='store_true', help='--slim: ray word loads L1 allocating (LDG.CA), only 3 result words + rounds left stored (less harness traffic)')
    a.add_argument('--lean-ld', action='store_true', help='only the L1 allocating ray loads of --lean (stores as the default build)')
    a.add_argument('--lean-st', action='store_true', help='only the 3 result words + rounds left of --lean (loads as the default build; host needs --lean)')
    a.add_argument('--hoist', action='store_true', help='ablation: constant input words loaded once before the outer loop (needs --loop --outer)')
    a.add_argument('--coal', action='store_true', help='ablation: one coalesced 16 byte STG.128 record per ray (host --coal)')
    a.add_argument('--lclock', action='store_true', help='HG_1 (with --slim --remap --loop, no --outer): clock64 before the first group and after the resume loop ended, stored to slots 16, 17 (start) and 18, 19 (end) of the lane record (host --lclock)')
    a.add_argument('--alu', type=int, default=0, help='Throughput_4: N dependent IADD3 (stall 15) after the group, before the wait')
    a.add_argument('--sleepb', type=int, default=0, help='with --smfilter: @P1 NANOSLEEP NS before the start clock (the block on SM B waits)')
    a.add_argument('--sr133', action='store_true', help='Interface_5: S2R SR_TTU_TICKET_INFO into R88 before TTUOPEN and R89 after the group')
    a.add_argument('--dualgo', action='store_true', help='Interface_5: OPEN, FUSE 28, 10 ST, GO, 10 ST, GO, 6 LD (same ray twice)')
    a.add_argument('--remap', action='store_true', help='move the group to R8..R43 and our registers below R50')
    a.add_argument('--gtime', action='store_true', help='global timer reads around the timed group (R22:R23, R24:R25)')
    a.add_argument('--preclk', action='store_true', help='diagnostic clock read into R88:R89 before the barrier')
    a.add_argument('--barnops', type=int, default=8, help='NOPs after the barrier: a warp issues a few instructions past BAR.SYNC before it stalls (8 measured enough at 20 warps)')
    x = a.parse_args()
    if x.sm == 120 and x.dump.endswith('sm89_pipeline_fnaddr-0x14000.bin'): x.dump = os.path.join(HERE, '..', '..', '..', 'notes', 'setup4_sm120', 'dump_sm120_fn-0x20000_len0x20000.bin')
    build(x.stage, x.bin, x.dump, x.out or '%s_s%d%s%s%s' % (x.bin, x.stage, 'c' if x.clock else '', 'b' if x.sync else '', 'r%d' % x.regs if x.regs else '') + ('m%d' % x.max_out if x.max_out != 95 else '') + ('p' if x.preclk else '') + ('n%d' % x.barnops if x.barnops != 8 else '') + ('t' if x.twice else '') + ('k' if x.compact_out else '') + ('f' if x.smfilter else '') + ('g' if x.gtime else '') + ('x' if x.remap else '') + ('h%d' % x.thrash if x.thrash else '') + ('q' if x.sr133 else '') + ('d' if x.dualgo else '') + ('z%d' % x.sleepb if x.sleepb else '') + ('a%d' % x.alu if x.alu else '') + ('l' if x.slim else '') + ('L%d' % x.loop if x.loop else '') + ('o%ds%d' % (x.outer, x.stride) if x.outer else '') + ('N' if x.lean else '') + ('Nl' if x.lean_ld else '') + ('Ns' if x.lean_st else '') + ('H' if x.hoist else '') + ('C' if x.coal else '') + ('I' if x.ias else '') + ('W%d' % x.prewarm if x.prewarm else '') + ('T%d' % x.l1t if x.l1t else '') + ('n' if x.l1t_nofirst else '') + ('S%d' % x.sectors if x.sectors != 1 else '') + ('Y' if x.bypass else '') + ('D' if x.decoy else '') + ('B' if x.bypass_first else '') + ('F' if x.ownfence else '') + ('K' if x.sibccctl else '') + ('O' if x.ownccctl else '') + ('e%d' % x.delay if x.delay else '') + ('M%d' % x.stream if x.stream else ''),
          x.clock, x.sync, x.regs, x.max_out, x.preclk, x.barnops, x.twice, x.compact_out, x.smfilter, x.gtime, x.remap, x.thrash, x.sr133, x.dualgo, x.sleepb, x.alu, x.slim, x.loop, x.outer, x.stride, x.lean, x.hoist, x.coal, x.lean_ld, x.lean_st, x.cctl, x.ias, x.prewarm, x.l1t, x.l1t_nofirst, x.sectors, x.bypass, x.decoy, x.bypass_first, x.sibling, x.ownfence, x.sibccctl, x.ownccctl, x.delay, x.stream, x.lclock, x.aluwarps, x.aluiters, x.aludelay, x.ttuwarp, x.aluunroll, x.same_record, x.sm)
