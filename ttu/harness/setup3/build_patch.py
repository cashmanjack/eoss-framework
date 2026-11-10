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
IN_PTR = 92
ALU_REG = 91                  # --alu: accumulator (plain layout R91; with --remap it is R45, free in that layout)

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


THRASH_PTR = 26                  # --thrash: logical slots 26, 27 = this lane's thrash buffer address
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


def build(stage, binpath, dump, out, clock=False, sync=False, regs=None, max_out=95, preclk=False, barnops=1,
          twice=False, compact_out=False, smfilter=False, gtime=False, remap=False, thrash=0, sr133=False, dualgo=False, sleepb=0, alu=0, slim=False):
    d = open(dump, 'rb').read()
    at = lambda o: d[DUMP_BASE + o:DUMP_BASE + o + 16]
    group = [at(GROUP_AT + 16 * k) for k in range(GROUP_LEN)]
    isetp = at(ISETP_AT)
    assert low12(group[0]) == 0x3d0 and low12(group[1]) == 0x9d4, 'group not at expected place'
    assert low12(isetp) == 0x80c and (W(isetp) >> 24) & 0xff == 46
    isetp = field(isetp, 24, 90)                      # compare R90 instead of R46

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
        if thrash: P[THRASH_PTR], P[THRASH_PTR + 1] = 58, 59                # --thrash buffer pointer
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
    in_ptr = 6 if remap else IN_PTR

    p = BinaryPatcher(binpath)
    sec = [s for s in p.sections if s[2].endswith('.ttu_one')]
    assert len(sec) == 1
    ins = list(elf.iter_instructions(p.data, sec[0]))
    orig = [b for _, b in ins]
    imm = lambda b: (W(b) >> 40) & 0xffffff
    ldg = {imm(b) // 4: b for b in orig if low12(b) == 0x981}
    stg = {imm(b) // 4: b for b in orig if low12(b) == 0x986}
    wide = [b for b in orig if low12(b) == 0x625]     # IMAD.WIDE.U32: in then out pointer
    assert sorted(ldg) == list(range(96)) and sorted(stg) == list(range(96)), 'template layout changed'
    assert len(wide) == 2 and (W(wide[0]) >> 16) & 0xff == 2 and (W(wide[1]) >> 16) & 0xff == 4
    first_wide = orig.index(wide[0])
    prologue = orig[:first_wide]
    assert all(low12(b) != 0x981 for b in prologue)
    pt = lambda b: ((W(b) >> 12) & 7) == 7 and ((W(b) >> 15) & 1) == 0
    ex = [i for i, b in enumerate(orig) if low12(b) == 0x94d and pt(b)]        # unpredicated EXIT only
    assert ex

    body = list(prologue)
    body.append(field(wide[0], 16, in_ptr))
    body.append(wide[1])
    CP = 60 if remap else 94                     # --slim: pointer pair to the shared record (R94:R95 plain, R60:R61 remapped)
    if slim:
        assert not (clock or gtime or preclk or sr133 or twice or thrash or dualgo or sleepb), '--slim: no timing and no second session'
        body.append(imov_cbank(CP, 0x160, 11))
        body.append(imov_cbank(CP + 1, 0x164, 11))
    for r in INPUTS:
        if slim and r in RAYSLOT:
            body.append(field(field(field(ldg[r], 16, phys(r)), 24, in_ptr), 40, 4 * RAYSLOT[r], 24))
        elif slim:
            body.append(field(field(ldg[r], 16, phys(r)), 24, CP))
        else:
            body.append(field(field(ldg[r], 16, phys(r)), 24, in_ptr))
    if smfilter:
        for r in (SM_A, SM_B):
            body.append(field(field(ldg[r], 16, phys(r)), 24, in_ptr))
    body.append(ctrl.write_control(ctrl.NOP_BYTES, wt_bar=0x3f, stall=2))   # all loads done
    if smfilter:   # leave unless this block's SM is A or B; before any TTU instruction or barrier
        body.append(s2r_virtualsmid(phys(SM_ID)))
        body.append(isetp_ne(1, phys(SM_ID), phys(SM_A), 2, 0x4))
        body.append(isetp_ne(0, phys(SM_ID), phys(SM_B), 13, 0x4))
        body.append(exit_if(0, 1))
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
        if thrash:   # same warp, same SM: N L1 allocating loads per lane, N x 4 KB of lines per warp
            assert remap, '--thrash needs --remap'
            for r in (THRASH_PTR, THRASH_PTR + 1):
                body.append(field(field(ldg[r], 16, phys(r)), 24, in_ptr))
            body.append(ctrl.write_control(ctrl.NOP_BYTES, wt_bar=0x3f, stall=2))
            for k in range(thrash):
                body.append(ldg_ca(8 + (k % 32), phys(THRASH_PTR), 4096 * k))   # junk into group regs (reloaded below)
                if k % 32 == 31 or k == thrash - 1:
                    body.append(ctrl.write_control(ctrl.NOP_BYTES, wt_bar=0x3f, stall=2))
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
    if clock:
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
    if clock:
        body.append(cs2r_clock(phys(CLK_END)))
    if gtime:
        body.append(cs2r_globaltimer(phys(GT_END)))
    if sr133:
        body.append(s2r_sr(89, 0x85))             # SR_TTU_TICKET_INFO after the group closed
        body.append(ctrl.write_control(ctrl.NOP_BYTES, wt_bar=0x3f, stall=2))
    if slim:
        outputs = list(SLIM_OUT)
    elif compact_out:   # only what the checks need: inputs (incl. TTULD destinations) and the clock
        outputs = sorted(set(INPUTS) | ({CLK_START, CLK_START + 1, CLK_END, CLK_END + 1} if clock else set())
                         | ({SM_A, SM_B, SM_ID} if smfilter else set())
                         | ({GT_START, GT_START + 1, GT_END, GT_END + 1} if gtime else set()))
    else:
        outputs = [r for r in OUTPUTS if r <= max_out]
    for r in outputs:
        if slim:
            body.append(field(field(stg[r], 32, phys(r)), 40, 4 * SLIM_OUT.index(r), 24))
        else:
            body.append(field(stg[r], 32, phys(r)))
    assert low12(orig[ex[0] + 1]) == 0x947, 'expected BRA self after EXIT'
    body += [orig[ex[0]], orig[ex[0] + 1]]            # EXIT, BRA self
    assert len(body) <= len(orig), (len(body), len(orig))
    body += [ctrl.NOP_BYTES] * (len(orig) - len(body))
    for (off, _), b in zip(ins, body):
        p.data[off:off + 16] = b
    if regs:
        top = max([phys(r) for r in set(INPUTS) | set(outputs)] + [in_ptr + 1, 5] + ([CP + 1] if slim else []))
        assert regs % 8 == 0 and regs >= top + 3 and regs <= 255, 'declared count must be >= highest register + 3 (R%d)' % top
        old = set_regcount(p.data, sec[0][2], regs)
        assert old, 'register count field not found'
        print('register count %s -> %d' % (sorted(set(old)), regs))
    p.save(out)
    sha = hashlib.sha256(open(out, 'rb').read()).hexdigest()
    top_reg = max([phys(r) for r in set(INPUTS) | set(outputs)] + [in_ptr + 1, 5])
    meta = {'stage': stage, 'remap': remap, 'regmap': P, 'top_reg': top_reg, 'clk': [phys(CLK_START), phys(CLK_END)], 'gt': [phys(GT_START), phys(GT_END)], 'twice': twice, 'thrash': thrash, 'sr133': sr133, 'dualgo': dualgo, 'sleepb': sleepb, 'alu': alu, 'slim': slim, 'group_len': GLEN, 'smfilter': smfilter, 'gtime': gtime, 'groups': groups, 'clock': clock, 'sync': sync, 'barnops': barnops if sync else 0, 'preclk': preclk, 'regs': regs, 'sha256': sha, 'section': sec[0][2], 'group_index': g0,
            'inputs': INPUTS, 'outputs': outputs, 'in_ptr': in_ptr}
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
    a.add_argument('--clock', action='store_true', help='clock reads around the group')
    a.add_argument('--sync', action='store_true', help='BAR.SYNC before the group (with --clock: before the start read)')
    a.add_argument('--regs', type=int, help='declared register count of ttu_one (>= 96)')
    a.add_argument('--max-out', type=int, default=95, help='store registers only up to this one')
    a.add_argument('--twice', action='store_true', help='two full sessions with the same ray; time the second')
    a.add_argument('--compact-out', action='store_true', help='store only input and clock registers')
    a.add_argument('--smfilter', action='store_true', help='blocks not on SM A or B (words 18, 19) exit before any TTU code')
    a.add_argument('--thrash', type=int, default=0, help='with --twice --remap: N L1 allocating loads per lane between the two sessions (N x 4 KB per warp)')
    a.add_argument('--slim', action='store_true', help='Throughput_6/7: ray words from one sector, other inputs broadcast from a shared record, 13 result words stored, no clock or image stores')
    a.add_argument('--alu', type=int, default=0, help='Throughput_4: N dependent IADD3 (stall 15) after the group, before the wait')
    a.add_argument('--sleepb', type=int, default=0, help='with --smfilter: @P1 NANOSLEEP NS before the start clock (the block on SM B waits)')
    a.add_argument('--sr133', action='store_true', help='Interface_5: S2R SR_TTU_TICKET_INFO into R88 before TTUOPEN and R89 after the group')
    a.add_argument('--dualgo', action='store_true', help='Interface_5: OPEN, FUSE 28, 10 ST, GO, 10 ST, GO, 6 LD (same ray twice)')
    a.add_argument('--remap', action='store_true', help='move the group to R8..R43 and our registers below R50')
    a.add_argument('--gtime', action='store_true', help='global timer reads around the timed group (R22:R23, R24:R25)')
    a.add_argument('--preclk', action='store_true', help='diagnostic clock read into R88:R89 before the barrier')
    a.add_argument('--barnops', type=int, default=8, help='NOPs after the barrier: a warp issues a few instructions past BAR.SYNC before it stalls (8 measured enough at 20 warps)')
    x = a.parse_args()
    build(x.stage, x.bin, x.dump, x.out or '%s_s%d%s%s%s' % (x.bin, x.stage, 'c' if x.clock else '', 'b' if x.sync else '', 'r%d' % x.regs if x.regs else '') + ('m%d' % x.max_out if x.max_out != 95 else '') + ('p' if x.preclk else '') + ('n%d' % x.barnops if x.barnops != 8 else '') + ('t' if x.twice else '') + ('k' if x.compact_out else '') + ('f' if x.smfilter else '') + ('g' if x.gtime else '') + ('x' if x.remap else '') + ('h%d' % x.thrash if x.thrash else '') + ('q' if x.sr133 else '') + ('d' if x.dualgo else '') + ('z%d' % x.sleepb if x.sleepb else '') + ('a%d' % x.alu if x.alu else '') + ('l' if x.slim else ''),
          x.clock, x.sync, x.regs, x.max_out, x.preclk, x.barnops, x.twice, x.compact_out, x.smfilter, x.gtime, x.remap, x.thrash, x.sr133, x.dualgo, x.sleepb, x.alu, x.slim)
