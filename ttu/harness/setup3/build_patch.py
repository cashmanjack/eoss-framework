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


def build(stage, binpath, dump, out, clock=False, sync=False, regs=None, max_out=95, preclk=False, barnops=1):
    d = open(dump, 'rb').read()
    at = lambda o: d[DUMP_BASE + o:DUMP_BASE + o + 16]
    group = [at(GROUP_AT + 16 * k) for k in range(GROUP_LEN)]
    isetp = at(ISETP_AT)
    assert low12(group[0]) == 0x3d0 and low12(group[1]) == 0x9d4, 'group not at expected place'
    assert low12(isetp) == 0x80c and (W(isetp) >> 24) & 0xff == 46
    isetp = field(isetp, 24, 90)                      # compare R90 instead of R46

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
    ex = [i for i, b in enumerate(orig) if low12(b) == 0x94d]
    assert ex

    body = list(prologue)
    body.append(field(wide[0], 16, IN_PTR))
    body.append(wide[1])
    for r in INPUTS:
        body.append(field(field(ldg[r], 16, r), 24, IN_PTR))
    body.append(ctrl.write_control(ctrl.NOP_BYTES, wt_bar=0x3f, stall=2))   # all loads done
    body.append(isetp)
    body.append(ctrl.write_control(ctrl.NOP_BYTES, stall=15))               # P0 ready
    if preclk:
        body.append(cs2r_clock(88))               # diagnostic: arrival at the barrier, R88:R89
    if sync:
        body.append(bar_sync0())                  # all warps of the block start together
        for _ in range(barnops):
            body.append(ctrl.write_control(ctrl.NOP_BYTES, stall=1))
    if clock:
        body.append(cs2r_clock(CLK_START))
    g0 = len(body)
    body += group if stage == 1 else [ctrl.NOP_BYTES] * GROUP_LEN
    body.append(ctrl.write_control(ctrl.NOP_BYTES, wt_bar=0x3f, stall=2))
    if clock:
        body.append(cs2r_clock(CLK_END))
    outputs = [r for r in OUTPUTS if r <= max_out]
    for r in outputs:
        body.append(field(stg[r], 32, r))
    body += [orig[ex[0]], orig[ex[0] + 1]]            # EXIT, BRA self
    assert len(body) <= len(orig), (len(body), len(orig))
    body += [ctrl.NOP_BYTES] * (len(orig) - len(body))
    for (off, _), b in zip(ins, body):
        p.data[off:off + 16] = b
    if regs:
        assert regs % 8 == 0 and 96 <= regs <= 255, 'body uses up to R95'
        old = set_regcount(p.data, sec[0][2], regs)
        assert old, 'register count field not found'
        print('register count %s -> %d' % (sorted(set(old)), regs))
    p.save(out)
    sha = hashlib.sha256(open(out, 'rb').read()).hexdigest()
    meta = {'stage': stage, 'clock': clock, 'sync': sync, 'barnops': barnops if sync else 0, 'preclk': preclk, 'regs': regs, 'sha256': sha, 'section': sec[0][2], 'group_index': g0,
            'inputs': INPUTS, 'outputs': outputs, 'in_ptr': IN_PTR}
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
    a.add_argument('--preclk', action='store_true', help='diagnostic clock read into R88:R89 before the barrier')
    a.add_argument('--barnops', type=int, default=8, help='NOPs after the barrier: a warp issues a few instructions past BAR.SYNC before it stalls (8 measured enough at 20 warps)')
    x = a.parse_args()
    build(x.stage, x.bin, x.dump, x.out or '%s_s%d%s%s%s' % (x.bin, x.stage, 'c' if x.clock else '', 'b' if x.sync else '', 'r%d' % x.regs if x.regs else '') + ('m%d' % x.max_out if x.max_out != 95 else '') + ('p' if x.preclk else '') + ('n%d' % x.barnops if x.barnops != 8 else ''), x.clock, x.sync, x.regs, x.max_out, x.preclk, x.barnops)
