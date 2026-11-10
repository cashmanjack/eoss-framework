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


def build(stage, binpath, dump, out):
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
    g0 = len(body)
    body += group if stage == 1 else [ctrl.NOP_BYTES] * GROUP_LEN
    body.append(ctrl.write_control(ctrl.NOP_BYTES, wt_bar=0x3f, stall=2))
    for r in OUTPUTS:
        body.append(field(stg[r], 32, r))
    body += [orig[ex[0]], orig[ex[0] + 1]]            # EXIT, BRA self
    assert len(body) <= len(orig), (len(body), len(orig))
    body += [ctrl.NOP_BYTES] * (len(orig) - len(body))
    for (off, _), b in zip(ins, body):
        p.data[off:off + 16] = b
    p.save(out)
    sha = hashlib.sha256(open(out, 'rb').read()).hexdigest()
    meta = {'stage': stage, 'sha256': sha, 'section': sec[0][2], 'group_index': g0,
            'inputs': INPUTS, 'outputs': OUTPUTS, 'in_ptr': IN_PTR}
    json.dump(meta, open(out + '.layout.json', 'w'), indent=1)
    used = g0 + GROUP_LEN + 1 + len(OUTPUTS) + 2
    print('built %s stage %d: %d of %d slots used, group at index %d, sha256 %s' % (out, stage, used, len(orig), g0, sha))
    return out


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument('--stage', type=int, choices=[0, 1], required=True)
    a.add_argument('--bin', default='/dev/shm/setup3/setup3')
    a.add_argument('--dump', default=os.path.expanduser('~/eoss/notes/setup4/sm89_pipeline_fnaddr-0x14000.bin'))
    a.add_argument('--out')
    x = a.parse_args()
    build(x.stage, x.bin, x.dump, x.out or '%s_s%d' % (x.bin, x.stage))
