"""Build the rehosted RTNN style radius search kernel: ttu_one's body replaced by a loop of TTU sessions with an inline distance test.

    build_rtnn.py --stage 0|1 --bin HOST --out OUT [--rounds K] [--inline full|none]

Every instruction comes from the EoSS encoder (ttu/rehost/emit_group.py for the group and the ISETP, ttu/rehost/asm.py for the rest) except the nvcc
prologue, the 2 pointer set up instructions and the load / store templates of the host's own ttu_one (as in ttu/harness/setup3/build_patch.py).
No dump and no OptiX code is read. Per lane (one query per lane; record = 96 words at in + 384 B x lane):
  loop: session; wait; clock; (inline test of the candidate: tag 0x50 and hit flag 1, record pointer x 8 -> primitive index (word 0) -> point
  (float4 table at R6:R7) -> d2 = fma(dz, dz, fma(dy, dy, dx*dx)) <= r2 (R8): count, store the primitive id); pending entries test; next query from the
  returned pending entries (halves swapped), constants reloaded from the record; branch back while pending entries remain and rounds are left.
Results (slots of the lane's out record): 3, 32, 33, 44, 45, 56..61, 70, 71 last answer; 6 count of neighbours, 7 sessions, 8 sum of start clocks, 9 sum of end clocks
(session window: end - start), 10 first start clock, 11 end clock after the loop, 13 rounds left; 16..79 neighbour primitive ids (first 64).
Registers: R4:R5 out pointer, R92:R93 in pointer, R6:R7 points, R8 r2 (bits), R9 count, R10 sessions, R96..R108 inline temps, R109 = 1.0f (unused), R110, R111 sums,
R112:R113, R114:R115 clocks, R116:R117 neighbour cursor, R118 rounds left, R119 first clock, R120 temp. Highest R120 (declared 128).
"""
import argparse, hashlib, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', '..')
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'ttu', 'harness', 'setup3'))
import build_patch as BP
from ttu.rehost import asm as A, emit_group as EG
from eoss.binary.patcher import BinaryPatcher
from eoss.binary import elf
from eoss.enc import control as ctrl

W, B, low12, field = BP.W, BP.B, BP.low12, BP.field
IN_PTR, OUT_PTR = 92, 4
EXTRA_IN = [6, 7, 8]
RELOAD = [30, 31, 34, 35, 56, 57, 58, 59, 60, 61, 62, 63, 68, 69]
LAST_ANS = [3, 32, 33, 44, 45, 56, 57, 58, 59, 60, 61, 70, 71]
R_CNT, R_SES, R_T0S, R_T1S, R_T0, R_T1, R_CUR, R_LEFT, R_C0, R_TMP = 9, 10, 110, 111, 112, 114, 116, 118, 119, 120
CAP = 64


def pred_bits(b, p, neg=0):
    return B(W(b) & ~(0xf << 12) | (p << 12) | (neg << 15))


TRACE_REGS = [2, 3, 30, 31, 32, 33, 34, 35, 44, 45, 56, 57, 58, 59, 60, 61, 62, 63, 68, 69, 70, 71]   # answer registers stored per session by --trace (R18:R19 = this lane's absolute trace address, 128 bytes per session)


def build(stage, binpath, out, rounds, inline='full', debug=False, trace=False):
    p = BinaryPatcher(binpath)
    sec = [s for s in p.sections if s[2].endswith('.ttu_one')]
    assert len(sec) == 1
    ins = list(elf.iter_instructions(p.data, sec[0])); orig = [b for _, b in ins]
    imm = lambda b: (W(b) >> 40) & 0xffffff
    ldg = {imm(b) // 4: b for b in orig if low12(b) == 0x981}
    stg = {imm(b) // 4: b for b in orig if low12(b) == 0x986}
    wide = [b for b in orig if low12(b) in (0x625, 0x825)]
    assert sorted(ldg) == list(range(96)) and sorted(stg) == list(range(96)) and len(wide) == 2, 'template layout changed'
    prologue = orig[:orig.index(wide[0])]
    pt = lambda b: ((W(b) >> 12) & 7) == 7 and ((W(b) >> 15) & 1) == 0
    ex = [i for i, b in enumerate(orig) if low12(b) == 0x94d and pt(b)]
    assert ex and low12(orig[ex[0] + 1]) == 0x947

    body = list(prologue)
    body.append(field(wide[0], 16, IN_PTR)); body.append(wide[1])                # in pointer -> R92:R93, out pointer -> R4:R5
    inputs = sorted(set(BP.INPUTS) | set(EXTRA_IN) | ({18, 19} if trace else set()))   # R18:R19 = the lane's trace offset (--trace)
    for r in inputs:
        body.append(field(field(ldg[r], 16, r), 24, IN_PTR))
    body.append(A.iadd3_imm(R_CNT, A.RZ, 0)); body.append(A.iadd3_imm(R_SES, A.RZ, 0))
    body.append(A.iadd3_imm(R_T0S, A.RZ, 0)); body.append(A.iadd3_imm(R_T1S, A.RZ, 0))
    body.append(A.iadd3_imm(R_LEFT, A.RZ, rounds))
    body.append(A.iadd3_imm(R_CUR, OUT_PTR, 4 * 16)); body.append(A.mov(R_CUR + 1, OUT_PTR + 1))     # neighbour cursor = out + 64 (the host checks that no 4 GB boundary is crossed)
    body.append(BP.cs2r_clock(R_T0))
    body.append(ctrl.write_control(ctrl.NOP_BYTES, stall=15))                  # the clock pair is written
    body.append(A.mov(R_C0, R_T0, stall=6))
    body.append(ctrl.write_control(ctrl.NOP_BYTES, wt_bar=0x3f, stall=2))      # loads done
    body.append(EG.isetp_p0_true(ra=90))
    top = len(body)
    body.append(ctrl.write_control(ctrl.NOP_BYTES, stall=15))                  # P0 ready; the loop branches back to this NOP (BITTU's probe sits after it)
    g = len(body)
    body += EG.group() if stage == 1 else [ctrl.NOP_BYTES] * 19
    body.append(ctrl.write_control(ctrl.NOP_BYTES, wt_bar=0x3f, stall=2))      # session closed, results here
    body.append(ctrl.write_control(BP.cs2r_clock(R_T1), stall=8))             # the clock pair needs a few cycles before it is read
    body += [A.iadd3(R_T0S, R_T0S, R_T0, stall=2), A.iadd3(R_T1S, R_T1S, R_T1, stall=2), A.iadd3_imm(R_SES, R_SES, 1, stall=2)]
    if trace:   # the answer registers of this session, before anything modifies them; then a wait until the stores have read them and R18
        for k, reg in enumerate(TRACE_REGS):
            body.append(field(field(field(stg[3], 24, 18), 32, reg), 40, 4 * k, 24))
        body += [ctrl.write_control(ctrl.NOP_BYTES, wt_bar=0x3f, stall=2), A.iadd3_imm(18, 18, 128, stall=6)]
    LEVELS = ['none', 'rec', 'pts', 'math', 'full']; lv = LEVELS.index(inline)
    P4, P5, P6 = (4, 0), (5, 0), (6, 0)
    if lv >= 1:   # decode the candidate, fetch its primitive index from the record
        body += [A.shr_u32(98, 33, 25, stall=6), A.isetp_imm(4, 'EQ', 98, 0x50, stall=13), A.isetp_reg(4, 'NE', 3, A.RZ, pp=4, stall=13),
                 A.imad_imm(96, 44, 8, stall=4), A.shl64_hi(97, 44, 3, 45, stall=6), A.ldg(98, 96, 0, pg=P4, sb=2, stall=4), ctrl.write_control(ctrl.NOP_BYTES, wt_bar=0x3f, stall=2)]
    if lv >= 2:   # the point
        body += [A.imad_imm(100, 98, 16, rc=6, pg=P4, stall=6), A.mov(101, 7, pg=P4, stall=6),
                 A.ldg(102, 100, 0, pg=P4, sb=2), A.ldg(103, 100, 4, pg=P4, sb=2), A.ldg(104, 100, 8, pg=P4, sb=2, stall=4), ctrl.write_control(ctrl.NOP_BYTES, wt_bar=0x3f, stall=2)]
    if lv >= 3:   # the distance test
        body += [A.fadd_neg(105, 102, 82, pg=P4, stall=2), A.fadd_neg(106, 103, 83, pg=P4, stall=2), A.fadd_neg(107, 104, 12, pg=P4, stall=6),
                 A.fmul(108, 105, 105, pg=P4, stall=6), A.ffma(108, 106, 106, 108, pg=P4, stall=6), A.ffma(108, 107, 107, 108, pg=P4, stall=6),
                 A.fsetp_le(5, 108, 8, pp=4, stall=13), A.isetp_imm(6, 'LT', R_CNT, CAP, pp=5, stall=13)]
    if lv >= 4:   # count and record the neighbour
        st = field(field(field(stg[3], 24, R_CUR), 32, 98), 40, 0, 24)
        body += [pred_bits(st, 6), ctrl.write_control(ctrl.NOP_BYTES, wt_bar=0x3f, stall=2), A.iadd3_imm(R_CUR, R_CUR, 4, pg=P6, stall=4), A.iadd3_imm(R_CNT, R_CNT, 1, pg=P5, stall=4)]   # the NOP waits until the store has read R116 (stand alone waits: consumer embedded waits did not hold in the R3 stage 0 runs)
    # pending entries and rounds
    body += [BP.lop3(R_TMP, 56, 57, 58, 0xfe, 6), BP.lop3(R_TMP, R_TMP, 59, 60, 0xfe, 6), BP.lop3(R_TMP, R_TMP, 61, 70, 0xfe, 6), BP.lop3(0, R_TMP, 71, 255, 0xfc, 6, pu=2),
             A.iadd3_imm(R_LEFT, R_LEFT, 0xffffffff, stall=6), A.isetp_reg(4, 'NE', R_LEFT, A.RZ, pp=2, stall=13)]     # P2 = pending, P4 = pending and rounds left
    for d, s in ((2, 70), (3, 71), (70, 56), (71, 57), (32, 58), (33, 59), (44, 60), (45, 61)):
        body.append(A.mov(d, s, pg=(4, 0), stall=1))
    # the previous answer's 0x380 low words (R62:R63) become the next query's 0x100 high pair (R54:R55), as OptiX's loop does: R54 = R62, R55 = (R55 & 0x7C000000) | (R63 & ~0x7C000000)
    body += [A.mov(54, 62, pg=(4, 0), stall=1), A.lop3_imm(55, 55, 0x7C000000, 63, 0xE2, pg=(4, 0), stall=6)]
    for r in RELOAD:
        body.append(A.ldg(r, IN_PTR, 4 * r, pg=(4, 0), sb=2, stall=4 if r == RELOAD[-1] else 1))
    body.append(ctrl.write_control(ctrl.NOP_BYTES, wt_bar=0x3f, stall=2))
    body.append(ctrl.write_control(BP.cs2r_clock(R_T0), stall=8))
    loop_bra = len(body)
    body.append(BP.bra_back(4, 7, (top - (loop_bra + 1)) * 16))                  # @P4 BRA top
    body.append(ctrl.write_control(BP.cs2r_clock(R_T1), stall=8))
    body.append(ctrl.write_control(ctrl.NOP_BYTES, stall=6))
    outs = [(s, s) for s in LAST_ANS] + [(6, R_CNT), (7, R_SES), (8, R_T0S), (9, R_T1S), (10, R_C0), (11, R_T1), (13, R_LEFT)]
    for slot, reg in outs:
        body.append(field(stg[slot], 32, reg))
    if debug:   # DEBUG ONLY (fails check_seq L4, never run on a TTU build): the inline temporaries to slots 20 ..
        for k, reg in enumerate((96, 97, 98, 100, 101, 102, 103, 104, 105, 106, 107, 108)):
            body.append(field(stg[20 + k], 32, reg))
    body += [orig[ex[0]], orig[ex[0] + 1]]
    assert len(body) <= len(orig), (len(body), len(orig))
    body += [ctrl.NOP_BYTES] * (len(orig) - len(body))
    for (off, _), b in zip(ins, body):
        p.data[off:off + 16] = b
    old = BP.set_regcount(p.data, sec[0][2], 128); assert old
    p.save(out)
    sha = hashlib.sha256(open(out, 'rb').read()).hexdigest()
    meta = {'stage': stage, 'remap': False, 'regmap': {}, 'top_reg': R_TMP, 'clk': [R_T0, R_T1], 'gt': [22, 24], 'twice': False, 'sibling': False, 'sm': 89, 'inputs': inputs,
            'outputs': sorted(set(s for s, _ in outs)), 'in_ptr': IN_PTR, 'section': sec[0][2], 'group_index': g, 'group_len': 19, 'groups': [g], 'sha256': sha, 'regs': 128,
            'emit': True, 'rtnn': 1, 'rounds': rounds, 'inline': inline, 'trace': bool(trace), 'loop_top': top, 'loop_bra': loop_bra}
    json.dump(meta, open(out + '.layout.json', 'w'), indent=1)
    print('built %s stage %d: %d instructions of %d slots, group at %d, loop top %d, bra %d, sha256 %s' % (out, stage, loop_bra + 1 + len(outs) + 5, len(orig), g, top, loop_bra, sha))
    return out


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument('--stage', type=int, choices=[0, 1], required=True); a.add_argument('--bin', required=True); a.add_argument('--out', required=True)
    a.add_argument('--rounds', type=int, default=100); a.add_argument('--debug', action='store_true'); a.add_argument('--trace', action='store_true'); a.add_argument('--inline', choices=['full', 'none', 'rec', 'pts', 'math'], default='full')
    x = a.parse_args()
    build(x.stage, x.bin, x.out, x.rounds, x.inline, x.debug, x.trace)
