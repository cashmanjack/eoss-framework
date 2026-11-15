"""Store words of the IAS traversal group (F91), computed from the private IAS pipeline dump by running the straight line
code in front of the group on raw instruction fields (no NVIDIA bytes are written anywhere; the output is 8 words).

    ias_words.py IAS_DUMP.bin [OUT.txt] [--flags 0xNN] [--mask 0xNN]

The code between the function start (dump offset 0xbab0 - 0x?) and the group at 0xbb60 is 47 instructions of 13 kinds. Inputs of the
first query: R31 = (rayFlags << 16) | (mask << 8) | 1 (same word as the GAS path's R37), R18:R19 = the IAS address, c[3][0x34] =
0x00011000 (checked by the host), R48..R54 and R44..R47 never reach a store that runs on the first query (they feed the predicated
0x100, 0x180, 0x190 stores; the 0x200 store is the one that runs). P0 for the group = (R24 != 1) with R24 = 0: fresh query, true.
Prints the words of store 0 (R2, R3, R48, R49) and store 0x80 (R52, R53, R48, R49), low to high, and the pointer rule of store 0x200:
(IAS + the 64 bit word at IAS + 0x20), high word + 0x10000000.
Every opcode is checked against the listing's mnemonic family by its raw bits; an unknown form stops the script.
"""
import sys
W = lambda b: int.from_bytes(b, 'little')
M32 = 0xffffffff


def run(dump, flags=0, mask=0xff, ias_addr=0x7f0000000000, off=0x700, c3_34=0x00011000):
    d = open(dump, 'rb').read()
    at0, atend = 0xbb60 - 0x350, 0xbb60          # from -0x87f0 (MOV R0, R21) to just before TTUOPEN
    R = [0] * 256; P = [False] * 8
    R[31] = (flags << 16) | (mask << 8) | 1
    R[18], R[19] = ias_addr & M32, ias_addr >> 32
    mem64 = off                                    # the 64 bit word at IAS + 0x20
    rv = lambda r: 0 if r == 255 else R[r]
    pv = lambda p: True if p == 7 else P[p]
    for o in range(at0, atend, 16):
        w = W(d[o:o + 16]); op = w & 0xfff
        pg, pn = (w >> 12) & 7, (w >> 15) & 1
        if not (pv(pg) ^ bool(pn)):
            continue
        rd, ra, rb, imm, rc = (w >> 16) & 0xff, (w >> 24) & 0xff, (w >> 32) & 0xff, (w >> 32) & M32, (w >> 64) & 0xff
        hi = w >> 64
        pdest = (hi >> 17) & 7
        sel_p = lambda: pv((w >> 87) & 7) ^ bool((w >> 90) & 1)
        if op == 0x202: R[rd] = rv(rb)                                        # MOV
        elif op == 0x381:                                                     # LDG.E.64 R2, [R18 + 0x20]
            assert rd == 2 and ra == 18 and (w >> 40) & 0xffffff == 0x20
            R[2], R[3] = mem64 & M32, mem64 >> 32
        elif op == 0xb82:                                                     # LDC.U8 R24, c[3][0x35]
            assert (imm >> 22) & 0x1f == 3 and (imm >> 6) & 0xffff == 0x35, hex(imm)
            R[rd] = (c3_34 >> 8) & 0xff
        elif op == 0x819:                                                     # SHF
            if (hi >> 8) & 0xff == 0x16: R[rd] = rv(rc) >> (imm & 31)        # SHF.R.U32.HI Rd, RZ, n, Rc
            else:
                assert (hi >> 8) & 0xff == 0x0e and ra == rc                  # SHF.L.W.U32.HI Rd, Ra, n, Ra: rotate left
                n = imm & 31; v = rv(ra); R[rd] = ((v << n) | (v >> (32 - n))) & M32
        elif op == 0x424: R[rd] = imm                                         # IMAD.U32 Rd, RZ, RZ, imm
        elif op == 0x224: R[rd] = rv(rc)                                      # IMAD.U32 Rd, RZ, RZ, Rc (a move)
        elif op == 0x812:                                                     # LOP3 / LOP32I with immediate as operand b
            lut, c = (hi >> 8) & 0xff, rv(rc)
            a = rv(ra); out = 0
            for k in range(32):
                out |= ((lut >> ((((a >> k) & 1) << 2) | (((imm >> k) & 1) << 1) | ((c >> k) & 1))) & 1) << k
            if rd == 255: P[pdest] = out != 0                                 # predicate form: result != 0
            else: R[rd] = out
        elif op == 0x807: R[rd] = rv(ra) if sel_p() else imm                  # SEL Rd, Ra, imm, Pp
        elif op == 0x207: R[rd] = rv(ra) if sel_p() else rv(rb)               # SEL Rd, Ra, Rb, Pp
        elif op == 0x210:                                                     # IADD3 Rd, [Pcarry], Ra, Rb, Rc
            s = rv(ra) + rv(rb) + rv(rc)
            R[rd] = s & M32
            if pdest != 7: P[pdest] = (rv(ra) + rv(rb)) > M32 if rc == 255 else s > M32
        elif op == 0x810: R[rd] = (rv(ra) + imm + rv(rc)) & M32               # IADD3 Rd, Ra, imm, Rc
        elif op == 0x824:                                                     # IMAD.X Rd, Ra, 1, Rc, Pcarry-in (high word of the 64 bit add)
            assert imm == 1 and (w >> 87) & 7 == 2
            R[rd] = (rv(ra) + rv(rc) + (1 if P[2] else 0)) & M32
        elif op == 0x80c:                                                     # ISETP.NE.AND Pu, ..., Ra, imm
            P[pdest] = rv(ra) != imm
        elif op == 0x805:                                                     # CS2R Rd, SRZ
            R[rd] = R[rd + 1] = 0
        elif op == 0x816:                                                     # PRMT Rd, Ra, sel, Rc
            src = rv(ra) | (rv(rc) << 32); out = 0
            for k in range(4): out |= ((src >> (8 * ((imm >> (4 * k)) & 7))) & 0xff) << (8 * k)
            R[rd] = out
        elif op == 0x945: pass                                                # BSSY: no data effect
        else:
            raise SystemExit('unknown opcode 0x%x at dump offset 0x%x' % (op, o))
    return R, P


if __name__ == '__main__':
    opt = {sys.argv[i]: sys.argv[i + 1] for i in range(1, len(sys.argv) - 1) if sys.argv[i].startswith('--')}
    a = [x for i, x in enumerate(sys.argv[1:], 1) if not x.startswith('--') and not sys.argv[i - 1].startswith('--')]
    flags, mask = int(opt.get('--flags', '0'), 0), int(opt.get('--mask', '0xff'), 0)
    R, P = run(a[0], flags, mask)
    words = [R[2], R[3], R[48], R[49], R[52], R[53], R[48], R[49]]
    print('P0 (group) = (R24 != 1) with R24 = %d: %s' % (R[24], 'true' if P[0] else 'false'))
    print('store 0    : %08x %08x %08x %08x' % tuple(words[:4]))
    print('store 0x80 : %08x %08x %08x %08x' % tuple(words[4:]))
    print('store 0x200: low pair = IAS + word at IAS+0x20, high word + 0x10000000 (R6:R7 = R42:R43 = %08x %08x for the test address); high pair R24:R25 = %08x %08x' % (R[6], R[7], R[24], R[25]))
    if len(a) > 1: open(a[1], 'w').write(' '.join('%08x' % x for x in words) + '\n')
