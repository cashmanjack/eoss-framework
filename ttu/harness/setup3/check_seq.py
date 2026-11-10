"""Checklist parts A and B (notes/SETUP3_CHECKLIST.md) on a patched setup3 binary.

    check_seq.py BIN        exit 0 and writes BIN.checked (its sha256) only if all pass

Stage 0 binaries must contain no TTU instruction at all. Stage 1 binaries must
contain exactly one fused group laid out as the SM89 rules require.
"""
import hashlib, json, sys, os

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
sys.path.insert(0, HERE)
from eoss.binary.patcher import BinaryPatcher
from eoss.binary import elf
from eoss.isa.model import ISASpec
from eoss.tui.decode import Decoder

OPEN, FUSE, ST, GO, LD = 0x3d0, 0x9d4, 0x3d1, 0x3d3, 0x3d2   # SM89: TTUGO 0x3d3, TTULD 0x3d2
TTU = {OPEN, FUSE, ST, GO, LD}
W = lambda b: int.from_bytes(b, 'little')
low12 = lambda b: W(b) & 0xfff
bit = lambda b, n: (W(b) >> n) & 1
pred = lambda b: ((W(b) >> 12) & 7, (W(b) >> 15) & 1)   # (Pg, not)


def check(path):
    meta = json.load(open(path + '.layout.json'))
    data = open(path, 'rb').read()
    sha = hashlib.sha256(data).hexdigest()
    res = []
    ok = lambda name, cond, info='': res.append((name, bool(cond), info))
    ok('sha256 matches build', sha == meta['sha256'], sha)
    p = BinaryPatcher(path)
    # every .text section of every cubin: TTU opcodes only inside ttu_one's group
    hits = []
    for s in p.sections:
        for off, b in elf.iter_instructions(p.data, s):
            if low12(b) in TTU:
                hits.append((s[2], off, b))
    sec = [s for s in p.sections if s[2] == meta['section']]
    ok('one ttu_one section', len(sec) == 1)
    body = [b for _, b in elf.iter_instructions(p.data, sec[0])]
    dec = Decoder(ISASpec.load(sm=89))
    ex = next(i for i, b in enumerate(body) if low12(b) == 0x94d)
    fails = [i for i in range(ex) if dec.decode(0, body[i]).n_failed]
    ok('A1 body decodes with 0 failed checks up to EXIT', not fails, fails[:5])
    g = meta['group_index']
    if meta['stage'] == 0:
        ok('stage 0: no TTU instruction anywhere', not hits, len(hits))
    else:
        ok('A2/A9 TTU only in ttu_one', all(h[0] == meta['section'] for h in hits))
        opens = [i for i, b in enumerate(body) if low12(b) == OPEN]
        ok('A2 exactly one TTUOPEN, at group start', opens == [g], opens)
        n = (W(body[g + 1]) >> 40) & 0x1f
        ok('A3 TTUMACROFUSE right after TTUOPEN', low12(body[g + 1]) == FUSE, n)
        grp = body[g + 2:g + 2 + n]
        ok('A4 next N are all TTU (ST, GO, LD)', len(grp) == n and all(low12(b) in (ST, GO, LD) for b in grp), n)
        ok('A5 instruction after the group is not TTU', low12(body[g + 2 + n]) not in TTU)
        kinds = [low12(b) for b in grp]
        ok('A4 group is 10 ST, then 1 GO, then 6 LD', kinds == [ST] * 10 + [GO] + [LD] * 6, kinds)
        lds = [i for i, b in enumerate(grp) if low12(b) == LD]
        close = [i for i in lds if bit(grp[i], 74)]
        ok('A6 exactly one TTULD with bit 74, and it is last', close == [n - 1] and lds[-1] == n - 1, close)
        ok('A7 CLOSE unpredicated', pred(grp[-1]) == (7, 0), pred(grp[-1]))
        ok('A8 bit 72 only on TTUOPEN', not any(bit(b, 72) for b in grp), [bit(b, 72) for b in grp])
        ok('A6 no other TTU anywhere in the binary', len(hits) == n + 2, len(hits))
        ok('A11 no branch or BSSY/BSYNC between OPEN and CLOSE',
           not any(low12(b) in (0x947, 0x945, 0x941, 0x94d, 0x950, 0x943, 0x949) for b in body[g:g + 2 + n]))
        ok('B15 no branch anywhere before EXIT', not any(low12(b) in (0x947, 0x949, 0x943) for b in body[:ex]))
    wt = lambda b: (W(b[8:]) >> 52) & 0x3f
    stall = lambda b: (W(b[8:]) >> 41) & 0xf
    nb = meta.get('barnops', 1) if meta.get('sync') else 0
    c = (1 if meta.get('clock') else 0) + ((1 + nb) if meta.get('sync') else 0) + (1 if meta.get('preclk') else 0)
    is_clk = lambda b, rd: low12(b) == 0x805 and (W(b) >> 16) & 0xff == rd and (W(b) >> 72) & 0xff == 0x50
    ok('B12 loads waited on before ISETP P0', low12(body[g - 3 - c]) == 0x918 and wt(body[g - 3 - c]) == 0x3f)
    ok('B12 ISETP P0 then a NOP with stall 15 before the group',
       low12(body[g - 2 - c]) == 0x80c and low12(body[g - 1 - c]) == 0x918 and stall(body[g - 1 - c]) == 15)
    ok('B13 NOP after the group waits on all scoreboards', low12(body[g + 19]) == 0x918 and wt(body[g + 19]) == 0x3f)
    if meta.get('clock'):
        ok('clock: CS2R R84 SR_CLOCKLO right before TTUOPEN', is_clk(body[g - 1], 84))
        ok('clock: CS2R R86 SR_CLOCKLO right after the wait NOP', is_clk(body[g + 20], 86))
    if meta.get('sync'):
        k = 1 if meta.get('clock') else 0
        b = body[g - 1 - k - nb]
        ok('sync: NOPs between BAR.SYNC and the clock read / group', all(low12(body[g - 1 - k - i]) == 0x918 for i in range(nb)))
        ok('sync: blocking BAR.SYNC 0 (bit 80 clear) before the clock read / group',
           low12(b) == 0xb1d and ((W(b) >> 80) & 1) == 0 and ((W(b) >> 77) & 7) == 0)
    regs = int(next(l for l in os.popen('cuobjdump --dump-resource-usage %s' % path).read().split('Function ttu_one:')[1].split('\n') if 'REG:' in l).split('REG:')[1].split()[0])
    ok('B14 register count above highest register used (95)', regs > 95, regs)
    if meta.get('regs'):
        from build_patch import regcount_fields
        f = regcount_fields(bytes(p.data), meta['section'])
        vals = [(sh, [v for _, v in at]) for _, sh, at in f]
        ok('B14 sh_info and EIATTR_REGCOUNT both equal the requested count',
           f and all(sh == meta['regs'] and at == [meta['regs']] for sh, at in vals), vals)
    ok('B14 no stack (template did not spill into what we run)', 'STACK:0' in os.popen('cuobjdump --dump-resource-usage %s' % path).read().split('Function ttu_one:')[1].split('\n')[1], '')
    for name, good, info in res:
        print('%-55s %s %s' % (name, 'PASS' if good else 'FAIL', info))
    allok = all(r[1] for r in res)
    if allok:
        open(path + '.checked', 'w').write(sha)
    print('ALL PASS' if allok else 'FAILED', sha)
    return allok


if __name__ == '__main__':
    sys.exit(0 if check(sys.argv[1]) else 1)
