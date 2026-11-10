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
    ex = next(i for i, b in enumerate(body) if low12(b) == 0x94d and pred(b) == (7, 0))   # the unpredicated EXIT
    fails = [i for i in range(ex) if dec.decode(0, body[i]).n_failed]
    ok('A1 body decodes with 0 failed checks up to EXIT', not fails, fails[:5])
    g = meta['group_index']
    groups = meta.get('groups') or [g]
    ok('layout: the timed group is the last session', groups[-1] == g, groups)
    if meta['stage'] == 0:
        ok('stage 0: no TTU instruction anywhere', not hits, len(hits))
    else:
        ok('A2/A9 TTU only in ttu_one', all(h[0] == meta['section'] for h in hits))
        opens = [i for i, b in enumerate(body) if low12(b) == OPEN]
        ok('A2 one TTUOPEN per session, each at its session start', opens == groups, (opens, groups))
        total = 0
        for si, gi in enumerate(groups):
            tag = 'session %d: ' % si
            n = (W(body[gi + 1]) >> 40) & 0x1f
            ok(tag + 'A3 TTUMACROFUSE right after TTUOPEN', low12(body[gi + 1]) == FUSE, n)
            grp = body[gi + 2:gi + 2 + n]
            ok(tag + 'A4 next N are all TTU (ST, GO, LD)', len(grp) == n and all(low12(b) in (ST, GO, LD) for b in grp), n)
            ok(tag + 'A5 instruction after the group is not TTU', low12(body[gi + 2 + n]) not in TTU)
            kinds = [low12(b) for b in grp]
            shape = [ST] * 10 + [GO] + [ST] * 10 + [GO] + [LD] * 6 if meta.get('dualgo') else [ST] * 10 + [GO] + [LD] * 6
            ok(tag + ('A4 group is 10 ST, GO, 10 ST, GO, 6 LD (dualgo)' if meta.get('dualgo') else 'A4 group is 10 ST, then 1 GO, then 6 LD'), kinds == shape, kinds)
            lds = [i for i, b in enumerate(grp) if low12(b) == LD]
            close = [i for i in lds if bit(grp[i], 74)]
            ok(tag + 'A6 exactly one TTULD with bit 74, and it is last', close == [n - 1] and lds[-1] == n - 1, close)
            ok(tag + 'A7 CLOSE unpredicated', pred(grp[-1]) == (7, 0), pred(grp[-1]))
            ok(tag + 'A8 bit 72 only on TTUOPEN', not any(bit(b, 72) for b in grp))
            # A12 (2026-11-10, Cesar): TTUST Rb, Rc and TTULD Rd, Rd2 are even aligned 64 bit register pairs on SM89
            # and SM120 (IDEST_SIZE 64; the spec requires even Rd and Rd2). Each base register must be even or RZ.
            fld = lambda b, lo: (W(b) >> lo) & 0xff
            regs = [(k, fld(b, lo)) for b in grp for k, lo in ((('ST Rb', 32), ('ST Rc', 64)) if low12(b) == ST else (('LD Rd', 16), ('LD Rd2', 64)) if low12(b) == LD else ())]
            odd = [(k, r) for k, r in regs if r != 255 and r % 2]
            ok(tag + 'A12 every TTUST/TTULD register pair base is even (or RZ)', not odd and len(regs) == 2 * sum(1 for b in grp if low12(b) in (ST, LD)), (odd, len(regs)))
            ok(tag + 'A11 no branch or BSSY/BSYNC between OPEN and CLOSE',
               not any(low12(b) in (0x947, 0x945, 0x941, 0x94d, 0x950, 0x943, 0x949) for b in body[gi:gi + 2 + n]))
            total += n + 2
            if si + 1 < len(groups):
                nxt = body[gi + 2 + n]
                ok(tag + 'next instruction after CLOSE waits on every scoreboard (session closed before the next OPEN)',
                   low12(nxt) == 0x918 and ((W(nxt[8:]) >> 52) & 0x3f) == 0x3f)
                ok(tag + 'no TTU instruction between this CLOSE and the next OPEN',
                   not any(low12(b) in TTU for b in body[gi + 2 + n:groups[si + 1]]))
        ok('A6 no other TTU anywhere in the binary', len(hits) == total, (len(hits), total))
        if meta.get('remap') and len(sys.argv) > 2:   # A10 with remap: only register fields may differ
            d = open(sys.argv[2], 'rb').read()
            dump = [d[0x14000 - 0xfb50 + 16 * k:0x14000 - 0xfb50 + 16 * k + 16] for k in range(19)]
            regmask = lambda b: W(b) & ~((0xff << 16) | (0xff << 32) | (0xff << 64)) if low12(b) in (ST, LD) else W(b)
            for gi in groups:
                same = all(regmask(a) == regmask(b) for a, b in zip(body[gi:gi + 19], dump))
                ok('A10 remapped group equals the OptiX group except register fields (session at %d)' % gi, same)
            pm = {int(k): v for k, v in meta['regmap'].items()}
            ok('A10 register map is a pair aligned one to one map', len(set(pm.values())) == len(pm) and all(pm[k] % 2 == k % 2 for k in pm))
        ok('B15 no branch anywhere before EXIT', not any(low12(b) in (0x947, 0x949, 0x943) for b in body[:ex]))
    wt = lambda b: (W(b[8:]) >> 52) & 0x3f
    stall = lambda b: (W(b[8:]) >> 41) & 0xf
    nb = meta.get('barnops', 1) if meta.get('sync') else 0
    c = (1 if meta.get('clock') else 0) + ((1 + nb) if meta.get('sync') else 0) + (1 if meta.get('preclk') else 0) + (1 if meta.get('gtime') else 0) + (1 if meta.get('sr133') else 0) + (1 if meta.get('sleepb') else 0)
    GL = meta.get('group_len', 19)
    if meta.get('sleepb'):   # @P1 NANOSLEEP right after the P0 stall NOP, before any clock read and the group
        ns = [i for i, b in enumerate(body[:ex]) if low12(b) == 0x95d]
        ok('sleepb: one @P1 NANOSLEEP, before the group', len(ns) == 1 and ns[0] < g and pred(body[ns[0]]) == (1, 0), ns)
    is_clk = lambda b, rd: low12(b) == 0x805 and (W(b) >> 16) & 0xff == rd and (W(b) >> 72) & 0xff == 0x50
    cs, ce = meta.get('clk', [84, 86])
    f4 = 4 if meta.get('smfilter') else 0   # S2R, 2 ISETP, @P0 EXIT sit between the wait and ISETP P0
    ok('B12 loads waited on before ISETP P0 (and before the SM filter)', low12(body[g - 3 - c - f4]) == 0x918 and wt(body[g - 3 - c - f4]) == 0x3f)
    ok('B12 ISETP P0 then a NOP with stall 15 before the group',
       low12(body[g - 2 - c]) == 0x80c and low12(body[g - 1 - c]) == 0x918 and stall(body[g - 1 - c]) == 15)
    ok('B13 NOP after the group waits on all scoreboards', low12(body[g + GL]) == 0x918 and wt(body[g + GL]) == 0x3f)
    if meta.get('clock'):
        ok('clock: CS2R SR_CLOCKLO right before TTUOPEN', is_clk(body[g - 1], cs), cs)
        ok('clock: CS2R SR_CLOCKLO right after the wait NOP', is_clk(body[g + GL + 1], ce), ce)
    if meta.get('sr133'):   # S2R SR_TTU_TICKET_INFO (0x85): one before the start clock, one after the end clock, both outside the group
        s2r = [i for i, b in enumerate(body[:ex]) if low12(b) == 0x919 and ((W(b) >> 72) & 0xff) == 0x85]
        ok('sr133: two S2R SR133, one before TTUOPEN and one after the group', len(s2r) == 2 and s2r[0] < g and s2r[1] > g + GL, s2r)
    pexits = [i for i, b in enumerate(body[:ex]) if low12(b) == 0x94d]
    if meta.get('smfilter'):
        first_ttu = min([i for i, b in enumerate(body) if low12(b) in TTU] or [ex])
        bars = [i for i, b in enumerate(body) if low12(b) == 0xb1d]
        ok('smfilter: one predicated EXIT, before any TTU instruction and any barrier',
           len(pexits) == 1 and pexits[0] < first_ttu and all(pexits[0] < x for x in bars) and pred(body[pexits[0]]) != (7, 0), pexits)
        ok('smfilter: it is preceded by S2R SR_VIRTUALSMID and two compares',
           low12(body[pexits[0] - 3]) == 0x919 and ((W(body[pexits[0] - 3]) >> 72) & 0xff) == 0x43
           and low12(body[pexits[0] - 2]) == 0x20c and low12(body[pexits[0] - 1]) == 0x20c)
    else:
        ok('no early EXIT', not pexits, pexits)
    if meta.get('gtime'):
        gt = lambda b, rd: low12(b) == 0x805 and (W(b) >> 16) & 0xff == rd and (W(b) >> 72) & 0xff == 0x52
        k = 1 if meta.get('clock') else 0
        g0r, g1r = meta.get('gt', [22, 24])
        ok('gtime: global timer reads around the clock reads', gt(body[g - 1 - k], g0r) and gt(body[g + 20 + k], g1r), (g0r, g1r))
    if meta.get('sync'):
        k = (1 if meta.get('clock') else 0) + (1 if meta.get('gtime') else 0)
        b = body[g - 1 - k - nb]
        ok('sync: NOPs between BAR.SYNC and the clock read / group', all(low12(body[g - 1 - k - i]) == 0x918 for i in range(nb)))
        ok('sync: blocking BAR.SYNC 0 (bit 80 clear) before the clock read / group',
           low12(b) == 0xb1d and ((W(b) >> 80) & 1) == 0 and ((W(b) >> 77) & 7) == 0)
    regs = int(next(l for l in os.popen('cuobjdump --dump-resource-usage %s' % path).read().split('Function ttu_one:')[1].split('\n') if 'REG:' in l).split('REG:')[1].split()[0])
    top = meta.get('top_reg', 95)
    ok('B14 declared registers >= highest register used + 3 (R%d)' % top, regs >= top + 3, regs)
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
