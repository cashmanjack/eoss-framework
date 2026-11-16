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
            if meta.get('ias'): shape = [ST] * 8 + [GO] + [LD] * 6   # F91 IAS group: fuse count 15
            ok(tag + ('A4 group is 10 ST, GO, 10 ST, GO, 6 LD (dualgo)' if meta.get('dualgo') else 'A4 group is 8 ST, GO, 6 LD (IAS)' if meta.get('ias') else 'A4 group is 10 ST, then 1 GO, then 6 LD'), kinds == shape, kinds)
            if meta.get('ias'):   # A13 (IAS only): the F91 shape, in the harness's own (possibly remapped) registers: store and load offsets, which stores are predicated and how
                offs = lambda b: (W(b) >> 40) & 0xffff
                ok(tag + 'A13 IAS store offsets 0, 0x40, 0x50, 0x80, 0x100, 0x180, 0x190, 0x200', [offs(b) for b in grp[:8]] == [0, 0x40, 0x50, 0x80, 0x100, 0x180, 0x190, 0x200], [hex(offs(b)) for b in grp[:8]])
                ok(tag + 'A13 IAS load offsets 0x300 0x310 0x320 0x380 0x3C0 0x3D0', [offs(b) for b in grp[9:]] == [0x300, 0x310, 0x320, 0x380, 0x3c0, 0x3d0], [hex(offs(b)) for b in grp[9:]])
                ok(tag + 'A13 IAS predicates: four stores unpredicated, then three @!P0 (0x100, 0x180, 0x190) and one @P0 (0x200); GO and loads unpredicated',
                   [pred(b) for b in grp[:8]] == [(7, 0)] * 4 + [(0, 1)] * 3 + [(0, 0)] and all(pred(b) == (7, 0) for b in grp[8:]), [pred(b) for b in grp])
                ok(tag + 'A13 IAS fuse count is 15', n == 15, n)
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
            dump = [d[0xbb60 + 16 * k:0xbb60 + 16 * k + 16] for k in range(17)] if meta.get('ias') else [d[0x14000 - 0xfb50 + 16 * k:0x14000 - 0xfb50 + 16 * k + 16] for k in range(19)]
            regmask = lambda b: W(b) & ~((0xff << 16) | (0xff << 32) | (0xff << 64)) if low12(b) in (ST, LD) else W(b)
            for gi in groups:
                same = all(regmask(a) == regmask(b) for a, b in zip(body[gi:gi + len(dump)], dump))
                ok('A10 remapped group equals the OptiX group except register fields (session at %d)' % gi, same)
            pm = {int(k): v for k, v in meta['regmap'].items()}
            ok('A10 register map is a pair aligned one to one map', len(set(pm.values())) == len(pm) and all(pm[k] % 2 == k % 2 for k in pm))
        brs = [i for i, b in enumerate(body[:ex]) if low12(b) in (0x947, 0x949, 0x943)]
        if meta.get('loop'):   # --loop: exactly one backward BRA, after the group, back to TTUOPEN; never inside a session
            off = lambda b: (W(b) >> 32) & 0xffffffff
            tgt = [i + 1 + ((off(body[i]) - (1 << 32 if off(body[i]) >> 31 else 0)) // 16) for i in brs]
            if meta.get('outer'):   # --outer: the inner branch to the TTUOPEN, then one outer backward BRA to the record's pointer set up, after the stores
                oa = meta['outer_at']
                ok('B15 outer: two backward BRAs: to the TTUOPEN (after the CLOSE), then to the record start (after the stores, before EXIT)',
                   len(brs) == 2 and all(low12(body[i]) == 0x947 for i in brs) and tgt == [g, oa] and brs[0] > g + 18 and brs[1] > brs[0] and oa < g, (brs, tgt, g, oa))
            else:
                ok('B15 loop: exactly one branch, a backward BRA to the TTUOPEN, after the CLOSE',
                   len(brs) == 1 and low12(body[brs[0]]) == 0x947 and tgt == [g] and brs[0] > g + 18, (brs, tgt, g))
            ok('B15 loop: no TTU instruction after the branch target other than the one group', len(hits) == 19, len(hits))
        elif meta.get('sibling'):   # q20 --sibling: exactly one branch, an @!P6 backward BRA to the spin load, after the first session's CLOSE and before the second group; no TTU instruction inside the spin
            off = lambda b: (W(b) >> 32) & 0xffffffff
            tgt = [i + 1 + ((off(body[i]) - (1 << 32 if off(body[i]) >> 31 else 0)) // 16) for i in brs]
            g1 = groups[0]
            if meta.get('sibccctl'):   # (h): the one extra branch in the main code is the forward @P1 BRA to the B path; the backward spin test below is about the rest
                fw = meta['sibccctl_bra']
                ok('B15 sibccctl: one forward @P1 BRA right after the smfilter EXIT, target = the B path after the main EXIT', fw in brs and pred(body[fw]) == (1, 0) and tgt[brs.index(fw)] == meta['bpath'] and meta['bpath'] > ex + 1, (fw, brs, tgt, meta['bpath'], ex))
                brs = [i for i in brs if i != fw]; tgt = [t for i, t in zip([i for i in range(len(body[:ex])) if low12(body[i]) in (0x947, 0x949, 0x943)], tgt) if i != fw]
            ok('B15 sibling: exactly one backward BRA (@!P6) between the sessions, to the spin load, no TTU instruction in the loop',
               len(brs) == 1 and low12(body[brs[0]]) == 0x947 and g1 + 18 < tgt[0] < brs[0] < groups[1] and ((W(body[brs[0]]) >> 12) & 0xf) == 0xe
               and low12(body[tgt[0]]) == 0x981 and not any(low12(body[i]) in TTU for i in range(tgt[0], brs[0] + 1)), (brs, tgt, groups))
            fl = [i for i, b in enumerate(body[:ex]) if low12(b) in (0x992, 0x9ab, 0x98f)]   # MEMBAR, ERRBAR, CCTL (this build's only memory fence opcodes)
            if meta.get('ownfence'):
                ok('B15 ownfence: exactly MEMBAR.ALL.GPU, ERRBAR, CCTL.IVALL in a row, unpredicated, after the first CLOSE and before the spin, outside every session',
                   len(fl) == 3 and fl[1] == fl[0] + 1 and fl[2] == fl[0] + 2 and g1 + 18 < fl[0] and fl[2] < tgt[0] and [low12(body[i]) for i in fl] == [0x992, 0x9ab, 0x98f] and all(pred(body[i]) == (7, 0) for i in fl), (fl, tgt, groups))
            else:
                ok('B15 no fence instruction in the TTU kernel unless --ownfence', not fl, fl)
        else:
            ok('B15 no branch anywhere before EXIT', not brs)
    if meta.get('cctl'):   # Interface_7: exactly one TTUCCTL.IVALL in stage 1 (none in stage 0), unpredicated, right after the first session's wait NOP, outside every session, followed by 4 stall 15 NOPs
        cc = [i for i, b in enumerate(body) if low12(b) == 0x3d5]
        g0 = groups[0]; GLc = meta.get('group_len', 19)
        if meta['stage'] == 1:
            ok('cctl: exactly one TTUCCTL, ivall bit 80 set, unpredicated', len(cc) == 1 and bit(body[cc[0]], 80) and pred(body[cc[0]]) == (7, 0), cc)
            if meta.get('l1t'):   # --l1t: only ordinary loads (LDG 0x981) and NOPs between the first session's wait NOP and the TTUCCTL
                ok('cctl (l1t): after the first session and its wait NOP, only LDG and NOP before it, before the second TTUOPEN', len(cc) == 1 and g0 + GLc + 1 <= cc[0] < groups[1] and all(low12(body[k]) in (0x981, 0x918) for k in range(g0 + GLc + 1, cc[0])), (cc, groups))
            else:
                ok('cctl: right after the first session and its wait NOP, before the second TTUOPEN', cc == [g0 + GLc + 1] and cc[0] < groups[1], (cc, groups))
            from build_patch import ttuccttl_bytes
            ok('cctl: bytes equal the EoSS encoder output for TTUCCTL.IVALL', bytes(body[cc[0]]) == ttuccttl_bytes())
            ok('cctl: followed by 4 NOPs with stall 15', all(low12(body[cc[0] + 1 + k]) == 0x918 and ((W(body[cc[0] + 1 + k][8:]) >> 41) & 0xf) == 15 for k in range(4)))
        else:
            ok('cctl: stage 0 has no TTUCCTL', not cc, cc)
    if meta.get('sibccctl'):   # q22 (h): the B path (after the main EXIT and BRA self): no session, no OPEN/FUSE/ST/GO/LD, exactly one TTUCCTL.IVALL (stage 1; NOP in stage 0), after its spin on flag1, never predicated
        bp = meta['bpath']; last = max(i for i in range(bp, len(body)) if low12(body[i]) == 0x947); tail = body[bp:last + 1]   # the section after the B path is NOP padding
        cc = [bp + i for i, b in enumerate(tail) if low12(b) == 0x3d5]
        ok('(h) B path starts after the main EXIT and BRA self, ends with EXIT and BRA self', bp == ex + 2 and low12(body[last - 1]) == 0x94d and pred(body[last - 1]) == (7, 0) and all(low12(b) == 0x918 for b in body[last + 1:]), (bp, ex, last))
        ok('(h) B path has no TTUOPEN, TTUMACROFUSE, TTUST, TTUGO or TTULD', not [b for b in tail if low12(b) in TTU])
        ok('(h) B path in the whole binary: exactly one TTUCCTL.IVALL in stage 1, none in stage 0, unpredicated, ivall bit 80', (len(cc) == 1 and bit(body[cc[0]], 80) and pred(body[cc[0]]) == (7, 0)) if meta['stage'] == 1 else not cc, cc)
        if meta['stage'] == 1:
            from build_patch import ttuccttl_bytes
            ok('(h) TTUCCTL bytes equal the EoSS encoder output', bytes(body[cc[0]]) == ttuccttl_bytes())
            ok('(h) no TTUCCTL anywhere else in the kernel', len([1 for b in body if low12(b) == 0x3d5]) == 1)
            bra = [i for i in range(bp, last) if low12(body[i]) == 0x947]
            ok('(h) TTUCCTL comes after the one backward spin BRA (flag1) and is followed by 4 stall 15 NOPs', len(bra) == 1 and bra[0] < cc[0] and all(low12(body[cc[0] + 1 + k]) == 0x918 and ((W(body[cc[0] + 1 + k][8:]) >> 41) & 0xf) == 15 for k in range(4)), (bra, cc))
        stg_ = [bp + i for i, b in enumerate(tail) if low12(b) == 0x986]
        ok('(h) B path stores exactly once (flag2 at +128)', len(stg_) == 1 and ((W(body[stg_[0]]) >> 40) & 0xffffff) == 128, stg_)
    wt = lambda b: (W(b[8:]) >> 52) & 0x3f
    stall = lambda b: (W(b[8:]) >> 41) & 0xf
    nb = meta.get('barnops', 1) if meta.get('sync') else 0
    CLK = meta.get('clock') and not meta.get('l1t')   # --l1t: the clock window is the timed reload between the sessions, not the group
    c = (1 if CLK else 0) + ((1 + nb) if meta.get('sync') else 0) + (1 if meta.get('preclk') else 0) + (1 if meta.get('gtime') else 0) + (1 if meta.get('sr133') else 0) + (1 if meta.get('sleepb') else 0)
    GL = meta.get('group_len', 19)
    NA = meta.get('alu', 0)
    if NA:   # Throughput_4: exactly NA IADD3 right after the group, none of them TTU, then the wait NOP
        ok('alu: %d IADD3 right after the group, before the wait NOP' % NA,
           all(low12(body[g + GL + k]) == 0x810 for k in range(NA)) and low12(body[g + GL + NA]) == 0x918)
    GL += NA
    if meta.get('sleepb'):   # @P1 NANOSLEEP right after the P0 stall NOP, before any clock read and the group
        ns = [i for i, b in enumerate(body[:ex]) if low12(b) == 0x95d]
        ok('sleepb: one @P1 NANOSLEEP, before the group', len(ns) == 1 and ns[0] < g and pred(body[ns[0]]) == (1, 0), ns)
    is_clk = lambda b, rd: low12(b) == 0x805 and (W(b) >> 16) & 0xff == rd and (W(b) >> 72) & 0xff == 0x50
    cs, ce = meta.get('clk', [84, 86])
    f4 = 4 if meta.get('smfilter') and not meta.get('twice') else 0   # S2R, 2 ISETP, @P0 EXIT sit between the wait and ISETP P0
    ok('B12 loads waited on before ISETP P0 (and before the SM filter)', low12(body[g - 3 - c - f4]) == 0x918 and wt(body[g - 3 - c - f4]) == 0x3f)
    ok('B12 ISETP P0 then a NOP with stall 15 before the group',
       low12(body[g - 2 - c]) == 0x80c and low12(body[g - 1 - c]) == 0x918 and stall(body[g - 1 - c]) == 15)
    ok('B13 NOP after the group waits on all scoreboards', low12(body[g + GL]) == 0x918 and wt(body[g + GL]) == 0x3f)
    if meta.get('l1t'):
        cl = [i for i, b in enumerate(body) if is_clk(b, cs) or is_clk(b, ce)]
        ok('l1t: exactly two CS2R SR_CLOCKLO (start, end), both between the sessions, no TTU instruction between them', len(cl) == 2 and groups[0] + GL < cl[0] < cl[1] < groups[1] and not [i for i in range(cl[0], cl[1]) if low12(body[i]) in TTU], cl)
    if CLK:
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
