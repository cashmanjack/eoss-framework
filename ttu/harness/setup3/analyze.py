"""Check a setup3 result file.

    analyze.py OUT.json STAGE

Always: OptiX reference agrees with the CPU reference (hit and t).
Stage 0 (group = NOPs): every input register comes back unchanged in all 32 lanes.
Stage 1: all lanes agree; prints which registers the TTULDs changed.
"""
import json, os, struct, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ref import intersect, RAYS

INPUTS = [2, 3, 12, 13, 14, 15, 16, 17, 30, 31, 32, 33, 34, 35, 44, 45, 46, 47,
          52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 68, 69, 70, 71, 82, 83, 90]
LD_DEST = [32, 33, 44, 45, 68, 69, 30, 31, 2, 3, 62, 63, 34, 35, 58, 59, 56, 57, 70, 71, 60, 61]
IAS_INPUTS = [2, 3, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17, 24, 25, 26, 27, 28, 29, 40, 41, 48, 49, 50, 51, 52, 53, 90]   # build_patch.py --ias
IAS_LD_DEST = [24, 25, 6, 7, 26, 27, 2, 3, 44, 45, 42, 43, 48, 49, 46, 47, 52, 53, 50, 51]
# SM120 GAS group (build_patch.py --sm 120, setup3 --sm120; registers read from the SM120 live OptiX code, notes/protocol.md SM120).
SM120_INPUTS = [6, 7, 12, 13, 14, 15, 24, 25, 30, 31, 32, 33, 34, 35, 38, 39, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 66, 67, 68, 69, 70, 71, 78, 79, 90]
# TTULD destinations (Rd = low pair, Rd2 = high pair; RZ = discarded). SM120: 0x300 R66 R52, 0x310 R62 (high RZ), 0x320 R38 (high RZ), 0x380 R34 R68, 0x3C0 R56 R58, 0x3D0 R60 R70.
SM120_LD_DEST = [66, 67, 52, 53, 62, 63, 38, 39, 34, 35, 68, 69, 56, 57, 58, 59, 60, 61, 70, 71]   # TTULD destinations: 0x300 Rd R66 Rd2 R52, 0x310 R62, 0x320 R38, 0x380 R34 R68, 0x3C0 R56 R58, 0x3D0 R60 R70
# CORRECTED after the first SM120 stage 1 run (2026-11-19): the SM89 protocol notes' "low / high" pair order was an unverified assumption. In both architectures Rd (bits 23:16) holds the words
# the notes called 'high' (prim | flags and -t at 0x300, the leaf block address at 0x310, the hit flag at 0x320) and Rd2 the 'low' pair (u, v at 0x300). SM120 0x310 and 0x320 therefore keep the same data as SM89.
# SM120 roles, first run (one triangle, ray along +z, t = 1, u = 0.25, v = 0.5, back face): R52 = u with the sign bit = back face, R53 = v, R66 = prim | flags (0x80000000 | prim),
# R67 = -t (float, sign set), R38:R39 = 0 and the hit flag (R39 = 1), R62:R63 = leaf block address (GAS + 0x300), R34:R35 = the instance word echoed. Miss: R52 R53 R66 R67 all ones (SM89 pattern, check on the first miss).
SM120_ROLES = {52: 'u (sign = back face)', 53: 'v', 66: 'prim | flags', 67: '-t', 39: 'hit flag', 62: 'leaf lo', 63: 'leaf hi'}
f = lambda u: struct.unpack('<f', struct.pack('<I', u))[0]


def main(path, stage):
    j = json.load(open(path)); ok = True
    global INPUTS, LD_DEST
    sm120 = bool(j.get('sm120'))
    if sm120: INPUTS, LD_DEST = SM120_INPUTS, SM120_LD_DEST
    elif j.get('ias'): INPUTS, LD_DEST = IAS_INPUTS, IAS_LD_DEST   # IAS group registers (setup3 --ias)
    if 'ray' in j:
        o, d = j['ray']
    else:
        o, d = RAYS[j['ray']]
    tris = j.get('tris') or None
    if tris or 'tris' not in j:
        best = (False, None, None, None, None)
        for k, tri in enumerate(tris or [None]):
            r = intersect(o, d, tri=tri) if tri else intersect(o, d)
            if r[0] and (not best[0] or r[1] < best[1]):
                best = r + (k,)
        hit, t, u, v, prim = best
    else:
        hit = t = u = v = prim = None    # too many triangles stored: OptiX only
    r0 = j['ref'][0]
    same = all(r == r0 for r in j['ref'])
    t_ox = f(int(r0['t'], 16))
    if hit is None:
        agree = True
        print('cpu: skipped (%s triangles) | optix lane0: hit %d t %r prim %d | all lanes same %s' % (j.get('ntris'), r0['hit'], t_ox, r0['prim'], same))
    else:
        agree = (r0['hit'] == 1) == hit and (not hit or (abs(t_ox - t) < 1e-6 and r0['prim'] == prim))
        print('cpu: hit %s t %s prim %s | optix lane0: hit %d t %r prim %d front %s gas_index %s | all lanes same %s | agree %s'
              % (hit, t, prim, r0['hit'], t_ox, r0['prim'], r0.get('front'), r0.get('gas_index'), same, agree))
    ok &= same and agree
    print('scene %s gas %s size %s root %s inst %s inst_ok %s c3_34 %s' % (j.get('scene'), j['gas'], j.get('gas_size'), j['root'], j['inst'], j['inst_ok'], j.get('c3_34')))
    if 'out' not in j:
        return ok
    print('ttu_one status:', j['ttu_status']); ok &= j['ttu_status'] == 'no error'
    ins, outs = j['in'], j['out']
    keep = sorted(set(INPUTS) | (set(LD_DEST) if stage else set()))   # stage 0: IAS load destinations that are not inputs hold leftovers
    lanes_same = all(outs[i][r] == outs[0][r] for i in range(32) for r in keep)
    print('input and TTULD registers identical in all lanes:', lanes_same)
    if stage == 0:
        bad = [r for r in INPUTS if any(outs[i][r] != ins[i][r] for i in range(32))]
        print('stage 0: input registers changed or lost:', bad or 'none')
        ok &= lanes_same and not bad
    else:
        ok &= lanes_same
        for r in LD_DEST:
            print('  R%-2d in %08x out %08x %s' % (r, ins[0][r], outs[0][r], '' if ins[0][r] == outs[0][r] else 'CHANGED'))
        if sm120:   # role hypothesis check (informational: does not change the exit code until a stage 1 run has confirmed the roles)
            o0 = outs[0]; R = lambda r: o0[r]
            if hit:
                ub, vb, tb2 = struct.unpack('<I', struct.pack('<f', u))[0], struct.unpack('<I', struct.pack('<f', v))[0], struct.unpack('<I', struct.pack('<f', t))[0]
                chk = {'u (R52)': abs((R(52) & 0x7fffffff) - ub) <= 1, 'v (R53)': abs(R(53) - vb) <= 1, '-t (R67, sign set)': abs((R(67) & 0x7fffffff) - tb2) <= 1 and R(67) >> 31 == 1,
                       'prim (R66 low bits)': (R(66) & 0x1fffffff) == prim, 'hit flag R39 == 1': R(39) == 1}
                face = R(52) >> 31
                print('  SM120 roles (R52 u, R53 v, R66 prim, R67 -t, R39 hit) against the CPU:', chk, '| back face (R52 sign) %d, OptiX front %s (back face = front 0)' % (face, r0.get('front')), '| leaf - gas = %#x' % (((R(63) << 32) | R(62)) - int(j['gas'], 16)))
            else:
                print('  SM120 miss: R52 R53 R66 R67 all ones:', all(R(r) == 0xffffffff for r in (52, 53, 66, 67)), '| R39 %08x' % R(39))
        if hit:
            tb = struct.unpack('<I', struct.pack('<f', t))[0]
            m = [r for r in LD_DEST if abs(outs[0][r] - tb) <= 1]
            print('  registers equal to the CPU t (%08x, within 1 ulp):' % tb, m or 'none')
    print('PASS' if ok else 'FAIL')
    return ok


if __name__ == '__main__':
    sys.exit(0 if main(sys.argv[1], int(sys.argv[2])) else 1)
