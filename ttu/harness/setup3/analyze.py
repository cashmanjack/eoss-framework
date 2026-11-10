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
f = lambda u: struct.unpack('<f', struct.pack('<I', u))[0]


def main(path, stage):
    j = json.load(open(path)); ok = True
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
    keep = sorted(set(INPUTS) | set(LD_DEST))
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
        if hit:
            tb = struct.unpack('<I', struct.pack('<f', t))[0]
            m = [r for r in LD_DEST if abs(outs[0][r] - tb) <= 1]
            print('  registers equal to the CPU t (%08x, within 1 ulp):' % tb, m or 'none')
    print('PASS' if ok else 'FAIL')
    return ok


if __name__ == '__main__':
    sys.exit(0 if main(sys.argv[1], int(sys.argv[2])) else 1)
