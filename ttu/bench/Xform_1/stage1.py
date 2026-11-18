"""Xform_1 stage 1 (approved by Jack 2026-11-18). stage1.py WORK
Phase A: stage 0 (/dev/shm/t9x/ias_s0n, no TTU instruction) for the 5 matrices x 6 rays, OptiX against the float64 CPU reference (as prepare.py). Phase B: stage 1 (/dev/shm/t9x/ias_s1, the F139 IAS group, checked build), one warp, one launch per matrix and ray, identity hit first.
Per launch (predictions model/predictions_phase10e.md): exit 0, 'no error', input registers (not load destinations) unchanged, the six decoded registers identical in all 32 lanes, hit flag equal to OptiX in the same process, |t|, |u| and v within 2 ulp of OptiX's; a miss gives R3 0 and all ones in R6:R7.
STOP at the first deviation (no retry). Raw JSON WORK/x1_<stage>_<imat>_<ray>_<TS>.json, never overwritten (the run is refused if the name exists).
The plan's last item (32 different rays in one launch) is NOT done here: setup3 takes one ray per launch (FILL is the register fill, not a ray set), it would need new host code."""
import json, os, struct, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); S3 = os.path.join(HERE, '..', '..', 'harness', 'setup3'); sys.path.insert(0, HERE); sys.path.insert(0, S3)
import prepare as P  # noqa: E402  (MATS, world_tri, u2f)
import ref  # noqa: E402
IN = [2, 3, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17, 24, 25, 26, 27, 28, 29, 40, 41, 48, 49, 50, 51, 52, 53, 90]; LD = [24, 25, 6, 7, 26, 27, 2, 3, 44, 45, 42, 43, 48, 49, 46, 47, 52, 53, 50, 51]
KEEP = [r for r in IN if r not in LD]; DEC = [2, 3, 6, 7, 24, 25, 26, 27]
def ulp(a, b): return abs((a & 0x7fffffff) - (b & 0x7fffffff))
def launch(work, binp, imat, ray, fill, tag, stage):
    out = os.path.join(work, 'x1_%s_%s_%s_%s.json' % (stage, imat, ray, tag))
    if os.path.exists(out): sys.exit('exists ' + out)
    p = subprocess.run([sys.executable, os.path.join(S3, 'run.py'), binp, ray, fill, out, '--ias', '--imat', imat, '--warps', '1', '--repeat', '1'], capture_output=True, text=True)
    if p.returncode or 'exit 0' not in p.stdout or 'no error' not in p.stdout: sys.exit('STOP %s %s %s\n%s%s' % (stage, imat, ray, p.stdout[-1800:], p.stderr[-600:]))
    return json.load(open(out)), p.stdout
def check1(j, r0):
    o, i = j['out'], j['in']; msgs = []
    if any(o[l][r] != i[l][r] for l in range(32) for r in KEEP): msgs.append('input registers changed')
    if any(o[l][r] != o[0][r] for l in range(32) for r in DEC): msgs.append('lanes differ')
    w = o[0]
    if bool(w[3] & 1) != bool(r0['hit']): msgs.append('hit flag %d vs optix %d' % (w[3], r0['hit']))
    elif r0['hit']:
        if ulp(w[7], int(r0['t'], 16)) > 2: msgs.append('t %08x vs %s' % (w[7], r0['t']))
        if ulp(w[24], int(r0['u'], 16)) > 2: msgs.append('u %08x vs %s' % (w[24], r0['u']))
        if ulp(w[25], int(r0['v'], 16)) > 2: msgs.append('v %08x vs %s' % (w[25], r0['v']))
    elif not (w[6] == 0xffffffff and w[7] == 0xffffffff): msgs.append('miss marker R6:R7 %08x %08x' % (w[6], w[7]))
    return msgs
def main(work):
    os.makedirs(work, exist_ok=True); ts = time.strftime('%Y%m%dT%H%M%S'); rays = ('hit', 'moved', 'moved2', 'miss', 't2', 'back'); bad0 = 0
    for imat in P.MATS:
        tri = P.world_tri(P.MATS[imat])
        for rn in rays:
            j, _ = launch(work, '/dev/shm/t9x/ias_s0n', imat, rn, 'zero', ts, 's0'); r0 = j['ref'][0]; o, d = ref.RAYS[rn]; hit, t, u, v = ref.intersect(o, d, tri=tri)
            ok = bool(r0['hit']) == hit and (not hit or (abs(P.u2f(int(r0['t'], 16)) - t) <= 2e-5 * max(1, abs(t)) and abs(P.u2f(int(r0['u'], 16)) - u) < 2e-5 and abs(P.u2f(int(r0['v'], 16)) - v) < 2e-5))
            ok = ok and not any(j['out'][l][r] != j['in'][l][r] for l in range(32) for r in IN)
            print('stage 0 %-7s %-7s optix hit %d cpu hit %d %s' % (imat, rn, r0['hit'], hit, 'ok' if ok else 'MISMATCH'), flush=True); bad0 += not ok
    if bad0: sys.exit('STOP: stage 0 mismatches %d, stage 1 not run' % bad0)
    print('stage 0: 30 of 30 ok; stage 1 starts', flush=True); n = 0
    for imat in P.MATS:
        for rn in rays:
            j, so = launch(work, '/dev/shm/t9x/ias_s1', imat, rn, 'zero', ts, 's1'); m = check1(j, j['ref'][0]); w = j['out'][0]; n += 1
            print('stage 1 %-7s %-7s R3 %d R7 %08x R24 %08x R25 %08x R26:27 %08x:%08x R2 %08x | optix t %s u %s v %s | %s' % (imat, rn, w[3], w[7], w[24], w[25], w[27], w[26], w[2], j['ref'][0]['t'], j['ref'][0]['u'], j['ref'][0]['v'], 'PASS' if not m else 'FAIL ' + '; '.join(m)), flush=True)
            if m: sys.exit('STOP at launch %d (no retry)' % n)
    print('stage 1: %d of 30 pass' % n)

if __name__ == '__main__': main(sys.argv[1])
