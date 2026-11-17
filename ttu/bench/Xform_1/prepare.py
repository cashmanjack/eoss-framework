"""Xform_1 prepare (stage 0 ONLY, no TTU instruction runs). prepare.py WORK
For each instance transform (id, rz37, rz90s2, rx37, snu) and each ray (hit, moved, moved2, miss, t2, back): the stage 0 IAS build (/dev/shm/t9x/ias_s0, check_seq ALL PASS) runs the OptiX reference through the
transformed IAS; its answer (hit, t, primitive, u, v) is compared with a float64 CPU reference on the triangle moved by the matrix. Also compares the register image words 'in' of lane 0 against the identity run (two identity runs first:
the words that differ between two identity runs are process dependent addresses and are ignored).
Raw JSON WORK/xf_<imat>_<ray>_<TS>.json (never overwritten, the run script refuses). Stage 1 is NOT run by this script and the stage 1 binary is never named here."""
import json, os, struct, subprocess, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); S3 = os.path.join(HERE, '..', '..', 'harness', 'setup3'); sys.path.insert(0, S3)
import ref  # noqa: E402
f32 = np.float32; c37 = float(np.cos(f32(0.6458))); s37 = float(np.sin(f32(0.6458)))
MATS = {'id': [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], 'rz37': [c37, -s37, 0, 0.05, s37, c37, 0, -0.03, 0, 0, 1, 0.02], 'rz90s2': [0, -2, 0, 0.1, 2, 0, 0, 0.2, 0, 0, 2, 0.3],
        'rx37': [1, 0, 0, 0, 0, c37, -s37, 0, 0, s37, c37, 0.1], 'snu': [2, 0, 0, 0.05, 0, 0.5, 0, -0.02, 0, 0, 1.5, 0.2]}   # keep in sync with setup3.cu --imat
def world_tri(m):
    m = [float(f32(x)) for x in m]; M = np.array(m).reshape(3, 4)
    return tuple(tuple(M[:, :3] @ np.array(v, float) + M[:, 3]) for v in ref.TRI)
def u2f(u): return struct.unpack('<f', struct.pack('<I', u & 0xffffffff))[0]
def run(work, imat, ray, tag):
    out = os.path.join(work, 'xf_%s_%s_%s.json' % (imat, ray, tag))
    p = subprocess.run([sys.executable, os.path.join(S3, 'run.py'), '/dev/shm/t9x/ias_s0', ray, 'zero', out, '--ias', '--imat', imat, '--warps', '1', '--repeat', '1'], capture_output=True, text=True)
    if 'exit 0' not in p.stdout or 'no error' not in p.stdout: sys.exit('STOP %s %s\n%s%s' % (imat, ray, p.stdout[-1500:], p.stderr[-500:]))
    return json.load(open(out))
def main(work):
    os.makedirs(work, exist_ok=True); ts = time.strftime('%Y%m%dT%H%M%S'); res = []
    a = run(work, 'id', 'hit', ts + 'a'); b = run(work, 'id', 'hit', ts + 'b'); ia, ib = a['in'][0], b['in'][0]; proc_dep = {i for i in range(len(ia)) if ia[i] != ib[i]}
    print('identity run twice: words of lane 0 image that differ between two identity runs (process dependent):', sorted(proc_dep), flush=True)
    bad = 0
    for imat in MATS:
        tri = world_tri(MATS[imat])
        for rn in ('hit', 'moved', 'moved2', 'miss', 't2', 'back'):
            j = a if (imat == 'id' and rn == 'hit') else run(work, imat, rn, ts); r0 = j['ref'][0]; o, d = ref.RAYS[rn]
            hit, t, u, v = ref.intersect(o, d, tri=tri); ok = (bool(r0['hit']) == hit)
            if hit and r0['hit']: ok = ok and abs(u2f(int(r0['t'], 16)) - t) <= 2e-5 * max(1, abs(t)) and abs(u2f(int(r0['u'], 16)) - u) < 2e-5 and abs(u2f(int(r0['v'], 16)) - v) < 2e-5 and r0['prim'] == 0
            diff = [i for i in range(len(j['in'][0])) if j['in'][0][i] != ia[i] and i not in proc_dep] if rn == 'hit' else '-'
            print('%-7s %-7s optix hit %d t %s | cpu hit %d t %s u %s v %s | %s | image words differing from id (hit ray): %s' % (imat, rn, r0['hit'], '%.6f' % u2f(int(r0['t'], 16)) if r0['hit'] else '-', hit, '%.6f' % t if hit else '-', '%.5f' % u if hit else '-', '%.5f' % v if hit else '-', 'MATCH' if ok else 'MISMATCH', diff), flush=True)
            bad += not ok
    print('mismatches:', bad)
if __name__ == '__main__': main(sys.argv[1])
