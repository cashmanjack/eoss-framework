"""R2 series: rounds of sessions on box scenes, gate = three way set equality (our decoded candidates, the OptiX intersection call log, the CPU expectation).
Stops at the first fault, hang, refusal or mismatch. Raw json per run in notes/rehost (timestamped, never overwritten)."""
import glob, json, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, '..', '..')
BUILD = sys.argv[1]; OUT = os.path.join(ROOT, 'notes', 'rehost')
VENV = os.path.expanduser('~/eoss/.venv/bin/python')
def boxes(sc):
    if sc == 'box2': return [(-.5, -.5, -.5, .5, .5, .5), (1.5, -.5, -.5, 2.5, .5, .5)]
    K = int(sc.split(':')[1])   # no numpy in the venv
    return [(-r, -r, -r, r, r, r) for r in (0.5 + 0.05 * k for k in range(K))]
RAYS = {'inside': ((0, 0, 0), (0, 0, 1), 1e-16), 'inside2': ((2, 0, 0), (0, 0, 1), 1e-16), 'xrow': ((-1, 0, 0), (1, 0, 0), 1e16), 'hit': ((0, 0, -1), (0, 0, 1), 1e16), 'outside': ((3, 3, 0), (0, 0, 1), 1e-16)}
def slab(b, o, d, tmax):
    t0, t1 = 0.0, tmax
    for a in range(3):
        lo, hi = b[a], b[a + 3]
        if abs(d[a]) < 1e-30:
            if o[a] < lo or o[a] > hi: return False
        else:
            ta, tb = (lo - o[a]) / d[a], (hi - o[a]) / d[a]
            if ta > tb: ta, tb = tb, ta
            t0 = max(t0, ta); t1 = min(t1, tb)
            if t0 > t1: return False
    return True
CASES = [('box2', 'inside'), ('box2', 'inside2'), ('box2', 'xrow'), ('box2', 'outside'), ('boxov:2', 'inside'), ('boxov:4', 'inside'), ('boxov:4', 'hit'), ('boxov:8', 'inside'), ('boxov:64', 'inside')]
if len(sys.argv) > 2: CASES = [tuple(c.split('/')) for c in sys.argv[2:]]
for sc, ray in CASES:
    lab = 'r2_%s_%s' % (sc.replace(':', '_'), ray)
    r = subprocess.run([VENV, os.path.join(HERE, 'r_run.py'), BUILD, ray, OUT, lab, '--scene', sc, '--aabb', '--warps', '1', '--repeat', '1', '--resume', '100'], capture_output=True, text=True)
    sys.stdout.write(''.join(l + '\n' for l in r.stdout.splitlines() if l.startswith(('nvidia-smi', 'ttu_one: no', 'ttu_one: ', 'resume:', 'r_run exit', 'refused', 'exit', 'check failed')) or 'rror' in l))
    if r.returncode: print('STOP: exit', r.returncode, r.stderr[-300:]); sys.exit(1)
    j = json.load(open(sorted(glob.glob(os.path.join(OUT, lab + '_*.json')))[-1]))
    o, d, t = RAYS[ray]; exp = [k for k, b in enumerate(boxes(sc)) if slab(b, o, d, t)]
    ox = [sorted(x[1:]) for x in j['islog']]; ours = [sorted(x) for x in j['cand_prim']]
    ordo = j['cand_prim'][0] == j['islog'][0][1:]
    nq = [abs(x) for x in j['lane_queries']]
    good = all(a == b == exp for a, b in zip(ours, ox)) and j['ttu_status'] == 'no error' and j['resume_status'] == 'no error' and all(x > 0 for x in j['lane_queries'])
    clk = [(rd['warp_clk'][0][1] - rd['warp_clk'][0][0]) for rd in j['resume_rounds']]
    print('%-9s %-8s expected %s | OptiX calls (lane 0) %s | ours (lane 0, answer order) %s | sets equal all lanes: %s | order equal: %s | sessions per lane %s | later-session cycles %s | PASS' % (sc, ray, exp if len(exp) < 9 else '%d prims' % len(exp), j['islog'][0][1:] if len(exp) < 9 else '%d calls' % j['islog'][0][0], j['cand_prim'][0] if len(exp) < 9 else '%d cands' % len(j['cand_prim'][0]), good, ordo, sorted(set(nq)), clk[:6]) if good else '%-9s %-8s MISMATCH expected %s ox %s ours %s -> STOP' % (sc, ray, exp, ox[0], ours[0]))
    if not good: sys.exit(1)
print('SERIES COMPLETE')
