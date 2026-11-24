"""Extra check after variants2_run.py (same variant binaries, other scene): results of a variant against its base build on a scene whose ray HITS (tri1), one launch each,
the same masks as variants2_run.results_equal. hitcheck.py OUTDIR SCENE NAME [NAME ...]. No new instruction sequence: the same checked builds, only the scene argument differs."""
import json, os, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import variants2_run as v
def one(binp, scene):
    v.preflight('hit'); jo = '%s/out/hc_%d.json' % (v.D, time.time_ns()); cmd = ['timeout', '-k', '2', '30', v.VENV, os.path.join(v.S3, 'run.py'), binp, 'hit', 'zero', jo, '--scene', scene, '--warps', '1', '--blocks', '1', '--sms', '0,0', '--repeat', '1']
    pr = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=v.ENV, start_new_session=True); out, _ = pr.communicate(timeout=50)
    ok = 'ttu_one: no error' in out and 'exit 0' in out and ' results same every launch: 1' in out
    if not ok: return 'fault or other: ' + out[-300:], None, jo
    j = json.load(open(jo)); return 'ran', j, jo
def main(outd, scene, names):
    res = {}
    for n in names:
        base = v.BASE[n]; bp = '%s/%s_s1' % (v.D, base); r = {}
        ca, ja, fa = one(bp, scene); cb, jb, fb = one(bp, scene); cv, jv, fv = one('%s/%s_s1' % (v.D, n), scene)
        for f in (fa, fb, fv):
            if os.path.exists(f): os.remove(f)
        if ca != 'ran' or cb != 'ran': sys.exit('STOP base failed on %s: %s %s' % (scene, ca, cb))
        if cv != 'ran': r = {'class': 'fault', 'text': cv}
        else:
            a, b, c = ja['out'], jb['out'], jv['out']; mask = {k for l in range(32) for k in range(len(a[l])) if a[l][k] != b[l][k]} | {84, 85, 86, 87}
            dif = sorted({k for l in range(32) for k in range(len(a[l])) if k not in mask and a[l][k] != c[l][k]})
            hit = [ja['ref'][0] if 'ref' in ja else None]
            r = {'class': 'ran correctly' if not dif else 'wrong result', 'differing_columns': dif, 'optix_ref_lane0': str(ja['ref'][0])[:120], 'lane0_base': [hex(a[0][k]) for k in dif[:8]], 'lane0_variant': [hex(c[0][k]) for k in dif[:8]]}
            ra, rv = ja['launches'][0]['warp_clk'][0], jv['launches'][0]['warp_clk'][0]; r['cycles_base'] = ra[1] - ra[0]; r['cycles_variant'] = rv[1] - rv[0]
        res[n] = r; print(n, scene, r, flush=True)
    fn = '%s/hitcheck_%s.json' % (outd, time.strftime('%Y%m%dT%H%M%S'))
    with open(fn, 'x') as f: json.dump(res, f)
if __name__ == '__main__': main(sys.argv[1], sys.argv[2], sys.argv[3:])
