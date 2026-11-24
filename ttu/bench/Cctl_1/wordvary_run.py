"""One word at a time variation of the ray stores (predictions model/predictions_ray_stores.md). One variant per call or several in order:
  wordvary_run.py OUTDIR W_c0_0 [W_c0_1 ...]
For each variant: preflight (nvidia-smi compute PIDs of anyone, who), then on BOTH a hit ray (tri1) and a miss ray scene (par:1024:s, session length): 1 launch of the unchanged
single session build and 1 of the variant (hit flag, t, u, v, prim read from the result registers, all 32 lanes must agree, differing columns against the base listed), then 30 launches
each for the session cycles. Classes: ran | fault | timeout or hang (health check after a non running outcome; stop everything if it fails). No process matching by name.
Raw to OUTDIR with mode x."""
import json, os, signal, statistics as st, struct, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import variants2_run as v
f32 = lambda u: struct.unpack('<f', struct.pack('<I', u))[0]
def run_scene(binp, scene, rep):
    jo = '%s/out/wv_%d.json' % (v.D, time.time_ns()); v.preflight('wv')
    cmd = ['timeout', '-k', '2', '30', v.VENV, os.path.join(v.S3, 'run.py'), binp, 'hit', 'zero', jo, '--scene', scene, '--warps', '1', '--blocks', '1', '--sms', '0,0', '--repeat', str(rep)]
    pr = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=v.ENV, start_new_session=True)
    try: out, _ = pr.communicate(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(pr.pid, signal.SIGKILL); pr.communicate(); return 'timeout or hang', None
    if 'Permission denied' in out or 'No such file' in out: sys.exit('STOP harness error: ' + out[-300:])
    if pr.returncode in (124, 137): return 'timeout or hang', out[-300:]
    if 'ttu_one: no error' in out and 'exit 0' in out and ' results same every launch: 1' in out:
        j = json.load(open(jo)); os.remove(jo); return 'ran', j
    return 'fault', out[-500:]
def decode(j):
    o = j['out']; agree = all(o[l][k] == o[0][k] for l in range(32) for k in range(len(o[0])) if k not in (84, 85, 86, 87) and k not in ADDR)
    return {'hit': o[0][3], 't': -f32(o[0][45]) if o[0][3] else None, 'u': f32(o[0][32] ^ 0x80000000), 'v': f32(o[0][33]), 'col44': hex(o[0][44]), 'col45': hex(o[0][45]), 'lanes_agree': agree}
ADDR = set()
def cyc(j): return [l['warp_clk'][0][1] - l['warp_clk'][0][0] for l in j['launches']]
def main(outd, names):
    os.makedirs(outd, exist_ok=True); os.makedirs(v.D + '/out', exist_ok=True); fn = '%s/wordvary_%s.json' % (outd, time.strftime('%Y%m%dT%H%M%S')); res = {}
    base = v.D + '/single_s1'
    for n in names:
        r = {}; print('=== %s' % n, flush=True); cls = 'ran'
        for scene in ('tri1', 'par:1024:s'):
            cb, jb = run_scene(base, scene, 1); cb2, jb2 = run_scene(base, scene, 1); cv, jv = run_scene('%s/%s_s1' % (v.D, n), scene, 1)
            if cb != 'ran' or cb2 != 'ran': sys.exit('STOP base failed: %s %s' % (cb, cb2))
            if cv != 'ran': cls = cv; r[scene] = {'class': cv, 'text': str(jv)[-300:]}; break
            a, b, c = jb['out'], jb2['out'], jv['out']; mask = {k for l in range(32) for k in range(len(a[l])) if a[l][k] != b[l][k]} | {84, 85, 86, 87}
            diffcols = sorted({k for l in range(32) for k in range(len(a[l])) if k not in mask and a[l][k] != c[l][k]})
            ADDR.clear(); ADDR.update(mask - {84, 85, 86, 87})
            cbm = run_scene(base, scene, 30); cvm = run_scene('%s/%s_s1' % (v.D, n), scene, 30)
            r[scene] = {'base': decode(jb), 'variant': decode(jv), 'differing_columns': diffcols, 'cycles_base': int(st.median(cyc(cbm[1]))), 'cycles_variant': int(st.median(cyc(cvm[1]))) if cvm[0] == 'ran' else cvm[0]}
            print(n, scene, 'base', {k: r[scene]['base'][k] for k in ('hit', 't', 'u', 'v')}, 'variant', {k: r[scene]['variant'][k] for k in ('hit', 't', 'u', 'v', 'lanes_agree')}, 'cols', diffcols, 'cycles', r[scene]['cycles_base'], r[scene]['cycles_variant'], flush=True)
        r['class'] = cls
        if cls != 'ran':
            ok, info = v.health(); r['health_after'] = {'ok': ok, 'info': str(info)}; print(n, 'OUTCOME', cls, 'health after', ok, info, flush=True)
            if not ok: res[n] = r; json.dump(res, open(fn, 'x')); sys.exit('STOP: health check failed after %s. Tell Jack immediately.' % n)
        res[n] = r
    json.dump(res, open(fn, 'x')); print('raw', fn)
if __name__ == '__main__': main(sys.argv[1], sys.argv[2:])
