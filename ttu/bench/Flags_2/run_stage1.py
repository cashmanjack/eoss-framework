"""Flags_2 stage 1 (approved by Jack 2026-11-17, plan notes/FLAGS2_PLAN.md). Runs prepare.py's builds one at a time: check_seq on the binary, run.py (idle GPU check, timeout -k 2 30, one warp, one launch), marker deleted again.
    run_stage1.py DIR VARIANT[,VARIANT...] [RAYS=hit,back]
Stops (sys.exit) on the first bad exit, hang, missing 'no error', lanes that disagree, or a result that differs from the OptiX behaviour of the same flag when the prediction says it must not.
Expected hit (hit ray = +z ray, back face; back ray = front face): from the OptiX runs of the same flags (notes/flags2/optix_*), the same as model/predictions_flags2.md."""
import json, os, struct, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); S3 = os.path.join(HERE, '..', '..', 'harness', 'setup3')
VENV = os.path.expanduser('~/eoss/.venv/bin/python')
EXP = {  # variant: (hit ray hits, back ray hits)
    'none': (1, 1), 'TERMINATE': (1, 1), 'DISABLE_ANYHIT': (1, 1), 'ENFORCE_ANYHIT': (1, 1), 'CULL_OPAQUE': (1, 1), 'mask128': (1, 1), 'mask15': (1, 1), 'mask1': (1, 1),
    'CULL_BACK': (0, 1), 'CULL_FRONT': (1, 0), 'CULL_NON_OPAQUE': (0, 0), 'mask0': (0, 0)}
f = lambda u: struct.unpack('<f', struct.pack('<I', u))[0]
d, variants = sys.argv[1], sys.argv[2].split(','); rays = (sys.argv[3] if len(sys.argv) > 3 else 'hit,back').split(',')
TS = time.strftime('%Y%m%dT%H%M%S'); W = os.path.join(HERE, '..', '..', '..', 'notes', 'flags2'); os.makedirs(W, exist_ok=True)
for v in variants:
    for ray in rays:
        b = '%s/%s/setup3_s1' % (d, v); out = '%s/stage1_%s_%s_%s.json' % (W, v, ray, TS)
        assert not os.path.exists(out)
        c = subprocess.run([VENV, os.path.join(S3, 'check_seq.py'), b], capture_output=True, text=True)
        if 'ALL PASS' not in c.stdout.strip().splitlines()[-1]: sys.exit('check_seq FAILED ' + b)
        print('RUN', v, ray, flush=True)
        r = subprocess.run([sys.executable, os.path.join(S3, 'run.py'), b, ray, 'zero', out], capture_output=True, text=True, env=dict(os.environ, TTU_GPU='1'))
        os.remove(b + '.checked')
        print(r.stdout[-700:].replace('\n', ' | '), r.stderr[-300:], flush=True)
        if r.returncode or 'ttu_one: no error' not in r.stdout or not os.path.exists(out): sys.exit('STOP: bad run %s %s' % (v, ray))
        j = json.load(open(out)); wo = j['warp_out']
        lanes = {(w[3], w[44] & 0x1fffffff, w[45], w[32], w[33]) for w in wo}
        if len(lanes) != 1: sys.exit('STOP: lanes disagree %s %s %s' % (v, ray, lanes))
        hit, prim, t, u, vv = lanes.pop(); exp = EXP[v][0 if ray == 'hit' else 1]
        ref = j['ref'][0]
        print('RESULT %-16s %-5s hit %d (expected %d) prim %d t %.6f u %08x v %08x | OptiX ref lane (flags NONE) hit %d' % (v, ray, hit, exp, prim, -f(t), u, vv, ref['hit']), flush=True)
        if hit != exp: sys.exit('STOP: unexpected outcome %s %s (computed word wrong or semantics differ): no further runs' % (v, ray))
