"""IAS group mask run (predictions model/predictions_ias_opaque.md). run_ias_masks.py WORK MASK...   (MASK like 0x0F; builds in /dev/shm/t9i/m<MASK>, instance mask 15)
One warp, one launch, ray 'hit', stage 1 through setup3 run.py (checked binary, idle GPU, timeout 30). Stops at the first non zero exit or missing block."""
import json, os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); S3 = os.path.join(HERE, '..', '..', 'harness', 'setup3')
work = sys.argv[1]; ts = time.strftime('%Y%m%dT%H%M%S'); f32 = lambda u: __import__('struct').unpack('<f', __import__('struct').pack('<I', u))[0]
for m in sys.argv[2:]:
    out = '%s/ias_s1_im15_rm%s_%s.json' % (work, m, ts)
    if os.path.exists(out): sys.exit('exists')
    p = subprocess.run([sys.executable, os.path.join(S3, 'run.py'), '/dev/shm/t9i/m%s/ias_s1' % m, 'hit', 'zero', out, '--ias', '--imask', '15', '--warps', '1', '--repeat', '1'], capture_output=True, text=True)
    ok = 'exit 0' in p.stdout and 'ttu_one: no error' in p.stdout and '1 block(s) ran' in p.stdout
    if not ok: sys.exit('STOP %s:\n%s%s' % (m, p.stdout[-2000:], p.stderr[-800:]))
    j = json.load(open(out)); w = j['warp_out'][0]
    print('ray mask %s: R3(hit) %d u %.3f v %.3f t-word R7 %08x R24:25 %08x %08x R26:27 %08x%08x | optix ref %s' % (m, w[3], f32(w[24]), f32(w[25]), w[7], w[24], w[25], w[27], w[26], j['ref'][0]), flush=True)
