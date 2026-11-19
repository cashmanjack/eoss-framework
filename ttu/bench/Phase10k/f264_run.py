"""F264 rerun (Phase 10k, predictions model/predictions_phase10k.md). The fix: the host needs --coal for the --coal build (it expands the 16 byte records).
    f264_run.py PHASE     PHASE: build | s0 | s1w1 | big      (one phase per call; each stops at the first failure; raw files timestamped, mode x)
Builds in /dev/shm/t10k (patched NVIDIA code: delete after). Stage 0 readback check: every record stored, same rounds left in all lanes, hit word 0."""
import json, os, struct, subprocess, sys, time
import numpy as np
os.environ['TTU_GPU'] = '1'; os.environ.pop('CUDA_VISIBLE_DEVICES', None)
R = os.path.expanduser('~/eoss'); D = '/dev/shm/t10k'; OUT = R + '/notes/p10k'; S3 = R + '/ttu/harness/setup3'
VENV = os.path.expanduser('~/eoss/.venv/bin/python'); RUN = S3 + '/run.py'
ts = time.strftime('%Y%m%dT%H%M%S'); K, OUTER = 32, 1000
FLAGS = ['--slim', '--remap', '--compact-out', '--regs', '96', '--max-out', '61', '--loop', str(K), '--outer', str(OUTER), '--stride', '0', '--same-record', '--hoist', '--coal']
HOST = ['--blockrays', '--slim', '--loop', '--coal']
def sh(c, **kw): return subprocess.run(c, capture_output=True, text=True, **kw)
def build():
    os.makedirs(D, exist_ok=True)
    for f in ('setup3', 'setup3_dev.ptx', 'words.txt'): sh(['cp', '/dev/shm/t9/' + f, D + '/' + f])
    for st in (0, 1):
        p = sh([VENV, S3 + '/build_patch.py', '--stage', str(st), '--bin', D + '/setup3', '--out', '%s/sr_s%d' % (D, st)] + FLAGS)
        if p.returncode: sys.exit('STOP build\n' + p.stdout[-800:] + p.stderr[-800:])
        c = sh([VENV, S3 + '/check_seq.py', '%s/sr_s%d' % (D, st)])
        if not c.stdout.strip() or 'ALL PASS' not in c.stdout.splitlines()[-1]: sys.exit('STOP check_seq %d\n' % st + c.stdout[-1500:])
        print('stage', st, c.stdout.splitlines()[-1], flush=True)
def go(binp, G, W, rep, tag):
    n = G * W * 32; stage1 = not tag.startswith('s0')
    rf = '%s/rays_%s.bin' % (D, tag); np.repeat(np.array([[0, 0, -1, 0, 0, 0, 1, 1e16]], np.float32), n, axis=0).tofile(rf)
    out = '%s/sr_%s_%s.json' % (OUT, tag, ts); fin = '%s/fin_%s.bin' % (D, tag); ref = '%s/ref_%s.bin' % (D, tag)
    if os.path.exists(out): sys.exit('exists ' + out)
    cmd = [sys.executable, RUN, binp, 'file:' + rf, 'zero', out, '--scene', 'tri1', '--warps', str(W), '--blocks', str(G), '--repeat', str(rep), '--dump-final', fin] + HOST + (['--dump-ref', ref] if stage1 else [])
    p = sh(cmd)
    ok = 'exit 0' in p.stdout and 'ttu_one: no error' in p.stdout and ('%d block(s) ran' % G) in p.stdout
    print(tag, 'OK' if ok else 'FAILED', flush=True)
    if not ok: sys.exit('STOP\n' + p.stdout[-2500:] + p.stderr[-1500:])
    j = json.load(open(out)); ms = [l['ms'] for l in j['launches']]
    a = np.fromfile(fin, dtype=np.uint32).reshape(-1, 14)       # R3 R32 R33 R44 R45 R56..R61 R70 R71 slot84 (rounds left)
    assert len(a) == n, (len(a), n)
    unst = int((a[:, 13] == 0x77777777).sum()); left = sorted(set(int(x) for x in a[:, 13]))
    hit = a[:, 0] != 0
    if not stage1:
        C4 = [0, 3, 4, 13]; same = int((a[:, C4] == a[0, C4]).all())   # the four stored words (R3, R44, R45, rounds left); the other columns are host expansion leftovers (compact records of other lanes overlap record 0). Stage 0 has no TTU: R3 holds an input flags word, not a hit flag
        print('  stage 0: records %d unstored %d all rows equal %d R3 %s rounds-left values %s launch ms %s' % (n, unst, same, hex(int(a[0, 0])), left[:6], ms), flush=True)
        if unst or not same or len(left) != 1: sys.exit('STOP stage 0 readback')
        return ms
    r = np.fromfile(ref, dtype=np.uint32).reshape(-1, 8); assert len(r) == n
    rhit = r[:, 0] != 0; both = hit & rhit
    t = np.abs(a[:, 4].view(np.float32)); rt = r[:, 1].view(np.float32); prim = (a[:, 3] & 0x1fffffff).astype(np.int64)
    bad = int((hit != rhit).sum()) + int((both & (t != rt)).sum()) + int((both & (prim != r[:, 2].astype(np.int64))).sum()) + unst
    print('  rays %d hits %d ref hits %d unstored %d wrong %d rounds-left values %s' % (n, int(hit.sum()), int(rhit.sum()), unst, bad, left[:6]), flush=True)
    print('  launch ms', ms, flush=True)
    for f in (fin, ref): os.remove(f)
    if bad: sys.exit('STOP wrong result')
    return ms
if __name__ == '__main__':
    ph = sys.argv[1]; os.makedirs(OUT, exist_ok=True); b0, b1 = D + '/sr_s0', D + '/sr_s1'
    if ph == 'build': build()
    elif ph == 's0': go(b0, 1, 1, 1, 's0_g1_w1'); go(b0, 1, 13, 1, 's0_g1_w13')
    elif ph == 's1w1': go(b1, 1, 1, 1, 's1_g1_w1')
    elif ph == 'big':
        res = {}
        for G in (1, 64, 128): res[G] = go(b1, G, 13, 6, 's1_g%d' % G)
        json.dump(res, open('%s/sr_summary_%s.json' % (OUT, ts), 'x'))
