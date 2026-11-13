"""Core set on any GPU (first run: RTX 4070 Ti, io). Config driven: SM count, TPC count and placement come from ttu/config/gpucfg.py.

    core_set.py WORK STEP        STEP: chase | latency | thr1 | thr8 | thr6 | mem2 | mem5 | all
    run with /usr/bin/python3 (numpy); builds use ~/eoss/.venv/bin/python (eoss package)

Safety (same as every TTU run): only the checked traversal group (build_patch verbatim + check_seq ALL PASS on every binary, run.py refuses
otherwise), stage 0 (no TTU) before stage 1 for every layout, smallest launch first (1 warp, 1 launch), every run under run.py's timeout 30,
first idle GPU only. ANY bad exit, wrong result, hang or missing block stops the script (sys.exit); nothing is retried.
Needs /dev/shm/t9/setup3 (host built from ttu/harness/setup3/setup3.cu), setup3_dev.ptx and words.txt.
"""
import json, os, statistics as st, struct, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'config')); import gpucfg  # noqa: E402
CFG = gpucfg.load(); os.environ['TTU_GPU'] = str(CFG.index)
S3 = os.path.join(HERE, '..', '..', 'harness', 'setup3')
VENV = os.path.expanduser('~/eoss/.venv/bin/python')
BASE, D = '/dev/shm/t9/setup3', '/dev/shm/t9c'
NSM, HALF = CFG.sm_count, CFG.half
f32 = lambda u: struct.unpack('<f', struct.pack('<I', u))[0]


def sh(c, **kw): return subprocess.run(c, capture_output=True, text=True, **kw)


def build(name, stage, flags):
    out = '%s/%s_s%d' % (D, name, stage)
    p = sh([VENV, os.path.join(S3, 'build_patch.py'), '--stage', str(stage), '--bin', BASE, '--out', out] + flags)
    if p.returncode: sys.exit(p.stdout + p.stderr)
    c = sh([VENV, os.path.join(S3, 'check_seq.py'), out])
    if 'ALL PASS' not in c.stdout.strip().splitlines()[-1]: sys.exit('check_seq FAILED %s\n%s' % (out, c.stdout[-2500:]))
    return out


def go(binp, rays, out, extra, need_blocks=None):
    p = sh([sys.executable, os.path.join(S3, 'run.py'), binp, rays, 'zero', out] + extra)
    ok = 'exit 0' in p.stdout and ' error' not in p.stdout.replace('no error', '') and 'ttu_one: no error' in p.stdout
    if need_blocks is not None: ok = ok and ('%d block(s) ran' % need_blocks) in p.stdout
    print('   ', os.path.basename(out), 'OK' if ok else 'FAILED', flush=True)
    if not ok: sys.exit('STOP (fault, hang, wrong exit or missing blocks):\n' + p.stdout[-2500:] + p.stderr[-1500:])
    return json.load(open(out))


def ref_ok(j):
    """Every stored warp output vs the OptiX reference (hit, primitive, t): return (checked, wrong). Skips lanes with pending entries."""
    bad = cnt = 0
    for k, wo in enumerate(j['warp_out']):
        if any(wo[x] for x in (56, 57, 58, 59, 60, 61, 70, 71)): continue
        rr = j['ref'][32 * k] if len(j['ref']) > 32 * k else j['ref'][0]; cnt += 1; hit = wo[3]
        rt = f32(int(rr['t'], 16))
        ok = (hit == 0 and rr['hit'] == 0) or (hit == 1 and rr['hit'] == 1 and (wo[44] & 0x1fffffff) == rr['prim'] and abs(-f32(wo[45]) - rt) <= 1e-6 * max(1, rt))
        bad += not ok
    return cnt, bad


def med(x): return st.median(x)


# ---------------------------------------------------------------- chase (plain CUDA)
def t_chase(work):
    exe = '/dev/shm/io/chase'
    p = sh([os.path.expanduser('~/opt/cuda-12.8/bin/nvcc'), '-O2', '-arch=sm_89', '-o', exe, os.path.join(HERE, '..', 'mem_latency', 'chase.cu')])
    if p.returncode: sys.exit(p.stderr)
    cases = [(16, 'ca'), (64, 'ca'), (1024, 'ca'), (4096, 'ca'), (16384, 'ca'), (16, 'cg'), (16, 'cross'), (64, 'cross'),
             (24576, 'ca'), (32768, 'ca'), (40960, 'ca'), (49152, 'ca'), (57344, 'ca'), (65536, 'ca'), (98304, 'ca'), (131072, 'ca'), (524288, 'ca'),
             (64, 'cold'), (4096, 'cold')]
    rows = []
    for kb, mode in cases:
        out = '%s/chase_%s_%d.json' % (work, mode, kb)
        r = sh(['timeout', '-k', '2', '60', exe, str(kb), mode, out], env=CFG.env())
        if r.returncode: sys.exit('chase %d %s exit %d %s' % (kb, mode, r.returncode, r.stderr))
        c = json.load(open(out))['cycles_per_load'][1:]; rows.append((kb, mode, med(c)))
        print('   chase %7d KB %-5s %7.1f cycles/load' % (kb, mode, med(c)), flush=True)
    json.dump(rows, open(work + '/chase.json', 'w'))


# ---------------------------------------------------------------- latency (Latency_1, 2, 6)
SCENES = [('tri1', 'tri1', 'hit'), ('grid:12', 'grid:12', 'tri:6'), ('grid:144', 'grid:144', 'tri:72'), ('grid:200', 'grid:200', 'tri:100'),
          ('grid:2500', 'grid:2500', 'tri:1250'), ('grid:4096', 'grid:4096', 'tri:2048'), ('grid:20736', 'grid:20736', 'tri:10368')]


def t_latency(work):
    s0, s1 = build('clk', 0, ['--clock', '--regs', '96', '--max-out', '93']), build('clk', 1, ['--clock', '--regs', '96', '--max-out', '93'])
    t0, t1 = build('twice', 0, ['--clock', '--twice', '--compact-out']), build('twice', 1, ['--clock', '--twice', '--compact-out'])
    print(' stage 0 first (no TTU), then smallest stage 1 (1 warp, 1 launch)')
    go(s0, 'hit', work + '/lat_s0.json', ['--warps', '1', '--repeat', '2'])
    go(t0, 'hit', work + '/lat_t0.json', ['--warps', '1', '--repeat', '2'])
    j = go(s1, 'hit', work + '/lat_first.json', ['--warps', '1', '--repeat', '1']); assert ref_ok(j)[1] == 0
    go(t1, 'hit', work + '/lat_twice_first.json', ['--warps', '1', '--repeat', '1'])
    res = []
    for name, scene, ray in SCENES:
        row = {'scene': name}
        for mode, flush in (('warm', 0), ('cold', 256)):
            j = go(s1, ray, '%s/lat_%s_%s.json' % (work, name.replace(':', ''), mode), ['--scene', scene, '--warps', '1', '--repeat', '30'] + (['--flush', str(flush)] if flush else []))
            c, b = ref_ok(j); assert b == 0, 'wrong result %s %s' % (name, mode)
            row[mode] = med([e - s for l in j['launches'][1:] for s, e in l['warp_clk']])
        j = go(t1, ray, '%s/lat_%s_twice.json' % (work, name.replace(':', '')), ['--scene', scene, '--warps', '1', '--repeat', '30'])
        row['second'] = med([e - s for l in j['launches'][1:] for s, e in l['warp_clk']])
        res.append(row); print('   ', row, flush=True)
    json.dump(res, open(work + '/latency.json', 'w'))


# ---------------------------------------------------------------- Throughput_1 (one SM, tri1 coherent)
def sweep_stats(j):
    L = j['launches'][1:]; w = j['warps']
    span = [max(e for _, e in l['warp_clk']) - min(s for s, _ in l['warp_clk']) for l in L]
    gaps = []
    for l in L:
        ends = sorted(e for _, e in l['warp_clk']); h = ends[len(ends) // 2:]
        gaps += [b - a for a, b in zip(h, h[1:])]
    return dict(w=w, rpc=med([32 * w / s for s in span]), lat=med([e - s for l in L for s, e in l['warp_clk']]), gap=med(gaps) if gaps else float('nan'))


def t_thr1(work):
    flags = ['--clock', '--sync', '--regs', '96', '--max-out', '93']
    a0, a1 = build('t1', 0, flags), build('t1', 1, flags)
    r0, r1 = build('t1r', 0, ['--clock', '--sync', '--remap', '--compact-out', '--regs', '64', '--max-out', '61']), build('t1r', 1, ['--clock', '--sync', '--remap', '--compact-out', '--regs', '64', '--max-out', '61'])
    go(a0, 'hit', work + '/t1_s0.json', ['--warps', '20', '--repeat', '2']); go(r0, 'hit', work + '/t1r_s0.json', ['--warps', '32', '--repeat', '2'])
    go(a1, 'hit', work + '/t1_first.json', ['--warps', '1', '--repeat', '1'])
    out = []
    for b, ws in ((a1, (1, 2, 4, 8, 16, 20)), (r1, (20, 24, 28, 32))):
        for w in ws:
            j = go(b, 'hit', '%s/t1_%s_w%d.json' % (work, 'r' if b == r1 else 'a', w), ['--warps', str(w), '--repeat', '30'])
            if b != r1: assert ref_ok(j)[1] == 0, 'wrong result'   # the remap build stores only inputs and clocks (--compact-out): no result words
            s = sweep_stats(j); s['layout'] = 'remap' if b == r1 else 'orig'; out.append(s); print('   ', s, flush=True)
    json.dump(out, open(work + '/thr1.json', 'w'))


# ---------------------------------------------------------------- Throughput_8 (full overlap) and Memory_5 (timing): SM filter builds
def t_thr8(work):
    fl = ['--clock', '--sync', '--smfilter', '--gtime', '--regs', '96', '--max-out', '93']
    b0, b1 = build('t8', 0, fl), build('t8', 1, fl)
    base = ['--scene', 'par:256:s', '--warps', '20', '--blocks', str(NSM), '--repeat', '6']
    go(b0, 't2', work + '/t8_s0.json', base + ['--sms', '0,1'], need_blocks=2)
    os.makedirs(work + '/w', exist_ok=True)
    # smallest TTU launch first: one block, one warp
    go(b1, 't2', work + '/t8_first.json', ['--scene', 'par:256:s', '--warps', '1', '--blocks', str(NSM), '--repeat', '1', '--sms', '0,0'], need_blocks=1)
    for p in ('0,0', '2,2', '0,1', '2,3', '10,11', '1,3', '0,2', '2,4'):
        j = go(b1, 't2', '%s/w/pair_%s.json' % (work, p), base + ['--sms', p], need_blocks=1 if p.split(',')[0] == p.split(',')[1] else 2)
        assert ref_ok(j)[1] == 0
    r = sh([sys.executable, os.path.join(HERE, '..', 'Throughput_8', 'full_overlap.py'), work + '/w']); print(r.stdout); open(work + '/thr8.txt', 'w').write(r.stdout)


def t_mem5(work):
    fl = ['--clock', '--smfilter', '--gtime', '--sleepb', '20000']
    b0, b1 = build('m5', 0, fl), build('m5', 1, fl)
    go(b0, 'hit', work + '/m5_s0.json', ['--warps', '1', '--blocks', str(NSM), '--repeat', '2', '--sms', '0,1'], need_blocks=2)
    go(b1, 'hit', work + '/m5_first.json', ['--warps', '1', '--blocks', str(NSM), '--repeat', '1', '--sms', '0,0'], need_blocks=1)
    out = {}
    for name, scene, ray in SCENES[:1] + [SCENES[2], SCENES[4]]:
        for case, sms in (('B_sibling_after_A', '0,1'), ('B_other_TPC_after_A', '0,2'), ('A_alone', '0,0')):
            j = go(b1, ray, '%s/m5_%s_%s.json' % (work, name.replace(':', ''), case), ['--scene', scene, '--warps', '1', '--blocks', str(NSM), '--repeat', '6', '--sms', sms],
                   need_blocks=1 if sms == '0,0' else 2)
            assert ref_ok(j)[1] == 0
            B = int(sms.split(',')[1]); lat = []
            for l in j['launches'][1:]:
                for b in l['per_block']:
                    if b['sm'] == B: lat += [e - s for s, e in b['warp_clk']]
            out['%s/%s' % (name, case)] = med(lat); print('   ', name, case, med(lat), flush=True)
    json.dump(out, open(work + '/mem5.json', 'w'))


# ---------------------------------------------------------------- Throughput_6 coherent: the original method (ttu/bench/Throughput_6/run.py: per block clock spans, 32 warps, one block per SM)
def t_thr6(work, scene=('tri1', 'tri1', 'hit', False), gs=None):
    sys.path.insert(0, os.path.join(HERE, '..', 'Throughput_6')); sys.path.insert(0, os.path.join(HERE, '..', 'Memory_4'))
    import importlib.util
    spec = importlib.util.spec_from_file_location('t6run', os.path.join(HERE, '..', 'Throughput_6', 'run.py')); T6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(T6)
    fl = ['--clock', '--sync', '--gtime', '--remap', '--compact-out', '--regs', '64', '--max-out', '61']
    b0, b1 = build('t6', 0, fl), build('t6', 1, fl)
    nm, sc, ray, _ = scene
    T6.GS = gs or sorted({1, 2, 4, 8, 16, 24, HALF - 1, HALF, HALF + 1, HALF + 15, NSM}); T6.SCENES = [scene]
    T6.go(b0, ray, work + '/t6_s0_first.json', sc, NSM, T6.W)   # stage 0 with every block (all must run)
    T6.go(b1, ray, work + '/t6_first.json', sc, 1, 1)          # smallest TTU launch: 1 block, 1 warp
    T6.go(b1, ray, work + '/t6_g1_w32.json', sc, 1, T6.W)
    for g in T6.GS:
        if g > 1: T6.go(b1, ray, '%s/%s_g%d_w%d.json' % (work, nm, g, T6.W), sc, g, T6.W)
    os.replace(work + '/t6_g1_w32.json', '%s/%s_g1_w%d.json' % (work, nm, T6.W))
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): T6.report(work)
    print(buf.getvalue()); open(work + '/thr6_%s.txt' % nm, 'w').write(buf.getvalue())


# ---------------------------------------------------------------- Memory_1 and Memory_2 (existing script, unchanged): second trace latency vs distinct rays, big and small L1
def t_mem2(work):
    m2 = os.path.join(HERE, '..', 'Memory_2')
    b0, b1 = build('m2', 0, ['--clock', '--twice', '--compact-out']), build('m2', 1, ['--clock', '--twice', '--compact-out'])
    go(b0, 'hit', work + '/m2_s0.json', ['--warps', '1', '--repeat', '2'])
    go(b1, 'hit', work + '/m2_first.json', ['--warps', '1', '--repeat', '1'])
    r = subprocess.run([sys.executable, os.path.join(m2, 'run_latency.py'), b0, b1, work + '/m2'])
    if r.returncode: sys.exit('run_latency failed')
    r = sh([sys.executable, os.path.join(m2, 'run_latency.py'), '--report', work + '/m2']); print(r.stdout); open(work + '/mem2.txt', 'w').write(r.stdout)


def t_thr6par(work): t_thr6(work, ('par64', 'par:64:s', 't2', False), [1, 8, HALF, HALF + 1, NSM])


def t_mem2n(work):
    """Memory_2 as in the triton log (Step 25): ncu RT fills, single session build vs two session build (refetch fraction). Needs ncu (RmProfilingAdminOnly 0)."""
    m2 = os.path.join(HERE, '..', 'Memory_2')
    s0c, s1c = build('m2c', 0, ['--clock']), build('m2c', 1, ['--clock'])
    s0t, s1t = build('m2k', 0, ['--clock', '--twice', '--compact-out']), build('m2k', 1, ['--clock', '--twice', '--compact-out'])
    go(s0c, 'hit', work + '/m2n_s0.json', ['--warps', '1', '--repeat', '2'])
    go(s1c, 'hit', work + '/m2n_first.json', ['--warps', '1', '--repeat', '1'])
    r = subprocess.run([sys.executable, os.path.join(m2, 'run.py'), s0c, s1c, s0t, s1t, work + '/m2n'])
    if r.returncode: sys.exit('Memory_2 run.py failed')
    r = sh([sys.executable, os.path.join(m2, 'run.py'), '--report', work + '/m2n']); print(r.stdout); open(work + '/mem2n.txt', 'w').write(r.stdout)


def t_mem1(work):
    """Memory_1 as in the triton log (Step 11): second trace latency, R distinct random rays (soup:100000), L1 ~90 KB vs reserved shared memory (--smem 98304 -> L1 ~20 KB)."""
    sys.path.insert(0, os.path.join(HERE, '..', 'Memory_4')); from prep import make_rays
    b0, b1 = build('m1', 0, ['--clock', '--twice', '--compact-out']), build('m1', 1, ['--clock', '--twice', '--compact-out'])
    go(b0, 'hit', work + '/m1_s0.json', ['--warps', '1', '--repeat', '2'])
    go(b0, 'hit', work + '/m1_s0_smem.json', ['--warps', '1', '--repeat', '2', '--smem', '98304'])
    go(b1, 'hit', work + '/m1_first.json', ['--warps', '1', '--repeat', '1'])
    exe = '/dev/shm/io/chase'
    for kb in (16, 24, 32, 48, 64, 80, 96, 112, 128, 160):   # plain chase: L1 size without and with the reservation
        for sm in (0, 98304):
            o = '%s/m1chase_%d_%d.json' % (work, kb, sm); r = sh(['timeout', '-k', '2', '60', exe, str(kb), 'ca', o] + ([str(sm)] if sm else []), env=CFG.env())
            if r.returncode: sys.exit('chase exit %d %s' % (r.returncode, r.stderr))
            print('   chase %3d KB smem %5d: %6.1f cycles/load' % (kb, sm, med(json.load(open(o))['cycles_per_load'][1:])), flush=True)
    out = []
    for R in (1, 4, 8, 16, 32, 64, 128, 256):
        W = max(1, R // 32); f = '%s/m1_rays_%d.bin' % (work, R); make_rays(512, 2026)[np.arange(32 * W) % R].tofile(f); row = {'R': R}
        for name, ex in (('l1big', []), ('l1small', ['--smem', '98304'])):
            j = go(b1, 'file:' + f, '%s/m1_%d_%s.json' % (work, R, name), ['--scene', 'soup:100000', '--warps', str(W), '--repeat', '11'] + ex)
            row[name] = med([e - s for l in j['launches'][1:] for s, e in l['warp_clk']])
        row['ratio'] = row['l1small'] / row['l1big']; out.append(row); print('   ', row, flush=True)
    json.dump(out, open(work + '/mem1.json', 'w'))


STEPS = dict(mem1=t_mem1, mem2n=t_mem2n, thr6par=t_thr6par, chase=t_chase, latency=t_latency, thr1=t_thr1, thr8=t_thr8, mem5=t_mem5, thr6=t_thr6, mem2=t_mem2)

if __name__ == '__main__':
    work, step = sys.argv[1:3]
    os.makedirs(work, exist_ok=True); os.makedirs(D, exist_ok=True); os.makedirs('/dev/shm/io', exist_ok=True)
    print('GPU', CFG.index, CFG.name, 'SMs', NSM, 'TPCs', CFG.tpc_count, 'L2 MB', CFG.l2_bytes >> 20, 'driver', CFG.driver, flush=True)
    for s in (STEPS if step == 'all' else [step]):
        print('==', s, flush=True); STEPS[s](work)
