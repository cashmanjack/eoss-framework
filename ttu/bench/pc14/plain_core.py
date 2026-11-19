"""pc14 (RTX 5090) plain CUDA part of the core set: pointer chase, Memory_1 plain chase, Cache_1 modes. No TTU code.
    python3 plain_core.py WORKDIR STEP...      STEP: chase | mem1 | l1sect | l1cap | l2lat
Rules (CLAUDE.md pc14): notes/pc14_gpucheck.sh before EVERY process (stop if it fails), timeout -k 2 30, GPU 0, raw output to a
timestamped file opened with mode x (never overwritten). Builds into $BUILD (default ../build/pc14) for sm_120."""
import json, os, statistics, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.normpath(os.path.join(HERE, '..', '..', '..'))
BUILD = os.environ.get('BUILD', os.path.join(REPO, '..', 'build', 'pc14')); os.makedirs(BUILD, exist_ok=True)
CHECK = os.path.join(REPO, 'notes', 'pc14_gpucheck.sh')
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES='0', CUDA_DEVICE_ORDER='PCI_BUS_ID')
work = sys.argv[1]; os.makedirs(work, exist_ok=True); TS = time.strftime('%Y%m%dT%H%M%S')


def build(name, src):
    exe = os.path.join(BUILD, name)
    r = subprocess.run(['nvcc', '-O2', '-arch=sm_120', '-o', exe, os.path.join(HERE, '..', src)], capture_output=True, text=True)
    if r.returncode: sys.exit(r.stderr)
    return exe


def run(cmd):
    if subprocess.run([CHECK], capture_output=True).returncode: sys.exit('GPU not free: stopping')
    t = time.time(); r = subprocess.run(['timeout', '-k', '2', '30'] + cmd, capture_output=True, text=True, env=ENV)
    return r, time.time() - t


def med(x): return statistics.median(x)


def t_chase(exe_unused=None):
    exe = build('chase', 'mem_latency/chase.cu'); rows = []
    mb = lambda m: m * 1024
    cases = [(16, 'ca'), (64, 'ca'), (1024, 'ca'), (4096, 'ca'), (16384, 'ca'), (16, 'cg'), (16, 'cross'), (64, 'cross'), (mb(32), 'ca'), (mb(48), 'ca'), (mb(64), 'ca'),
             (mb(80), 'ca'), (mb(88), 'ca'), (mb(90), 'ca'), (mb(92), 'ca'), (mb(93), 'ca'), (mb(94), 'ca'), (mb(95), 'ca'), (mb(96), 'ca'), (mb(97), 'ca'), (mb(98), 'ca'), (mb(99), 'ca'),
             (mb(100), 'ca'), (mb(101), 'ca'), (mb(102), 'ca'), (mb(104), 'ca'), (mb(106), 'ca'), (mb(112), 'ca'), (mb(128), 'ca'), (mb(256), 'ca'), (mb(512), 'ca'), (64, 'cold'), (4096, 'cold')]
    for kb, mode in cases:
        out = '%s/chase_%s_%d_%s.json' % (work, mode, kb, TS)
        r, dt = run([exe, str(kb), mode, out])
        if r.returncode: sys.exit('chase %d %s exit %d (%.1f s) %s' % (kb, mode, r.returncode, dt, r.stderr[-300:]))
        c = json.load(open(out))['cycles_per_load'][1:]; rows.append((kb, mode, med(c), min(c), max(c)))
        print('   chase %8d KB %-5s median %7.1f (min %.1f max %.1f) %.1f s' % (kb, mode, med(c), min(c), max(c), dt), flush=True)
    with open('%s/chase_%s.json' % (work, TS), 'x') as f: json.dump(rows, f)


def t_mem1():
    exe = build('chase', 'mem_latency/chase.cu'); rows = []
    for kb in (16, 20, 24, 28, 32, 48, 64, 80, 96, 112, 128, 160):
        for sm in (0, 98304):
            out = '%s/m1chase_%d_%d_%s.json' % (work, kb, sm, TS)
            r, dt = run([exe, str(kb), 'ca', out] + ([str(sm)] if sm else []))
            if r.returncode: sys.exit('m1 chase %d smem %d exit %d %s' % (kb, sm, r.returncode, r.stderr[-300:]))
            c = json.load(open(out))['cycles_per_load'][1:]; rows.append((kb, sm, med(c)))
            print('   m1 chase %3d KB smem %6d: %7.1f cycles/load' % (kb, sm, med(c)), flush=True)
    with open('%s/mem1_%s.json' % (work, TS), 'x') as f: json.dump(rows, f)


def t_cache(mode):
    exe = build('cache1', 'Cache_1/cache1.cu')
    r, dt = run([exe, mode])
    out = '%s/cache1_%s_%s.txt' % (work, mode, TS)
    with open(out, 'x') as f: f.write(r.stdout)
    print(mode, 'exit', r.returncode, '%.1f s' % dt, len(r.stdout.splitlines()), 'lines', out, r.stderr[-200:], flush=True)
    if r.returncode: sys.exit('STOP')


for s in sys.argv[2:]:
    if s == 'chase': t_chase()
    elif s == 'mem1': t_mem1()
    else: t_cache(s)
